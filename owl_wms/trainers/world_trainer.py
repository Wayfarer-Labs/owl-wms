from ema_pytorch import EMA
from pathlib import Path
import tqdm
import wandb
import gc
import itertools

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..models.world import WorldModel, PromptEncoder
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_samples
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from ..muon import init_muon


class WorldTrainer(BaseTrainer):
    """Trainer for WorldModel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup GLOO
        if self.world_size > 1:
            assert dist.is_initialized()
            self.pg_cpu = dist.new_group(backend="gloo")
        else:
            self.pg_cpu = None

        self.model = WorldModel(self.model_cfg).train()
        self.ema = None
        self.opt = None
        self.total_step_counter = 0

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
        freeze(self.decoder)

        self.prompt_encoder = PromptEncoder()

        self.autocast_ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.total_accum_steps = self.train_cfg.total_accum_steps
        assert self.total_accum_steps % self.world_size == 0
        self.accum_steps_per_device = self.total_accum_steps // self.world_size

    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return
        super().save({
            'model': self.get_raw_model(self.model).state_dict(),
            'ema_model': self.get_raw_model(self.ema.ema_model).state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        })

    def load(self) -> None:
        # VAE
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        # Prompt Encoder
        self.prompt_encoder = self.prompt_encoder.cuda().eval()

        # Online model, EMO, Optimizer
        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        assert self.train_cfg.opt.lower() == "muon"
        self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)

        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        if ckpt:
            state = super().load(ckpt)
            self.get_raw_model(self.ema.ema_model).load_state_dict(state["ema_model"], strict=True)
            self.get_raw_model(self.model).load_state_dict(state["model"], strict=True)
            self.opt.load_state_dict(state["opt"])
            self.total_step_counter = int(state.get("steps", 0))
            del state  # free memory

    @torch.no_grad()
    def update_buffer(self, name: str, value: torch.Tensor, value_ema: torch.Tensor | None = None):
        """Set the buffer `name` (e.g. 'core.transformer.foo') across ranks and EMA."""
        online = self.model.module if isinstance(self.model, DDP) else self.model
        buf_online = online.get_buffer(name)
        buf_ema = self.ema.ema_model.get_buffer(name)

        if self.rank == 0:
            buf_online.copy_(value.to(buf_online))
        if self.world_size > 1:
            dist.broadcast(buf_online, 0)

        buf_ema.copy_(buf_online)

    def prep_batch(self, batch):
        """Move to cuda, and if necessary use encoder to convert rgb to latent (x)"""
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        assert "rgb" not in batch, "rgb not supported, pass latents"

        if "mouse" in batch or "buttons" in batch:
            assert "controller_inputs" not in batch, "passed mouse or button, but already have `controller_inputs`"
            xs = tuple(filter(lambda x: x is not None, [batch.pop("mouse"), batch.pop("buttons")]))
            batch["controller_inputs"] = torch.cat(xs, dim=-1)

        if "prompt" in batch:
            assert "prompt_emb" not in batch, "passed prompt to convert, but already have batch item `prompt_emb`"
            batch["prompt_emb"] = self.prompt_encoder(batch.pop("prompt"))

        batch["x"] = (batch["x"] / self.train_cfg.vae_scale).bfloat16()

        return batch

    def train_loader(self):
        return get_loader(
            self.train_cfg.data_id,
            **self.train_cfg.data_kwargs
        )

    def eval_loader(self):
        per_dev_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size
        return get_loader(
            self.train_cfg.sample_data_id,
            batch_size=per_dev_samples,
            **self.train_cfg.sample_data_kwargs
        )

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        self.load()

        timer = Timer()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        # Dataset setup
        train_loader = self.train_loader()
        eval_loader = iter(self.eval_loader())

        # TODO: clean up sampler use
        self.sampler_only_return_generated = self.train_cfg.sampler_kwargs.pop("only_return_generated")
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        for epoch in range(self.train_cfg.epochs):
            for mini_batches in tqdm.tqdm(
                    itertools.batched(train_loader, n=self.accum_steps_per_device),
                    total=len(train_loader) // self.accum_steps_per_device,
                    disable=self.rank != 0,
                    desc=f"Epoch: {epoch}"
            ):
                train_loss = self.train_step(mini_batches)
                metrics.log('train_loss', train_loss)

                self.ema.update()

                self.log_step(metrics, timer, eval_loader, sampler)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    self.save()

                self.barrier()

    def train_step(self, mini_batches):
        # fwd-bwd over all mini batches
        loss_sum = 0
        for batch in mini_batches:
            batch = self.prep_batch(batch)
            loss = self.fwd_step(batch)
            loss.backward()
            loss_sum += loss.item()

        # optimizer step
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        return loss_sum

    @torch.compile
    def fwd_step(self, batch):
        return self.conditional_flow_matching_loss(self.model, **batch) / self.accum_steps_per_device

    def conditional_flow_matching_loss(self, model, x, **kw):
        """
        x0: [B, N, C, H, W] clean latents (timestep 0.0)
        """
        x0 = x
        B, N = x0.size(0), x0.size(1)

        with torch.no_grad():
            ts = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()
            x1 = torch.randn_like(x0)  # gaussian @ timestep 1.0
            x_t = x0 + (x1 - x0) * ts.view(B, N, 1, 1, 1)  # lerp to noise level @ ts
            v_target = x1 - x0

        with self.autocast_ctx:
            v_pred = model(x_t, ts, **kw)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def log_step(self, metrics, timer, sample_loader, sampler):
        wandb_dict = metrics.pop()
        wandb_dict['time'] = timer.hit()
        timer.reset()

        # eval / sample step
        if self.total_step_counter % self.train_cfg.sample_interval == 0:
            eval_wandb_dict = self.eval_step(sample_loader, sampler)
            if self.rank == 0:
                wandb_dict.update(eval_wandb_dict)

        if self.rank == 0:
            wandb.log(wandb_dict)

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        """Gather *t* from every rank onto rank 0 and return concatenated copy."""
        if t is None:
            return None
        if self.pg_cpu is None:
            assert self.world_size == 1
            return t.detach().cpu()
        tc = t.detach().cpu()
        if self.rank == 0:
            bufs = [torch.empty_like(tc) for _ in range(self.world_size)]
            dist.gather(tc, gather_list=bufs, dst=0, group=self.pg_cpu)
            return torch.cat(bufs, dim=dim)
        else:
            dist.gather(tc, dst=0, group=self.pg_cpu)

    def eval_step(self, sample_loader, sampler):
        ema_model = self.get_module(ema=True)

        # ---- Generate Samples ----
        eval_batch = self.prep_batch(next(sample_loader))
        vid, prompt_emb, controller_inputs = [eval_batch.get(k) for k in ("x", "prompt_emb", "controller_inputs")]

        if self.train_cfg.num_seed_frames:
            vid = vid[:, :self.train_cfg.num_seed_frames]

        with self.autocast_ctx:
            latent_vid = sampler(
                ema_model, vid, prompt_emb, controller_inputs, self.train_cfg.num_generated_frames
            )

        if self.sampler_only_return_generated:
            latent_vid, controller_inputs = (
                x[:, vid.size(1):] if x is not None else None for x in (latent_vid, controller_inputs)
            )

        video_out = self.decode_fn(latent_vid * self.train_cfg.vae_scale)

        # ---- Optionally Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Generate Media Artifacts ----
        video_out, controller_inputs = map(self._gather_concat_cpu, (video_out, controller_inputs))

        # TODO: clean this hack
        mouse, btn = None, None
        if eval_batch["controller_inputs"] is not None:
            mouse, btn = map(
                self._gather_concat_cpu,
                torch.split(eval_batch["controller_inputs"], [2, 11], dim=-1)
            )
        eval_wandb_dict = to_wandb_samples(video_out, mouse, btn, fps=60) if self.rank == 0 else None

        # ---- Eval Loss ----
        eval_loss = self.conditional_flow_matching_loss(ema_model, **eval_batch)
        if self.world_size > 1:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            eval_loss /= self.world_size
        if self.rank == 0:
            eval_wandb_dict["eval_loss"] = eval_loss.item()

        dist.barrier()

        return eval_wandb_dict
