from ema_pytorch import EMA
from pathlib import Path
import tqdm
import wandb
import itertools

import torch
import torch.nn as nn
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


# Prevent eager mode by increasing recompile limit
import torch._dynamo as dynamo
dynamo.config.recompile_limit = 32


# speed up fp32
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# TODO: replace with itertools.batched in python3.13
batched = lambda it, n: iter(lambda it=iter(it): tuple(itertools.islice(it, n)), ())


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

        self.prompt_encoder = PromptEncoder(self.train_cfg.prompt_encoder_model_id)

        self.autocast_ctx = torch.amp.autocast('cuda', torch.bfloat16)

        assert self.train_cfg.total_accum_steps % self.world_size == 0
        self.accum_steps_per_device = self.train_cfg.total_accum_steps // self.world_size

    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return
        super().save({
            'model': self.get_raw_model(self.model).state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        })

    def load(self) -> None:
        # Static immutable models
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        self.prompt_encoder = self.prompt_encoder.cuda().eval()

        # Online model, EMA, Optimizer
        self.model = self.model.cuda()
        self.quantize(self.model)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        raw = self.get_raw_model(self.model)
        self.ema = EMA(raw, beta=0.999, update_after_step=0, update_every=1, include_online_model=False)

        assert self.train_cfg.opt.lower() == "muon"
        self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)

        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        if ckpt:
            state = super().load(ckpt)

            self.get_raw_model(self.model).load_state_dict(state["model"], strict=True)
            self.ema.load_state_dict(state["ema"], strict=True)
            self.opt.load_state_dict(state["opt"])
            self.total_step_counter = int(state.get("steps", 0))

            del state  # free memory

    def quantize(self, model):
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

        def mx_friendly_linears(mod, fqn):
            return isinstance(mod, nn.Linear) and (mod.in_features % 32 == 0)

        cfg = Float8DynamicActivationFloat8WeightConfig(
            round_scales_to_power_of_2=True,
            pad_inner_dim=False,
            emulate=False,
        )

        quantize_(self.model, cfg, filter_fn=mx_friendly_linears)


    @torch.no_grad()
    def set_buffer(self, model: torch.nn.Module, name: str, value: torch.Tensor):
        """Set buffer `name` on `model` (supports dotted paths), in-place; broadcast same buffer to all devices"""
        mod = model.module if isinstance(model, DDP) else model
        sub, sep, buf = name.rpartition(".")
        target = (mod.get_submodule(sub) if sep else mod).get_buffer(buf if sep else name)
        if self.rank == 0:
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        if self.world_size > 1:
            dist.broadcast(target, src=0)

    @torch.no_grad()
    def attn_window_update(self):
        # step -> (local_window, global_window)
        online_updates = {0: (1, 1), 50000: (2, 4), 100000: (3, 9), 150000: (4, 16)}
        ema_updates = online_updates
        # TODO: Need to assert that the final step window is equal to model config

        def apply(model, local_window, global_window):
            self.set_buffer(model, "transformer.local_window", torch.tensor(local_window, dtype=torch.int32))
            self.set_buffer(model, "transformer.global_window", torch.tensor(global_window, dtype=torch.int32))

        step = self.total_step_counter
        if step in online_updates:
            apply(self.model, *online_updates[step])
        if step in ema_updates:
            apply(self.ema.ema_model, *ema_updates[step])

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

        # scale latents
        batch["x"] = (batch["x"] / self.train_cfg.vae_scale).bfloat16()

        return batch

    def train_loader(self):
        return get_loader(
            self.train_cfg.data_id,
            **self.train_cfg.data_kwargs
        )

    def eval_loader(self):
        return get_loader(
            self.train_cfg.eval_data_id,
            **self.train_cfg.eval_data_kwargs
        )

    def eval_sample_loader(self):
        return get_loader(
            self.train_cfg.sample_data_id,
            **self.train_cfg.sample_data_kwargs
        )

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        self.load()

        # Dataset setup
        self.train_loader = self.train_loader()
        self.sample_loader = iter(self.eval_sample_loader())

        timer = Timer()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        # TODO: clean up, sampler use
        self.sampler_only_return_generated = self.train_cfg.sampler_kwargs.pop("only_return_generated")
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        for epoch in range(self.train_cfg.epochs):
            for mini_batches in tqdm.tqdm(
                    batched(self.train_loader, n=self.accum_steps_per_device),
                    total=len(self.train_loader) // self.accum_steps_per_device,
                    disable=self.rank != 0,
                    desc=f"Epoch: {epoch}"
            ):
                # self.attn_window_update()

                train_loss = self.train_step(mini_batches)
                metrics.log('train_loss', train_loss)
                self.ema.update()

                self.log_step(metrics, timer, sampler)

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
        x0: [B, N, C, H, W] clean latents (sigma=0.0)
        """
        x0 = x
        B, N = x0.size(0), x0.size(1)

        with torch.no_grad():
            # sigma = torch.rand(B, N, device=x0.device, dtype=x0.dtype)  # Optional: U(0,1)
            sigma = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()  # LogitNormal(0,1)
            eps = torch.finfo(sigma.dtype).eps
            sigma = sigma.clamp(eps, 1 - eps)

            # Sequence of "current frames" noised at random uniform levels
            x1 = torch.randn_like(x0)  # gaussian @ sigma 1.0
            x_t = x0 + (x1 - x0) * sigma.view(B, N, 1, 1, 1)  # lerp to noise level @ sigma
            v_target = x1 - x0

            frame_timestamp = getattr(model, "module", model).get_frame_timestamps(kw.pop("fps"), N, x0.device)  # [B, N]

            if getattr(self.train_cfg, "inference_matching", False):
                # Construct sequence of slightly-noised previous frames
                sigma_p = x0.new_full((B, N), self.train_cfg.noise_prev)
                x1_p = torch.randn_like(x0)
                x_p = x0 + (x1_p - x0) * sigma_p.view(B, N, 1, 1, 1)

                x_t = torch.cat([x_t, x_p], dim=1),
                sigma = torch.cat([sigma, sigma_p], dim=1),

                # mask for static / sampled noise
                curr_frame_mask = (torch.arange(N * 2, device=x0.device) < N).repeat(B, 1)

                # Repeat doc_ids / frame timestamps
                frame_timestamp = frame_timestamp.repeat(1, 2)  # [B, 2N]
                if "doc_id" in kw and kw["doc_id"] is not None:
                    kw["doc_id"] = kw["doc_id"].repeat(1, 2)  # [B, 2N]

            else:
                curr_frame_mask = None

        with self.autocast_ctx:
            v_pred = model(
                x_t, sigma,
                curr_frame_mask=curr_frame_mask,
                frame_timestamp=frame_timestamp,
                **kw
            )
            v_pred = v_pred[:, :N]  # only compute loss on x_t branch

            # Experimental: Don't predict first frame
            # v_pred, v_target = v_pred[:, 1:], v_target[:, 1:]
            # ########

        if getattr(self.train_cfg, "ELBO_loss", False):
            w = (sigma / (1.0 - sigma)).pow(2).view(B, N, 1, 1, 1)
            w = w / (w.mean().detach() + 1e-12)
            return ((v_pred - v_target) ** 2 * w).mean()
        else:
            return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def log_step(self, metrics, timer, sampler):
        wandb_dict = metrics.pop()
        wandb_dict['time'] = timer.hit()
        timer.reset()

        # eval / sample step
        if self.total_step_counter % self.train_cfg.sample_interval == 0:
            eval_wandb_dict = self.eval_step(sampler)
            sample_wandb_dict = self.sample_step(sampler)
            if self.rank == 0:
                wandb_dict.update(eval_wandb_dict)
                wandb_dict.update(sample_wandb_dict)

        if self.rank == 0:
            wandb.log(wandb_dict, step=self.total_step_counter)

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

    def eval_step(self, sampler, do_sample=True):
        ema_model = self.ema.ema_model
        ema_model.eval()

        eval_wandb_dict = {}

        # Always reset the eval-loss DataLoader so each eval starts from the beginning
        ema_val_loss = self.aggregate_eval_loss(ema_model, self.eval_loader())
        if self.rank == 0:
            eval_wandb_dict = {"eval_loss": ema_val_loss}

        dist.barrier()

        return eval_wandb_dict

    def aggregate_eval_loss(self, model, loader):
        target_n = getattr(self.train_cfg, "n_eval_loss_samples", 0) // self.world_size
        if not target_n:
            dist.barrier()
            return None

        num, den = 0.0, 0.0
        loss_iter = iter(loader)
        while den < target_n:
            b = self.prep_batch(next(loss_iter))
            loss = self.conditional_flow_matching_loss(model, **b)
            elems = b["x"].numel()
            num += loss.item() * elems
            den += elems
        if self.world_size > 1:
            t = torch.tensor([num, den], device=f"cuda:{self.local_rank}", dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            num, den = float(t[0].item()), float(t[1].item())

        return num / max(1.0, den)

    def sample_step(self, sampler):
        ema_model = self.ema.ema_model
        ema_model.eval()

        # ---- Batch & labels ----
        eval_batch = self.prep_batch(next(self.sample_loader))
        vid, prompt_emb, controller_inputs = [eval_batch.get(k) for k in ("x", "prompt_emb", "controller_inputs")]
        if self.train_cfg.num_seed_frames:
            vid = vid[:, :self.train_cfg.num_seed_frames]

        lw = int(ema_model.transformer.local_window.item())
        gw = int(ema_model.transformer.global_window.item())
        fps = int(eval_batch["fps"])

        def mk_labels(fps_val: int, n: int):
            base = {"noise_prev": self.train_cfg.noise_prev, "local attn": lw, "global attn": gw}
            return [{"fps": fps_val, **base} for _ in range(n)]

        # ---- Generate ----
        with self.autocast_ctx:
            latent_vid = sampler(
                ema_model, vid, prompt_emb, controller_inputs,
                fps=eval_batch["fps"], num_frames=self.train_cfg.num_generated_frames,
                noise_prev=self.train_cfg.noise_prev
            )

        if self.sampler_only_return_generated:
            latent_vid = None if latent_vid is None else latent_vid[:, vid.size(1):]
            controller_inputs = None if controller_inputs is None else controller_inputs[:, vid.size(1):]

        video_out = self.decode_fn(latent_vid * self.train_cfg.vae_scale)

        # ---- Optional latent artifact ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            lat_cpu = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                out_dir = Path(self.train_cfg.eval_sample_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(lat_cpu, out_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Gather & log ----
        video_out = self._gather_concat_cpu(video_out)
        ci = self._gather_concat_cpu(controller_inputs)
        mouse, btn = (None, None) if ci is None else torch.split(ci, [2, 11], dim=-1)

        num_gt_frames = 0 if self.sampler_only_return_generated else self.train_cfg.num_seed_frames

        if self.rank == 0:
            n_out = 0 if video_out is None else video_out.size(0)
            labels_out = mk_labels(fps, n_out)
            return to_wandb_samples(
                video_out, mouse, btn, labels=labels_out, num_gt_frames=num_gt_frames
            )
        return None
