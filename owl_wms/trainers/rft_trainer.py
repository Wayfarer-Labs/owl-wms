from ema_pytorch import EMA
from pathlib import Path
import tqdm
import wandb
import gc
import re

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_samples
from ..utils import batch_permute_to_length
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn


class RFTTrainer(BaseTrainer):
    """
    Trainer for rectified flow transformer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg).train()

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.scheduler = None

        self.total_step_counter = 0
        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        freeze(self.decoder)

    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return

        save_dict = {
            'model': self.get_raw_model(self.model).state_dict(),
            'ema': self.get_raw_model(self.ema).state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self) -> None:
        """Build runtime objects and optionally restore a checkpoint."""
        # ----- model & helpers -----
        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        state = None
        if ckpt:
            state = super().load(ckpt)

            # Allow legacy checkpoints: strip module and _orig_mod
            pat = r'^(?:(?:_orig_mod\.|module\.)+)?([^.]+\.)?(?:(?:_orig_mod\.|module\.)+)?'
            state["model"] = {re.sub(pat, r'\1', k): v for k, v in state["model"].items()}
            state["ema"] = {re.sub(pat, r'\1', k): v for k, v in state["ema"].items()}

            self.model.load_state_dict(state["model"], strict=True)
            self.total_step_counter = state.get("steps", 0)

        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model
        self.model = torch.compile(self.model)

        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        # ----- EMA, optimiser, scheduler -----
        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler:
            sched_cls = get_scheduler_cls(self.train_cfg.scheduler)
            self.scheduler = sched_cls(self.opt, **self.train_cfg.scheduler_kwargs)

        # ----- optional checkpoint restore -----
        if ckpt:
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])
            if self.scheduler and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])

        del state

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

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        if save_dict:
            self.ema.load_state_dict(save_dict['ema'])
            self.opt.load_state_dict(save_dict['opt'])
            if self.scheduler is not None and 'scheduler' in save_dict:
                self.scheduler.load_state_dict(save_dict['scheduler'])
            self.total_step_counter = save_dict['steps']
            del save_dict

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        # if self.rank == 0:
        #     wandb.watch(self.get_module(), log='all')

        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            **self.train_cfg.data_kwargs
        )

        n_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size  # round up to next world_size
        sample_loader = get_loader(
            self.train_cfg.sample_data_id,
            n_samples,
            **self.train_cfg.sample_data_kwargs
        )
        sample_loader = iter(sample_loader)

        if self.train_cfg.data_id == "cod_s3_mixed":
            loader.dataset.sleep_until_queues_filled()
            self.barrier()

        self.sampler_only_return_generated = self.train_cfg.sampler_kwargs.pop("only_return_generated")
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm.tqdm(loader, total=len(loader), disable=self.rank != 0, desc=f"Epoch: {epoch}"):
                batch = [t.cuda() for t in batch]
                batch[0] = batch[0] / self.train_cfg.vae_scale  # vid

                with ctx:
                    loss = self.model(*batch) / accum_steps
                    loss.backward()

                metrics.log('diffusion_loss', loss)
                del loss, batch

                local_step += 1
                if local_step % accum_steps == 0:

                    # Optimizer updates
                    if self.train_cfg.opt.lower() != "muon":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        timer.reset()

                        # Sampling commented out for now
                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                eval_wandb_dict = self.eval_step(sample_loader, sampler)
                                gc.collect()
                                torch.cuda.empty_cache()
                                if self.rank == 0:
                                    wandb_dict.update(eval_wandb_dict)

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        self.save()

                    self.barrier()

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        """Gather *t* from every rank onto rank 0 and return concatenated copy."""
        if self.world_size == 1:
            return t.cpu()
        if self.rank == 0:
            parts = [t.cpu()]
            scratch = torch.empty_like(t)
            for src in range(1, self.world_size):
                dist.recv(scratch, src=src)
                parts.append(scratch.cpu())
            return torch.cat(parts, dim=dim)
        dist.send(t, dst=0)

    def eval_step(self, sample_loader, sampler):
        ema_model = self.get_module(ema=True).core

        # ---- Generate Samples ----
        vid, mouse, btn = [x.cuda() for x in next(sample_loader)]
        mouse, button = batch_permute_to_length(mouse, btn, sampler.num_frames + vid.size(1))
        vid = vid / self.train_cfg.vae_scale

        latent_vid = sampler(ema_model, vid, mouse, button)

        if self.sampler_only_return_generated:
            latent_vid, mouse, button = (x[:, vid.size(1):] for x in (latent_vid, mouse, button))

        video_out = self.decode_fn(latent_vid * self.train_cfg.vae_scale) if self.decode_fn is not None else None

        # ---- Optionally Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Generate Media Artifacts ----
        video_out, mouse, button = map(self._gather_concat_cpu, (video_out, mouse, button))
        eval_wandb_dict = to_wandb_samples(video_out, mouse, button) if self.rank == 0 else None
        dist.barrier()

        return eval_wandb_dict
