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


class WanEncoderDecoder:
    """
    Minimal encode/decode wrapper for Diffusers' AutoencoderKLWan.

    encode(rgb)  : [T,3,H,W] | [B,3,T,H,W] | [B,T,3,H,W] -> [B,T,C,H,W]  (model space)
    decode(z)    : [B,T,C,H,W] | [B,C,T,H,W] -> [B,Tpix,3,H,W]           (pixels)
                    where Tpix = 1 + 4 * (T - 1)
    """
    def __init__(self, vae, batch_size: int = 2, dtype=torch.float32):
        vae.decoder = torch.compile(vae.decoder)
        vae.encoder = torch.compile(vae.encoder)
        self.vae = vae.eval()

        self.bs  = int(batch_size)
        self.dt  = dtype
        cfg = vae.config
        self.sf  = float(getattr(cfg, "scaling_factor", 1.0))
        self.m   = getattr(cfg, "latents_mean", None)
        self.s   = getattr(cfg, "latents_std",  None)
        self.C   = int(getattr(cfg, "z_dim", getattr(cfg, "latent_channels", 16)))

    # ---------- helpers ----------
    def _dev(self):
        return next(self.vae.parameters()).device

    def _rgb_to_b3thw(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [T,3,H,W], [B,3,T,H,W], [B,T,3,H,W]  ->  [B,3,T,H,W]
        if x.ndim == 4:                     # [T,3,H,W]
            if x.shape[1] != 3: raise ValueError(f"Expected [T,3,H,W], got {tuple(x.shape)}")
            return x.permute(1, 0, 2, 3).unsqueeze(0)
        if x.ndim == 5 and x.shape[1] == 3: # [B,3,T,H,W]
            return x
        if x.ndim == 5 and x.shape[2] == 3: # [B,T,3,H,W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"RGB must be [T,3,H,W] or [B,3,T,H,W] or [B,T,3,H,W]; got {tuple(x.shape)}")

    def _to_model_space(self, z_vae: torch.Tensor) -> torch.Tensor:
        if self.m is not None and self.s is not None:
            mean = torch.as_tensor(self.m, device=z_vae.device, dtype=z_vae.dtype).view(1, -1, 1, 1, 1)
            std  = torch.as_tensor(self.s, device=z_vae.device, dtype=z_vae.dtype).view(1, -1, 1, 1, 1)
            return (z_vae - mean) * (self.sf / std)
        return z_vae * self.sf

    def _to_vae_space(self, z_model: torch.Tensor) -> torch.Tensor:
        if self.m is not None and self.s is not None:
            mean = torch.as_tensor(self.m, device=z_model.device, dtype=z_model.dtype).view(1, -1, 1, 1, 1)
            std  = torch.as_tensor(self.s, device=z_model.device, dtype=z_model.dtype).view(1, -1, 1, 1, 1)
            return (z_model / self.sf) * std + mean
        return z_model / self.sf

    def _pad_to_4k_plus_1(self, z_bcthw: torch.Tensor) -> tuple[torch.Tensor, int]:
        # WAN VAE expects latent T such that output frames = 1 + 4*(T-1)
        T = z_bcthw.shape[2]
        target = ((T - 1 + 3) // 4) * 4 + 1
        if target == T:
            return z_bcthw, T
        pad = target - T
        z_pad = torch.cat([z_bcthw, z_bcthw[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        return z_pad, T

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB -> model-space latents, returning [B,T,C,H,W]
        """
        x = self._rgb_to_b3thw(rgb)
        # Optional normalization if dataset is uint8 [0,255]
        if x.dtype == torch.uint8:
            x = x.to(torch.float32).div_(127.5).sub_(1.0)
        x = x.to(self._dev(), dtype=self.dt)
        parts = []
        for x_chunk in x.split(self.bs, dim=0):
            z_vae = self.vae.encode(x_chunk, return_dict=True).latent_dist.sample()  # [b,C,T,H,W]
            parts.append(self._to_model_space(z_vae))
        z = torch.cat(parts, dim=0)  # [B,C,T,H,W]
        return z.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Model-space latents -> RGB, returning [B,Tpix,3,H,W]
        """
        # Normalize to [B,C,T,H,W] for VAE
        if z.ndim != 5:
            raise ValueError(f"Latents must be 5D, got {tuple(z.shape)}")
        # Deterministic: accept [B,T,C,H,W] or [B,C,T,H,W] -> [B,C,T,H,W]
        if z.shape[1] == self.C:
            z_bcthw = z
        elif z.shape[2] == self.C:
            z_bcthw = z.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Neither dim1 nor dim2 equals latent C={self.C}; got {tuple(z.shape)}")
        z_bcthw = z_bcthw.to(self._dev(), dtype=self.dt)

        # Pad to valid temporal length, convert to VAE space, decode
        z_bcthw, orig_T = self._pad_to_4k_plus_1(z_bcthw)
        parts = []
        for z_chunk in z_bcthw.split(self.bs, dim=0):
            z_in = self._to_vae_space(z_chunk.to(torch.float32))  # VAE in fp32
            pix = self.vae.decode(z_in, return_dict=True).sample  # [b,3,Tpix,H,W]
            parts.append(pix)
        y = torch.cat(parts, dim=0)  # [B,3,Tpix,H,W]

        # Trim to frames implied by original latent length and return [B,Tpix,3,H,W]
        want_Tpix = 1 + 4 * (orig_T - 1)
        if y.shape[2] >= want_Tpix:
            y = y[:, :, :want_Tpix]
        return y.permute(0, 2, 1, 3, 4).contiguous()


class CraftTrainer(BaseTrainer):
    """Trainer for Craft model"""
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

        from diffusers import AutoencoderKLWan
        self.decoder = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        freeze(self.decoder)

        self.autocast_ctx = torch.amp.autocast('cuda', torch.bfloat16)

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
            state["ema_model"] = {
                k.replace("module.", "").replace("_orig_mod.", "").replace("ema_model.", ""): v
                for k, v in state["ema"].items() if k.startswith("ema_model.")
            }
            state["ema_model"] = {re.sub(pat, r'\1', k): v for k, v in state["ema_model"].items()}

            self.model.load_state_dict(state["model"], strict=True)
            self.total_step_counter = state.get("steps", 0)

        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model

        self.model = torch.compile(self.model)

        self.decoder = self.decoder.cuda().eval()
        self.encoder_decoder = WanEncoderDecoder(self.decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        # ----- optimiser, scheduler -----
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler:
            sched_cls = get_scheduler_cls(self.train_cfg.scheduler)
            self.scheduler = sched_cls(self.opt, **self.train_cfg.scheduler_kwargs)

        # ----- optional checkpoint restore -----
        if ckpt:
            if self.world_size > 1:
                self.ema.ema_model.module.load_state_dict(state["ema_model"])
            else:
                self.ema.ema_model.load_state_dict(state["ema_model"])
            self.opt.load_state_dict(state["opt"])
            if self.scheduler and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])

        del state

    def prep_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
            if getattr(self.train_cfg, "raw_rgb", False):
                batch[0] = self.encoder_decoder.encode(batch[0])
        else:
            batch = batch.cuda()
            if getattr(self.train_cfg, "raw_rgb", False):
                batch = self.encoder_decoder.encode(batch)
        batch[0] = batch[0].bfloat16()
        return batch

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

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            **self.train_cfg.data_kwargs
        )

        n_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size  # round up to next world_size
        sample_loader = iter(get_loader(
            self.train_cfg.sample_data_id,
            n_samples,
            **self.train_cfg.sample_data_kwargs
        ))

        self.sampler_only_return_generated = self.train_cfg.sampler_kwargs.pop("only_return_generated")
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm.tqdm(loader, total=len(loader), disable=self.rank != 0, desc=f"Epoch: {epoch}"):
                train_steps = local_step // accum_steps

                batch = self.prep_batch(batch)
                loss = self.fwd_step(batch, train_steps)
                loss = loss / accum_steps
                loss.backward()

                metrics.log('loss', loss)

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

                    self.log_step(metrics, timer, sample_loader, sampler)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        self.save()

                    self.barrier()

    def fwd_step(self, batch, train_step):
        with self.autocast_ctx:
            loss = self.model(*batch)
        return loss

    @torch.no_grad()
    def log_step(self, metrics, timer, sample_loader, sampler):
        wandb_dict = metrics.pop()
        wandb_dict['time'] = timer.hit()
        timer.reset()

        # Sampling commented out for now
        if self.total_step_counter % self.train_cfg.sample_interval == 0:
            eval_wandb_dict = self.eval_step(sample_loader, sampler)
            gc.collect()
            torch.cuda.empty_cache()
            if self.rank == 0:
                wandb_dict.update(eval_wandb_dict)

        if self.rank == 0:
            wandb.log(wandb_dict)

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        """Gather *t* from every rank onto rank 0 and return concatenated copy."""
        if t is None:
            return None
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
        eval_batch = self.prep_batch(next(sample_loader))
        if len(eval_batch) == 3:
            vid, mouse, btn = eval_batch
            mouse, btn = batch_permute_to_length(mouse, btn, sampler.num_frames + vid.size(1))
        else:
            vid, = eval_batch
            mouse, btn = None, None

        with self.autocast_ctx:
            latent_vid = sampler(ema_model, vid, mouse, btn)

        if self.sampler_only_return_generated:
            latent_vid, mouse, btn = (x[:, vid.size(1):] if x is not None else None for x in (latent_vid, mouse, btn))

        video_out = self.encoder_decoder.decode(latent_vid.float()) if self.encoder_decoder is not None else None
        if getattr(self.train_cfg, "raw_rgb", False):
            mouse = mouse.repeat_interleave(4, dim=1) if mouse is not None else None
            btn = btn.repeat_interleave(4, dim=1) if btn is not None else None

        # ---- Optionally Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Generate Media Artifacts ----
        video_out, mouse, btn = map(self._gather_concat_cpu, (video_out, mouse, btn))
        eval_wandb_dict = to_wandb_samples(video_out, mouse, btn, fps=24) if self.rank == 0 else None
        dist.barrier()

        return eval_wandb_dict
