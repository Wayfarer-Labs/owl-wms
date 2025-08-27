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


class WorldTrainer(BaseTrainer):
    """Trainer for WorldModel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = WorldModel(self.model_cfg).train()
        self.ema = None
        self.opt = None
        self.total_step_counter = 0

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        from diffusers import AutoencoderKLWan
        self.decoder = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        freeze(self.decoder)
        self.prompt_encoder = PromptEncoder()

        self.autocast_ctx = torch.amp.autocast('cuda', torch.bfloat16)

        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        self.accum_steps = max(1, accum_steps)

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
        self.decoder = self.decoder.cuda().eval()
        self.encoder_decoder = WanEncoderDecoder(self.decoder, self.train_cfg.vae_batch_size)
        # Prompt Encoder
        self.prompt_encoder = self.prompt_encoder.cuda().eval()

        # Online model, EMO, Optimizer
        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.model = torch.compile(self.model)

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
        if "rgb" in batch:
            assert "x" not in batch, "passed rgb to convert, but already have batch item `x` (latents)"
            batch["x"] = self.encoder_decoder.encode(batch.pop("rgb"))
        if "mouse" in batch or "buttons" in batch:
            assert "controller_inputs" not in batch, "passed mouse or button, but already have `controller_inputs`"
            xs = tuple(filter(lambda x: x is not None, [batch.pop("mouse"), batch.pop("buttons")]))
            batch["controller_inputs"] = torch.cat(xs, dim=-1)

        ####
        # Hack: REMOVE
        # Subsample controller inputs
        if batch["controller_inputs"].size(1) != batch["x"].size(1):
            batch["controller_inputs"] = batch["controller_inputs"][:, ::4, :]
        ####

        if "prompt" in batch:
            assert "prompt_emb" not in batch, "passed prompt to convert, but already have batch item `prompt_emb`"
            batch["prompt_emb"] = self.prompt_encoder(batch.pop("prompt"))
        return batch

    def train_loader(self):
        return get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            **self.train_cfg.data_kwargs
        )

    def eval_loader(self):
        n_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size  # round up to next world_size
        return get_loader(
            self.train_cfg.sample_data_id,
            n_samples,
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
                    itertools.batched(train_loader, n=self.accum_steps),
                    total=len(train_loader) // self.accum_steps,
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
            loss = self.fwd_step(batch) / self.accum_steps
            loss.backward()
            loss_sum += loss.item()

        # optimizer step
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        return loss_sum / self.accum_steps

    def fwd_step(self, batch):
        return self.conditional_flow_matching_loss(**batch)

    @torch.compile
    def conditional_flow_matching_loss(self, x, **kw):
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
            v_pred = self.model(x_t, ts, **kw)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def log_step(self, metrics, timer, sample_loader, sampler):
        wandb_dict = metrics.pop()
        wandb_dict['time'] = timer.hit()
        timer.reset()

        # eval / sample step
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
        ema_model = self.get_module(ema=True)

        # ---- Generate Samples ----
        eval_batch = self.prep_batch(next(sample_loader))
        vid, prompt_emb, controller_inputs = [eval_batch.get(k) for k in ("x", "prompt_emb", "controller_inputs")]

        with self.autocast_ctx:
            latent_vid = sampler(ema_model, vid, prompt_emb, controller_inputs)

        if self.sampler_only_return_generated:
            latent_vid, controller_inputs = (
                x[:, vid.size(1):] if x is not None else None for x in (latent_vid, controller_inputs)
            )

        video_out = self.encoder_decoder.decode(latent_vid.float()) if self.encoder_decoder is not None else None

        # ---- Optionally Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Generate Media Artifacts ----
        video_out, controller_inputs = map(self._gather_concat_cpu, (video_out, controller_inputs))

        # HACK, remove
        mouse, btn = controller_inputs.split((2, 11), dim=-1)
        ######

        if self.rank == 0:
            eval_wandb_dict = to_wandb_samples(video_out, mouse, btn, fps=60)
        dist.barrier()

        return eval_wandb_dict
