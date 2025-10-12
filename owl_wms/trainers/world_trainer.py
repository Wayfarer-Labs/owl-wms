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
torch.set_float32_matmul_precision("high")  # (low: bf16, high: tf32, highest: fp32)

# TODO: replace with itertools.batched in python3.13
batched = lambda it, n: iter(lambda it=iter(it): tuple(itertools.islice(it, n)), ())


# TODO REMOVE
torch._dynamo.config.optimize_ddp = False
#####


class WorldTrainer(BaseTrainer):
    """Trainer for WorldModel"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rank == 0:
            self.setup_wandb_metrics()

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

    def setup_wandb_metrics(self):
        wandb.define_metric("eval_frame_step", hidden=True)
        wandb.define_metric("eval_frame_loss/*", step_metric="eval_frame_step")
        wandb.define_metric("eval_sigma_step", hidden=True)
        wandb.define_metric("eval_sigma_loss/*", step_metric="eval_sigma_step")
        wandb.define_metric("eval_at_step", hidden=True)
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("train_loss", step_metric="global_step")

    @staticmethod
    def get_raw_model(model):
        while True:
            if hasattr(model, "module"):
                model = model.module
            elif hasattr(model, "_orig_mod"):
                model = model._orig_mod
            else:
                return model

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

        # self.quantize(self.model)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        raw = self.get_raw_model(self.model)
        self.ema = EMA(raw, beta=0.999, update_after_step=0, update_every=1, include_online_model=False)

        assert self.train_cfg.opt.lower() == "muon"
        self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)

        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)

            self.get_raw_model(self.model).load_state_dict(state["model"], strict=True)
            self.ema.load_state_dict(state["ema"], strict=True)
            self.opt.load_state_dict(state["opt"])
            self.total_step_counter = int(state.get("steps", 0))

            del state  # free memory

    def quantize(self, model):
        from torchao.float8 import convert_to_float8_training

        def only_mlp(mod: nn.Module, fqn: str) -> bool:
            return isinstance(mod, nn.Linear) and (fqn == "mlp" or fqn.endswith(".mlp") or ".mlp." in fqn)

        convert_to_float8_training(self.model, filter_fn=only_mlp)

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

        bi_warmup = getattr(self.train_cfg, "bidirectional_warmup", None)
        if bi_warmup is None:
            return
        elif bi_warmup == 0 or bi_warmup == self.total_step_counter:
            self.model.module.transformer.attn_masker.config.causal = True
        elif self.total_step_counter == 0:
            self.model.module.transformer.attn_masker.config.causal = False
        return
        """
        # step -> (local_window, global_window)
        # online_updates = {0: (1, 1), 50000: (2, 4), 100000: (3, 9), 150000: (4, 16)}
        online_updates = {0: (1, 1), 50_000: (8, 32)}
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
        """

    def prep_batch(self, batch):
        """Move to cuda, and if necessary use encoder to convert rgb to latent (x)"""
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        assert "rgb" not in batch, "rgb not supported, pass latents"

        if "mouse" in batch or "buttons" in batch:
            assert "controller_inputs" not in batch, "passed mouse or button, but already have `controller_inputs`"
            xs = tuple(filter(lambda x: x is not None, [batch.pop("mouse"), batch.pop("buttons")]))
            batch["controller_inputs"] = torch.cat(xs, dim=-1)

        # TODO: Clean up hacks
        if "captions" in batch:
            # Hack: We only use the first caption produced (for each element in the batch)
            # Hack: hardcoding "setting", captions used should be configurable
            batch["prompt"] = [
                (caps[min(caps, key=int)].get("setting", "") if caps else "")
                for caps in batch.pop("captions")
            ]
        # ########

        if "prompt" in batch:
            assert "prompt_emb" not in batch, "passed prompt to convert, but already have batch item `prompt_emb`"
            batch["prompt_emb"] = self.prompt_encoder(batch.pop("prompt"))

        # scale latents
        batch["x"] = (batch["x"] / self.train_cfg.vae_scale).bfloat16()

        # prepare frame temporal position ids (timestamps)
        batch["frame_timestamp"] = getattr(self.model, "module", self.model).get_frame_timestamps(
            batch.pop("fps"), batch["x"].size(1), batch["x"].device
        )

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

        if self.rank == 0 and getattr(self.train_cfg, "wandb_watch", False):
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
            loss = self.loss_step(self.model, batch)
            loss.backward()
            loss_sum += loss.item()

        # optimizer step
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        return loss_sum

    @torch.compile
    def loss_step(self, model, batch, reduction="mean", return_sigma=False):
        if getattr(self.train_cfg, "sfpt", False):
            loss = self.sfpt_loss(model, **batch, reduction=reduction, return_sigma=return_sigma)
        else:
            loss = self.conditional_flow_matching_loss(model, **batch, reduction=reduction, return_sigma=return_sigma)
        if return_sigma:
            loss, sigma = loss
            return (loss / self.accum_steps_per_device), sigma
        return loss

    def sfpt_loss(self, model, x, reduction="mean", return_sigma=False, **kw):
        x0 = x
        B, N = x0.size(0), x0.size(1)

        # sample diffusion forcing noised frames
        with torch.no_grad():
            sigma = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()  # LogitNormal(0,1)
            x1 = torch.randn_like(x0)
            x_t = torch.lerp(x0, x1, sigma.view(B, N, 1, 1, 1))

        # Predict priors given ground truth
        with self.autocast_ctx:
            # TODO: maybe no_grad this?
            v_pred_hat = model(x_t, sigma, **kw)
        clean_sigma = torch.full_like(sigma, self.train_cfg.noise_prev)
        x_hat = x_t + (clean_sigma - sigma).view(B, N, 1, 1, 1) * v_pred_hat

        # Construct sequence with predicted clean frames and original noised frames
        # noised frames can only attend to clean frames
        kw2 = {
            **kw,
            "frame_timestamp": kw["frame_timestamp"].repeat(1, 2),
            "doc_id": kw["doc_id"].repeat(1, 2) if kw.get("doc_id", None) is not None else None,
            "curr_frame_mask": (torch.arange(N * 2, device=x0.device) < N).repeat(B, 1),  # N noised (1), N clean (0)
        }

        with self.autocast_ctx:
            v_pred = model(
                torch.cat((x_t, x_hat), dim=1),
                torch.cat((sigma, clean_sigma), dim=1),
                **kw2
            )
            v_pred = v_pred[:, :N]  # only compute loss on x_t branch

        v_target = x1 - x0
        losses = F.mse_loss(v_pred, v_target, reduction=reduction)
        return (losses, sigma[:, :N]) if return_sigma else losses

    def conditional_flow_matching_loss(self, model, x, reduction="mean", return_sigma=False, **kw):
        """
        x0: [B, N, C, H, W] clean latents (sigma=0.0)
        """
        x0 = x
        B, N = x0.size(0), x0.size(1)

        with torch.no_grad():
            # sigma = torch.rand(B, N, device=x0.device, dtype=x0.dtype)  # Optional: U(0,1)
            sigma = torch.randn(B, N, device=x0.device, dtype=x0.dtype).sigmoid()  # LogitNormal(0,1)

            v_target = torch.randn_like(x0) - x0  # iid
            frame_timestamp = kw.pop("frame_timestamp")

            if getattr(self.train_cfg, "inference_matching", False):
                # Construct sequence of constant-noise, "denoised", previous frames
                sigma = torch.cat((sigma, x0.new_full((B, N), self.train_cfg.noise_prev)), dim=1)

                # repeat labels: [B, 2N]
                v_target = v_target.repeat(1, 2, 1, 1, 1)
                x0 = x0.repeat(1, 2, 1, 1, 1)
                frame_timestamp = frame_timestamp.repeat(1, 2)
                if kw.get("doc_id", None) is not None:
                    kw["doc_id"] = kw["doc_id"].repeat(1, 2)

                # mask: true=sampled noises, false=static noise @ noise_prev
                curr_frame_mask = (torch.arange(N * 2, device=x0.device) < N).repeat(B, 1)

            else:
                curr_frame_mask = None

            x_t = (x0 + v_target * sigma.view(B, -1, 1, 1, 1)).type_as(x0)
            sigma = sigma.type_as(x0)

        with self.autocast_ctx:
            v_pred = model(
                x_t, sigma,
                curr_frame_mask=curr_frame_mask,
                frame_timestamp=frame_timestamp,
                **kw
            )[:, :N]  # only compute loss on x_t branch

        losses = F.mse_loss(v_pred, v_target[:, :N], reduction=reduction)
        return (losses, sigma[:, :N]) if return_sigma else losses

    @torch.inference_mode()
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
            wandb_dict["global_step"] = self.total_step_counter
            wandb.log(wandb_dict)  # no step=

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
        eval_loss, timestep_loss_curve, sigma_hist = self.aggregate_eval_loss(ema_model, self.eval_loader())
        if self.rank == 0:
            # Log eval scalar now (no step=) so it doesn't collide with training rows
            if eval_loss is not None:
                wandb.log({"eval_loss": float(eval_loss), "eval_at_step": self.total_step_counter})
            eval_wandb_dict = {}
            # log scalar history so the LinePlot renders in Charts (no Tables created)
            if timestep_loss_curve and timestep_loss_curve[0]:
                xs, ys = timestep_loss_curve
                series_key = f"eval_frame_loss/{self.total_step_counter}"
                for f, y in zip(xs, ys):
                    wandb.log({
                        "eval_frame_step": int(f),            # x-axis
                        series_key: float(y),                 # y-axis (unique series)
                        "eval_at_step": self.total_step_counter,  # metadata for filtering
                    })
            # log sigma-binned histogram as a slidable series
            if sigma_hist and sigma_hist[0]:
                bx, by = sigma_hist
                series_key = f"eval_sigma_loss/{self.total_step_counter}"
                for b, y in zip(bx, by):
                    wandb.log({
                        "eval_sigma_step": int(b),
                        series_key: float(y),
                        "eval_at_step": self.total_step_counter,
                    })

        dist.barrier()

        return eval_wandb_dict

    @torch.inference_mode()
    def aggregate_eval_loss(self, model, loader):
        target = getattr(self.train_cfg, "n_eval_loss_samples", 0) // self.world_size
        if not target:
            return None, ([], []), ([], [])

        device = torch.device(f"cuda:{self.local_rank}")
        tot = torch.zeros(2, device=device, dtype=torch.float64)  # [sum_loss, count]
        fsum = fcnt = None
        remaining = int(target)

        # sigma histogram accumulators
        num_bins = 20
        bin_sums = torch.zeros(num_bins, device=device, dtype=torch.float64)
        bin_counts = torch.zeros(num_bins, device=device, dtype=torch.float64)
        edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=device)

        for batch in loader:
            batch = self.prep_batch(batch)
            per, sig = self.loss_step(model, batch, reduction="none", return_sigma=True)
            bsz = int(batch["x"].shape[0])

            pf = per.mean(dim=(0, 2, 3, 4)).to(device=device, dtype=torch.float64)  # [N]
            if fsum is None:
                fsum = torch.zeros_like(pf)
                fcnt = torch.zeros_like(pf)
            fsum += pf * bsz
            fcnt += bsz

            tot[0] += per.sum().to(torch.float64)
            tot[1] += torch.tensor(per.numel(), device=device, dtype=torch.float64)

            # accumulate sigma-binned per-frame losses
            per_frame = per.mean(dim=(2, 3, 4)).to(dtype=torch.float64)  # [B,N]
            vals = per_frame.reshape(-1)
            sigv = sig.reshape(-1)
            idx = torch.bucketize(sigv, edges, right=False) - 1
            idx = idx.clamp_(0, num_bins - 1)
            bin_sums.scatter_add_(0, idx, vals)
            bin_counts.scatter_add_(0, idx, torch.ones_like(vals))

            remaining -= bsz
            if remaining <= 0:
                break

        if self.world_size > 1:
            dist.all_reduce(tot)
            dist.all_reduce(fsum)
            dist.all_reduce(fcnt)
            dist.all_reduce(bin_sums)
            dist.all_reduce(bin_counts)

        loss = (tot[0] / torch.clamp_min(tot[1], 1)).item()
        if self.rank != 0:
            return loss, ([], []), ([], [])

        ys = (fsum / torch.clamp_min(fcnt, 1)).tolist()
        sigma_means = (bin_sums / torch.clamp_min(bin_counts, 1)).tolist()
        return loss, (list(range(len(ys))), ys), (list(range(num_bins)), sigma_means)

    def sample_step(self, sampler):
        ema_model = self.ema.ema_model
        ema_model.eval()

        # ---- Batch & labels ----

        # TODO: Clean this up
        raw_batch = next(self.sample_loader)
        # keep literal prompt(s) before prep_batch() converts them to embeddings
        literal_prompt = raw_batch.get("prompt")
        if literal_prompt is None and "captions" in raw_batch:
            literal_prompt = [
                (caps[min(caps, key=int)].get("setting", "") if caps else "")
                for caps in raw_batch["captions"]
            ]
        eval_batch = self.prep_batch(raw_batch)
        # ########

        vid, prompt_emb, controller_inputs = [eval_batch.get(k) for k in ("x", "prompt_emb", "controller_inputs")]
        if self.train_cfg.num_seed_frames:
            vid = vid[:, :self.train_cfg.num_seed_frames]

        lw = ema_model.transformer.local_window  # int(ema_model.transformer.local_window.item())
        gw = ema_model.transformer.global_window  # int(ema_model.transformer.global_window.item())
        fps = int(raw_batch["fps"])

        def mk_labels(fps_val: int, n: int):
            base = {"noise_prev": self.train_cfg.noise_prev, "local attn": lw, "global attn": gw}
            return [{"fps": fps_val, **base} for _ in range(n)]

        # ---- Generate ----
        with self.autocast_ctx:
            latent_vid = sampler(
                ema_model, vid, prompt_emb, controller_inputs,
                fps=raw_batch["fps"], num_frames=self.train_cfg.num_generated_frames,
                noise_prev=self.train_cfg.noise_prev,
                noise_distribution=getattr(self.train_cfg, "noise_distribution", "iid"),
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
        if self.pg_cpu is not None:
            _bufs = [None] * self.world_size
            dist.all_gather_object(_bufs, literal_prompt, group=self.pg_cpu)
            literal_prompt = [p for b in _bufs for p in ((b if isinstance(b, list) else [b]) if b is not None else [])]

        num_gt_frames = 0 if self.sampler_only_return_generated else self.train_cfg.num_seed_frames

        if self.rank == 0:
            n_out = 0 if video_out is None else video_out.size(0)
            labels_out = mk_labels(fps, n_out)
            return to_wandb_samples(
                video_out, mouse, btn,
                labels=labels_out, num_gt_frames=num_gt_frames,
                prompts=(literal_prompt or None),
            )
        return None
