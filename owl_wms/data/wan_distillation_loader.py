import random
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist


class AutoEpochDistributedSampler(DistributedSampler):
    """Shuffle every epoch automatically."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class WanPairDataset(Dataset):
    """
    Emits one sampled pair per run dir per epoch:
      returns dict with:
        'x_a': [F,C,H,W], 'time_a': [F],  # higher time (earlier in denoising)
        'x_b': [F,C,H,W], 'time_b': [F],  # lower  time (later   in denoising)
        'x_clean': [F,C,H,W], 'time_clean': [F],  # teacher clean endpoint (last step)

    Pair sampling:
      Δ ~ Uniform(delta_range)
      t_start ~ Uniform(t_min, t_max - Δ)
      t_end   = t_start + Δ
      Snap both to the discrete K-step grid (files 00000000..000000{K-1}).
    """

    # pure noise: 999
    sigmas = [
        0.9996664524078369, 0.9911890625953674, 0.982419490814209, 0.9733423590660095, 0.9639410972595215,
        0.9541981220245361, 0.9440944194793701, 0.9336096048355103, 0.9227216839790344, 0.911406934261322,
        0.8996397852897644, 0.887392520904541, 0.8746352195739746, 0.8613352179527283, 0.847457230091095,
        0.8329627513885498, 0.8178097009658813, 0.8019521832466125, 0.7853399515151978, 0.7679176926612854,
        0.7496247887611389, 0.7303943037986755, 0.7101521492004395, 0.688816487789154, 0.666296124458313,
        0.6424896121025085, 0.6172835826873779, 0.5905508399009705, 0.5621483325958252, 0.5319145321846008,
        0.4996665418148041, 0.4651961028575897, 0.42826521396636963, 0.38860073685646057, 0.34588751196861267,
        0.29975995421409607, 0.24979163706302643, 0.1954820156097412, 0.1362396627664566, 0.07136054337024689,
        0.0
    ][1:]

    boundary = 0.875   # WAN boundary ratio
    boundary_margin = 0.015   # keep a little distance from the kink

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        # 1) find run dirs by presence of the final step file
        self.run_dirs = sorted(
            p.parent
            for d in self.root.iterdir() if d.is_dir() and "." not in d.name
            for p in d.rglob("00000039_latents.pt")
            if all("." not in part for part in p.parent.relative_to(self.root).parts)
        )
        if not self.run_dirs:
            raise FileNotFoundError(f"No runs found under {root_dir}")

        # cache step indices per run (assume files are 000000xx_latents.pt)
        self._steps = {}
        for d in self.run_dirs:
            steps = sorted(int(fp.stem.split("_")[0]) for fp in d.glob("*_latents.pt"))
            if len(steps) >= 3:
                self._steps[d] = steps

    def __len__(self):
        # one pair per run per epoch
        return len(self.run_dirs)

    @staticmethod
    def _load_step(run_dir: Path, i: int) -> torch.Tensor:
        """file 000000{i}_latents.pt is [C,F,H,W]; return [F,C,H,W]."""
        x_cf_hw = torch.load(run_dir / f"{i:08d}_latents.pt", map_location="cpu")
        return x_cf_hw.permute(1, 0, 2, 3).contiguous()

    def _pick_pair_indices(self, steps):
        import random
        # avoid picking the final clean step as u (dt_u=0); pick a small random gap
        # also keep both indices on the same side of the expert boundary
        K = len(steps)
        while True:
            i = random.randrange(0, K - 2)  # 0..K-3, guarantees b+1 exists
            gap = 1                         # enforce adjacency: a = b-1
            a, b = i, i + gap               # a = k-1, b = k
            t_a = self.sigmas[steps[a]]
            t_b = self.sigmas[steps[b]]
            # also require next (k+1) to be on the same side of the boundary as u=b
            t_bp1 = self.sigmas[steps[b + 1]] if (b + 1) < K else None
            # fix typo + add boundary margin
            same_side = lambda x, y: ((x >= self.boundary + self.boundary_margin) and (y >= self.boundary + self.boundary_margin)) \
                                   or ((x <= self.boundary - self.boundary_margin) and (y <= self.boundary - self.boundary_margin))

            ok_next = True if t_bp1 is None else same_side(t_bp1, t_b)
            if same_side(t_a, t_b) and ok_next:
                return a, b

    def __getitem__(self, idx):
        run_dir = self.run_dirs[idx]
        steps = self._steps[run_dir]

        i_a, i_b = self._pick_pair_indices(steps)

        x_a = self._load_step(run_dir, steps[i_a])      # [F,C,H,W]  (k-1)
        x_b = self._load_step(run_dir, steps[i_b])      # [F,C,H,W]  (k)
        x_next = self._load_step(run_dir, steps[i_b+1]) # [F,C,H,W]  (k+1)
        F = x_a.shape[0]

        # WAN clock -> normalized [0,1] clock using the actual saved range [0,991]
        sigma_a = float(self.sigmas[steps[i_a]])  # e.g., 991, 982, ...
        sigma_b = float(self.sigmas[steps[i_b]])  # ...
        sigma_next = float(self.sigmas[steps[i_b+1]])
        assert sigma_a > sigma_b > sigma_next
        time_a = torch.full((F,), sigma_a, dtype=torch.float32)
        time_b = torch.full((F,), sigma_b, dtype=torch.float32)

        # teacher clean endpoint (assume step 39 exists)
        x_clean = self._load_step(run_dir, 39)               # [F,C,H,W]
        time_clean = torch.zeros((F,), dtype=torch.float32)  # WAN t(39)=0.0

        return {
            "x_a": x_a, "time_a": time_a,
            "x_b": x_b, "time_b": time_b,
            "x_next": x_next, "time_next": torch.full((F,), sigma_next, dtype=torch.float32),
            "x_clean": x_clean, "time_clean": time_clean,
        }


def collate_fn(batch, batch_columns: list):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    # commented out: need higher precision timesteps
    # stacked = {k: (t.bfloat16() if t.dtype == torch.float32 else t) for k, t in stacked.items()}
    return [stacked[col] for col in batch_columns]


def get_pair_loader(
    batch_size: int,
    dataset_path: str,
    batch_columns: list,
):
    """Same interface; returns batches of [x_a, time_a, x_b, time_b]."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WanPairDataset(dataset_path)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs,
    )


class WanGTWindowDataset(Dataset):
    """
    Returns a single tensor per item:
      x_gt: [W, C, H, W] where W is window_length (or full F if window_length=None)

    Each item is sampled from step-39 (fully denoised) of one run directory.
    """
    def __init__(self, root_dir: str, *, window_length: int | None = None, step_index: int = 39):
        self.root = Path(root_dir)
        self.window_length = window_length
        self.step_index = step_index

        # Find all run dirs that contain the target step file (e.g., 00000039_latents.pt)
        pattern = f"{step_index:08d}_latents.pt"
        self.run_dirs = sorted(
            p.parent
            for d in self.root.iterdir() if d.is_dir() and "." not in d.name
            for p in d.rglob(pattern)
            if all("." not in part for part in p.parent.relative_to(self.root).parts)
        )
        if not self.run_dirs:
            raise FileNotFoundError(f"No runs with {pattern} found under {root_dir}")

    @staticmethod
    def _load_step(run_dir: Path, step_idx: int) -> torch.Tensor:
        """Load 000000{step_idx}_latents.pt shaped [C,F,H,W], return [F,C,H,W]."""
        x_cf_hw = torch.load(run_dir / f"{step_idx:08d}_latents.pt", map_location="cpu")
        return x_cf_hw.permute(1, 0, 2, 3).contiguous()

    def __len__(self):
        # One sample per run per epoch (fresh random window each epoch if window_length is set)
        return len(self.run_dirs)

    def __getitem__(self, idx):
        run_dir = self.run_dirs[idx]
        x_fchw = self._load_step(run_dir, self.step_index)   # [F,C,H,W]
        return {"x": x_fchw[: self.window_length]}  # [W,C,H,W]


def get_sample_loader(
    batch_size: int,
    dataset_path: str,
    batch_columns: "list[str]",
    window_length: int | None = None,
    *,
    step_index: int = 39,
):
    """
    Returns a DataLoader that yields only a single tensor per batch:
      batch: [B, W, C, H, W]  (or [B, F, C, H, W] if window_length=None)

    Notes:
      - Assumes all runs share the same F if window_length=None (typical for WAN with fixed num_frames).
      - Uses DDP sampler when initialized under torch.distributed.
    """
    ds = WanGTWindowDataset(
        dataset_path,
        window_length=window_length,
        step_index=step_index,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True  # add drop_last=True if you want equal iters
        )
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs,
    )
