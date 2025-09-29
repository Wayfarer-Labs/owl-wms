from .npy_table import NpyTable

import random
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from dotenv import load_dotenv
load_dotenv()


class AutoEpochDistributedSampler(DistributedSampler):
    """Ensure we shuffle every epoch"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class WindowedViewDataset(Dataset):
    """
    A sliding-window view over an NpyTable.
    Indexes into (row_idx, start_offset) pairs.
    """
    def __init__(
        self,
        table_dir: str,
        window_length: int,
        sampling_periods: tuple[int, ...],
        include_missing_features: bool = False,
        include_truncated: bool = True,
        meta_cols: tuple = ("vid_path", "missing", "truncated", "seq_len", "fps"),
        array_columns: set | None = None,
        base_fps: int | None = None
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)
        self.sampling_periods = sampling_periods

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = [c for c in array_columns if c not in meta_cols]

        seq_len, missing, truncated, fps = self.table[["seq_len", "missing", "truncated", "fps"]]
        self.fps = fps

        self._index = []
        for i, (L, miss, trunc, f) in enumerate(zip(seq_len, missing, truncated, fps)):
            if base_fps is not None and (int(f) == 0 or base_fps % int(f) != 0):
                continue
            if not include_missing_features and miss:
                continue
            if not include_truncated and trunc:
                continue
            for start in range(0, L, window_length):
                # keep if any stride fits with phase=0; exact phase is handled in __getitem__
                if start + max(self.sampling_periods) * window_length <= L:
                    self._index.append((i, start))

        print(f"{len(self._index)} samples qualified out of {len(seq_len)} total videos")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        column_arrays = self.table.get(self.array_columns, rows=[row])
        rng = random.Random((row << 32) + start)  # deterministic per (row, start)
        stride = self.sampling_periods[rng.randrange(len(self.sampling_periods))]
        # uniform phase + unbiased interior shift within W = window_length * max_stride
        phase = rng.randrange(stride)
        kmax = (self.window_length * max(self.sampling_periods) - phase + stride - 1)//stride - self.window_length
        shift = rng.randrange(max(kmax, 0) + 1)
        off = start + phase + shift * stride
        out = {
            col: torch.from_numpy(arr_list[0][off : off + stride * self.window_length : stride])
            for col, arr_list in zip(self.array_columns, column_arrays)
        }
        out["fps"] = torch.tensor(int(self.fps[row]) // stride, dtype=torch.long)

        return out


def collate_fn(batch, batch_columns: list, latent_column: str | None = None):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    # TODO: fix hack, buttons should be preprocessed as float
    stacked = {
        k: t.bfloat16() if (t.dtype == torch.float32 or k == "buttons") else t
        for k, t in stacked.items()
        if k in batch_columns
    }
    if latent_column:
        stacked["x"] = stacked.pop(latent_column)
    assert len(stacked) == len(batch_columns)
    return stacked


def get_loader(
        batch_size,
        dataset_path,
        seq_len,
        batch_columns,
        latent_column=None,
        sampling_periods: tuple = (1,),
        base_fps=None,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(
        dataset_path,
        seq_len,
        sampling_periods=sampling_periods,
        array_columns=set(batch_columns),
        base_fps=base_fps,
    )

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)  # shuffle in sampler
    else:
        loader_kwargs = dict(shuffle=True, generator=torch.Generator().manual_seed(0))

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns, latent_column=latent_column),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs
    )
