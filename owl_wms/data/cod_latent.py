from .npy_table import NpyTable

import numpy as np

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
        meta_cols: tuple = ("vid_path", "missing", "truncated", "seq_len", "fps", "sample_rel_path"),
        array_columns: set | None = None,
        legal_fps: list[int] | None = None,
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)
        self.sampling_periods = sampling_periods
        self.requested_columns = set(array_columns) if array_columns is not None else set(self.table.columns)

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = [c for c in array_columns if c not in meta_cols]

        self.meta_passthrough = [c for c in self.requested_columns if c not in set(self.array_columns) and c != "fps"]

        seq_len, missing, truncated, fps = self.table[["seq_len", "missing", "truncated", "fps"]]
        self.fps = fps

        want_caps = ("captions" in self.requested_columns)
        if "captions" in self.array_columns:
            self.array_columns.remove("captions")
        self._captions_index = self.table["captions"] if (want_caps and "captions" in self.table.columns) else None

        self._index = []
        for i, (L, miss, trunc, f) in enumerate(zip(seq_len, missing, truncated, fps)):
            if legal_fps is not None and int(f) not in legal_fps:
                continue
            if not include_missing_features and miss:
                continue
            if not include_truncated and trunc:
                continue

            for start in range(0, L, window_length):
                # keep if any stride fits with phase=0; exact phase is handled in __getitem__
                if start + max(self.sampling_periods) * window_length <= L:
                    # If we're using captions (i.e., prompts), require at least one overlap.
                    if self._captions_index is not None:
                        s0 = start
                        s_end = start + max(self.sampling_periods) * window_length - 1
                        caps = self._captions_index[i] or []
                        keep = False
                        for cap in caps:
                            fr = cap.get("frame_range")
                            if fr:
                                cmin, cmax = int(fr[0]), int(fr[1])
                            else:
                                fi = cap.get("frame_indices") or []
                                if not fi:
                                    continue
                                cmin, cmax = int(fi[0]), int(fi[-1])
                            if not (cmax < s0 or cmin > s_end):
                                keep = True
                                break
                        if not keep:
                            continue
                    self._index.append((i, start))

        uniq, counts = np.unique(np.asarray(self.fps, int), return_counts=True)
        total = counts.sum()
        dist_str = ", ".join([f"{int(u)}: {c/total:.3g}" for u, c in zip(uniq, counts)])
        print(f"fps distribution: [{dist_str}]")
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

        # Captions: use only start/end for in-bounds check; anchor at s0 or caption start
        if self._captions_index is not None:
            s0 = off
            s_end = off + stride * (self.window_length - 1)
            cmap = {}
            for cap in (self._captions_index[row] or []):
                fr = cap.get("frame_range")
                if fr:
                    cmin, cmax = int(fr[0]), int(fr[1])
                else:
                    fi = cap.get("frame_indices") or []
                    if not fi:
                        continue
                    cmin, cmax = int(fi[0]), int(fi[-1])
                if cmax < s0 or cmin > s_end:
                    continue
                anchor = s0 if cmin < s0 else cmin
                cmap[int(anchor - s0)] = {k: v for k, v in cap.items() if isinstance(v, str)}
            out["captions"] = cmap

        out["fps"] = torch.tensor(int(self.fps[row]) // stride, dtype=torch.long)

        # Ensure all requested non-array columns are present.
        # If the column doesn't exist in the table schema, fill with None.
        for m in self.meta_passthrough:
            if m in out:  # already set (e.g., captions handled above)
                continue
            if m in self.table.columns:
                out[m] = self.table[m][row]
            else:
                out[m] = None

        return out


def collate_fn(batch, batch_columns: list, latent_column: str | None = None):
    # Stack tensors; keep meta (e.g., captions) as lists, with len(meta_list) == batch_size
    stacked = {}
    for k in batch_columns:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            t = torch.stack(vals)
            # TODO: buttons should be preprocessed as float
            stacked[k] = t.bfloat16() if (t.dtype == torch.float32 or k == "buttons") else t
        else:
            stacked[k] = vals
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
        legal_fps=None,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(
        dataset_path,
        seq_len,
        sampling_periods=sampling_periods,
        array_columns=set(batch_columns),
        legal_fps=legal_fps,
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
