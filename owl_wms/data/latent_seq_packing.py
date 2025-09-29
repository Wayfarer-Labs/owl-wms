from .npy_table import NpyTable

from functools import partial
import random

import numpy as np

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
        e = self._auto_epoch
        super().set_epoch(e)
        base = getattr(self.dataset, "_epoch_span", len(self.dataset))
        self._auto_epoch += 1
        for i in super().__iter__():
            yield i + e * base


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
        legal_fps: list[int] | None = None,
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)
        self.sampling_periods = sampling_periods
        self.max_stride = max(self.sampling_periods)

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = [c for c in array_columns if c not in meta_cols]

        seq_len, miss, trunc, fps = [np.asarray(x) for x in self.table[["seq_len", "missing", "truncated", "fps"]]]

        mask = np.ones_like(seq_len, bool)
        if not include_missing_features:
            mask &= ~miss
        if not include_truncated:
            mask &= ~trunc
        if legal_fps is not None:
            mask &= np.isin(fps, legal_fps)

        sel = fps[mask]
        bad = np.unique([f for f in sel if any(f % p for p in self.sampling_periods)])
        if bad.size:
            raise ValueError(f"bad fps for {self.sampling_periods}: {bad.tolist()}")

        self._docs = np.nonzero(mask)[0]
        self._lens = seq_len[mask].astype(np.int64)
        self._fps = fps[mask].astype(np.float32)

        assert (self._lens > 0).all()

        self._build_packing()  # deterministic first epoch
        uniq, counts = np.unique(self._fps.astype(int), return_counts=True)
        total = counts.sum()
        dist_str = ", ".join([f"{int(u)}: {c / total:.3g}" for u, c in zip(uniq, counts)])
        print(f"fps distribution: [{dist_str}]")
        print(f"{len(self._slices)} packed windows over {len(self._docs)} documents")

    def set_epoch(self, epoch: int):
        rs = np.random.RandomState(epoch)   # deterministic across ranks
        W = self.window_length * self.max_stride
        self._build_packing(rs.permutation(len(self._docs)), int(rs.randint(W)))

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, idx):
        # lazily rebuild per-epoch inside workers
        base = getattr(self, "_epoch_span", len(self._slices))
        epoch = idx // base
        if epoch != getattr(self, "_local_epoch", -1):
            self._epoch_span = base  # freeze span for this epoch
            rs = np.random.RandomState(epoch)
            W = self.window_length * self.max_stride
            self._build_packing(rs.permutation(len(self._docs)), int(rs.randint(W)))
            self._local_epoch = epoch
        idx = idx % len(self._slices)

        sample, doc_id = {c: [] for c in self.array_columns}, []

        for doc, lo, hi in self._slices[idx]:
            row = self._row_lookup[doc]
            arrays = self.table.get(self.array_columns, rows=[row])

            for col, arr in zip(self.array_columns, arrays):
                sample[col].append(arr[0][lo:hi])
            doc_id.extend([doc] * (hi - lo))

        seed_doc, seed_lo, _ = self._slices[idx][0]
        rng = random.Random(((int(seed_doc) << 32) + int(seed_lo)) ^ epoch)  # now varies by epoch
        stride = self.sampling_periods[rng.randrange(len(self.sampling_periods))]
        phase = rng.randrange(stride)
        # unbiased subwindow shift within the strided view
        W = self.window_length * self.max_stride
        n_avail = (W - phase + stride - 1) // stride
        shift_max = max(0, n_avail - self.window_length)
        k = rng.randrange(shift_max + 1) if shift_max else 0
        start = phase + k * stride
        out = {
            k: torch.from_numpy(np.concatenate(v)[start::stride][: self.window_length])
            for k, v in sample.items()
        }
        doc_full = np.asarray(doc_id, dtype=np.int64)
        out["doc_id"] = torch.from_numpy(doc_full[start::stride][: self.window_length]).long()
        # per-doc fps from first segment, adjusted by stride (float)
        base_fps = int(self._fps_perm[seed_doc])
        assert base_fps % stride == 0, f"base fps {base_fps} must be divisible by stride {stride}"
        fps_val = base_fps // stride
        out["fps"] = torch.tensor(fps_val, dtype=torch.long)

        return out

    def _build_packing(self, perm=None, shift=0):
        if perm is None:
            perm = np.arange(len(self._docs))
        W = self.window_length * self.max_stride
        shift = int(shift % W)
        assert len(perm) == len(self._lens)
        self._row_lookup = self._docs[perm]
        self._slices = self.get_window_slices(perm, shift)
        self._fps_perm = self._fps[perm]

    def get_window_slices(self, perm, shift):
        """
        Pack a permutation of `lengths` into fixed-width `window`s.
        Return List[Chunk] where each Chunk = list[(doc, start, end)] and `end` is exclusive.
        """
        lens = self._lens[perm]
        start = np.concatenate(([0], lens.cumsum()[:-1]))        # global offsets

        # require enough raw frames for the largest stride
        W = self.window_length * self.max_stride

        # apply a global circular shift of the W grid: boundaries at k*W - shift
        start_shifted = start + shift
        first = start_shifted // W
        n_win = (start_shifted + lens - 1) // W - first + 1

        assert n_win.sum() > 0

        # expand per-doc data to per-window rows
        rows = n_win.sum()
        doc = np.repeat(np.arange(len(perm)), n_win)

        # offset = running index reset at every doc
        offset = np.repeat(n_win.cumsum() - n_win, n_win)
        win_id = np.repeat(first, n_win) + np.arange(rows) - offset

        g0 = np.repeat(start, n_win)

        # window edges in original coordinates
        left = win_id * W - shift
        right = (win_id + 1) * W - shift
        s_idx = np.maximum(g0, left) - g0
        e_idx = np.minimum(g0 + np.repeat(lens, n_win), right) - g0

        # `win_id` is already non-decreasing → just split where it changes
        cuts = np.flatnonzero(np.diff(win_id)) + 1
        blocks = np.split(np.column_stack([doc, s_idx, e_idx]), cuts)

        slices = [list(map(tuple, blk)) for blk in blocks]

        # remove last sequence if its truncated
        return [s for s in slices if sum(hi - lo for _, lo, hi in s) == W]


def collate_fn(batch, batch_columns: list, latent_column: str | None = None):
    batch_columns = batch_columns + ["doc_id"]  # needed for sequence packing
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
        dataset_path,
        seq_len,
        batch_columns,
        latent_column,
        batch_size=1,
        sampling_periods: tuple = (1,),
        legal_fps=None,

):
    assert batch_size == 1

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(
        dataset_path,
        seq_len,
        array_columns=batch_columns,
        sampling_periods=sampling_periods,
        legal_fps=legal_fps,
    )

    sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader_kwargs = dict(sampler=sampler, shuffle=False)  # always shuffle in sampler

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
