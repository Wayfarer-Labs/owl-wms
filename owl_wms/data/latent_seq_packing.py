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
        sampling_periods: tuple[int, ...] = (1, 2, 3),
        include_missing_features: bool = False,
        include_truncated: bool = True,
        meta_cols: tuple = ("tarball", "pt_idx", "missing", "truncated", "seq_len", "fps"),
        array_columns: set | None = None,
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)
        self.sampling_periods = sampling_periods

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = [c for c in array_columns if c != "fps"]
            # self.array_columns = array_columns
            # TODO: GET FPS FROM NPY TABLE

        seq_len, miss, trunc = [np.asarray(x) for x in self.table[["seq_len", "missing", "truncated"]]]
        # seq_len, miss, trunc, fps_full = [np.asarray(x) for x in self.table[["seq_len", "missing", "truncated", "fps"]]]

        # TODO: GET FPS FROM NPY TABLE
        self._fps_full = np.asarray([60] * len(trunc))
        ######

        mask = np.ones_like(seq_len, bool)
        if not include_missing_features:
            mask &= ~miss
        if not include_truncated:
            mask &= ~trunc

        self._docs = np.nonzero(mask)[0]
        self._lens = seq_len[mask].astype(np.int64)

        assert (self._lens > 0).all()

        self._build_packing()  # deterministic first epoch
        print(f"{len(self._slices)} packed windows over {len(self._docs)} documents")

    def set_epoch(self, epoch: int):
        rs = np.random.RandomState(epoch)   # deterministic across ranks
        self._build_packing(rs.permutation(len(self._docs)))

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, idx):
        sample, doc_id = {c: [] for c in self.array_columns}, []

        for doc, lo, hi in self._slices[idx]:
            row = self._row_lookup[doc]
            span = hi - lo
            arrays = self.table.get(self.array_columns, rows=[row])

            for col, arr in zip(self.array_columns, arrays):
                sample[col].append(arr[0][lo:hi])
            doc_id.extend([doc] * span)

        # deterministic stride per packed window
        seed_doc, seed_lo, _ = self._slices[idx][0]
        rng = random.Random((int(seed_doc) << 32) + int(seed_lo))
        stride = self.sampling_periods[rng.randrange(len(self.sampling_periods))]

        out = {}
        for k, v in sample.items():
            arr = np.concatenate(v)
            out[k] = torch.from_numpy(arr[::stride][: self.window_length])
        out["doc_id"] = torch.tensor(np.asarray(doc_id)[::stride][: self.window_length], dtype=torch.long)
        fps_val = float(self._fps_full[self._row_lookup[seed_doc]])
        out["fps"] = torch.tensor(fps_val / float(stride))
        return out

    def _build_packing(self, perm=None):
        if perm is None:
            perm = np.arange(len(self._docs))
        assert len(perm) == len(self._lens)
        self._row_lookup = self._docs[perm]
        self._slices = self.get_window_slices(perm)

    def get_window_slices(self, perm):
        """
        Pack a permutation of `lengths` into fixed-width `window`s.
        Return List[Chunk] where each Chunk = list[(doc, start, end)] and `end` is exclusive.
        """
        W = self.window_length * max(self.sampling_periods)
        lens = self._lens[perm]

        start = np.concatenate(([0], lens.cumsum()[:-1]))        # global offsets

        first = start // W
        n_win = (start + lens - 1) // W - first + 1

        assert n_win.sum() > 0

        # expand per-doc data to per-window rows
        rows = n_win.sum()
        doc = np.repeat(np.arange(len(perm)), n_win)

        # offset = running index reset at every doc
        offset = np.repeat(n_win.cumsum() - n_win, n_win)
        win_id = np.repeat(first, n_win) + np.arange(rows) - offset

        g0 = np.repeat(start, n_win)
        s_idx = np.maximum(g0, win_id * W) - g0
        e_idx = np.minimum(g0 + np.repeat(lens, n_win), (win_id + 1) * W) - g0

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


def get_loader(batch_size, dataset_path, seq_len, batch_columns, latent_column=None, sampling_periods=(1, 2, 3)):
    assert batch_size == 1

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(dataset_path, seq_len, array_columns=batch_columns, sampling_periods=sampling_periods)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)  # shuffle in sampler
    else:
        loader_kwargs = dict(shuffle=True)  # no sampler, shuffle in dataloader

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
