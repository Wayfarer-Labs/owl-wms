from .npy_table import NpyTable

from functools import partial

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
        include_missing_features: bool = False,
        include_truncated: bool = True,
        meta_cols: tuple = ("tarball", "pt_idx", "missing", "truncated", "seq_len"),
        array_columns: set | None = None,
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = array_columns

        seq_len, miss, trunc = [np.asarray(x) for x in self.table[["seq_len", "missing", "truncated"]]]
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

        out = {k: torch.from_numpy(np.concatenate(v)) for k, v in sample.items()}
        out["doc_id"] = torch.tensor(doc_id, dtype=torch.long)
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
        lens = self._lens[perm]
        start = np.concatenate(([0], lens.cumsum()[:-1]))        # global offsets

        first = start // self.window_length
        n_win = (start + lens - 1) // self.window_length - first + 1

        assert n_win.sum() > 0

        # expand per-doc data to per-window rows
        rows = n_win.sum()
        doc = np.repeat(np.arange(len(perm)), n_win)

        # offset = running index reset at every doc
        offset = np.repeat(n_win.cumsum() - n_win, n_win)
        win_id = np.repeat(first, n_win) + np.arange(rows) - offset

        g0 = np.repeat(start, n_win)
        s_idx = np.maximum(g0, win_id * self.window_length) - g0
        e_idx = np.minimum(g0 + np.repeat(lens, n_win), (win_id + 1) * self.window_length) - g0

        # `win_id` is already non-decreasing → just split where it changes
        cuts = np.flatnonzero(np.diff(win_id)) + 1
        blocks = np.split(np.column_stack([doc, s_idx, e_idx]), cuts)

        slices = [list(map(tuple, blk)) for blk in blocks]

        # remove last sequence if its truncated
        return [s for s in slices if sum(hi - lo for _, lo, hi in s) == self.window_length]


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


def get_loader(batch_size, dataset_path, seq_len, batch_columns, latent_column=None):
    assert batch_size == 1

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WindowedViewDataset(dataset_path, seq_len, array_columns=batch_columns)

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
