import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import functools
import os

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
        print(f"Epoch: {self._auto_epoch}")
        return super().__iter__()


def batch_windowed(examples, window_length):
    out = {k: [] for k in examples}
    for k, col in examples.items():
        for t in col:
            # split into windowed chunks. drop the tail so length % window_length == 0
            t = t[: (t.size(0) // window_length) * window_length]
            # split into windows and extend our output list
            out[k].extend(t.view(t.size(0) // window_length, window_length, *t.shape[1:]).unbind(0))
    return out


def collate_fn(batch):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    return [stacked[k] for k in ("latent", "audio", "mouse", "button")]


def get_loader(batch_size, bucket_name, window_length, **_):  # TODO: no extra kwargs
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    num_workers = os.cpu_count() // 4 or 1

    bucket_name = "lapp0/cod-data-latent-360x640to8x8"  # TODO: remove hard-coded
    ds = datasets.load_dataset(bucket_name, split="train")
    ds.set_format(type="numpy", columns=list(ds.column_names))
    ds = ds.map(
        functools.partial(batch_windowed, window_length=window_length),
        batched=True,
        batch_size=1,
        num_proc=num_workers,
        remove_columns=ds.column_names
    )
    ds.set_format(type="torch", columns=list(ds.column_names))

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)  # shuffle in sampler
    else:
        loader_kwargs = dict(shuffle=True)  # no sampler, shuffle in dataloader

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 * world_size,
        **loader_kwargs
    )
