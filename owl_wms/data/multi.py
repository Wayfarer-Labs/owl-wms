import random
from typing import Iterator, Sequence

import torch.distributed as dist


class MultiLoader:
    def __init__(self, loaders: Sequence[object], seed: int = 0):
        self.loaders = list(loaders)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return sum(len(ld) for ld in self.loaders)

    def __iter__(self) -> Iterator[object]:
        lengths = [len(ld) for ld in self.loaders]
        order = [i for i, L in enumerate(lengths) for _ in range(L)]  # proportional
        epoch_seed = self._seed + self._epoch
        random.Random(epoch_seed).shuffle(order)                       # reshuffle each epoch
        print(
            f"MultiLoader: epoch={self._epoch} "
            f"samples_per_loader={dict(enumerate(lengths))} total={sum(lengths)}",
        )
        iters = [iter(ld) for ld in self.loaders]
        for i in order:
            try:
                yield next(iters[i])
            except StopIteration:
                continue
        self._epoch += 1


# TODO: allow specification that all GPUs get samples from same subloader on the same step


def get_loader(**data_kwargs):
    """
    Expects data_kwargs to be a dict like:
      {
        "batch_size": <default for sub-loaders>,    # optional
        # optional defaults merged into each child data_kwargs (child overrides)
        "defaults": { ... },
        "loaders": [
          {"data_id": str, "data_kwargs": { "batch_size": int, ... }},
          ...
        ],
      }

    - Uses each child loader's own config (including its 'batch_size' if provided).
    """
    seed = int(data_kwargs.pop("seed", 0))
    if dist.is_available() and dist.is_initialized():
        seed += 1234567 * dist.get_rank()

    defaults = dict(data_kwargs.pop("defaults", {}))
    loaders_cfg = data_kwargs.pop("loaders", None)
    if loaders_cfg is None:
        raise ValueError("For data_id='multi', set data_kwargs.loaders: [...]")

    # Late import to avoid circular import
    from . import get_loader as _base_get_loader

    subs = []
    for cfg in loaders_cfg:
        sub_kwargs = {**defaults, **dict(cfg.get("data_kwargs", {}))}
        subs.append(_base_get_loader(cfg["data_id"], **sub_kwargs))
    return MultiLoader(subs, seed=seed)
