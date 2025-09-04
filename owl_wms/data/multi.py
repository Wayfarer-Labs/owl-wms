import random
from typing import Iterator, Sequence


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
        random.Random(self._seed + self._epoch).shuffle(order)        # reshuffle each epoch
        iters = [iter(ld) for ld in self.loaders]
        for i in order:
            try:
                yield next(iters[i])
            except StopIteration:
                continue
        self._epoch += 1


def get_loader(batch_size: int, **data_kwargs):
    """
    Expects data_kwargs to be a dict like:
      {
        "batch_size": <default for sub-loaders>,    # optional
        "loaders": [
          {"data_id": str, "data_kwargs": { "batch_size": int, ... }},
          ...
        ],
        # "seed": <int>  # optional
      }

    - Prefers data_kwargs["batch_size"] over positional batch_size.
    - Also respects per-sub-loader data_kwargs["batch_size"].
    """
    assert batch_size == 1

    # Default per-sub-loader BS prefers data_kwargs.batch_size over positional arg
    seed = int(data_kwargs.pop("seed", 0))  # you said you don’t want to set it; defaults to 0
    loaders_cfg = data_kwargs.pop("loaders", None)
    if loaders_cfg is None:
        raise ValueError("For data_id='multi', set data_kwargs.loaders: [...]")

    # Late import to avoid circular import
    from . import get_loader as _base_get_loader

    subs = []
    for cfg in loaders_cfg:
        sub_kwargs = dict(cfg.get("data_kwargs", {}))
        # Pop per-loader batch_size from sub-kwargs to avoid arg collision
        per_bs = int(sub_kwargs.pop("batch_size", 1))
        subs.append(_base_get_loader(cfg["data_id"], per_bs, **sub_kwargs))
    return MultiLoader(subs, seed=seed)
