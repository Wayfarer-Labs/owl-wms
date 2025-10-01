import torch.distributed as dist
import os
import datetime as dt

def setup(force=False, timeout=None):
    init_kwargs = dict(timeout=dt.timedelta(seconds=timeout)) if timeout else {}

    # Always under torchrun: fail fast, no silent fallback
    dist.init_process_group(backend="nccl", init_method="env://", **init_kwargs)
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    return global_rank, local_rank, world_size

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
