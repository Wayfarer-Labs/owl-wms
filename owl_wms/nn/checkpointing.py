from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)
