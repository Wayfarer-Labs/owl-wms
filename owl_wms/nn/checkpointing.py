from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def maybe_ckpt(do_ckpt, function, *args, **kwargs):
    if do_ckpt:
        return checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)
