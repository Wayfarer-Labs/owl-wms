import torch
import torch.nn.functional as F


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),)).type_as(x)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))
