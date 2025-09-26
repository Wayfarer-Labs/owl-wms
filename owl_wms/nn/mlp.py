import torch
from torch import nn
import torch.nn.functional as F


class MLPCustom(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out, mlp_glu=False):
        super().__init__()

        self.up_proj = nn.Linear(dim_in, dim_middle, bias=False)
        self.down_proj = nn.Linear(dim_middle, dim_out, bias=False)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
        with torch.no_grad():
            self.down_proj.weight.data *= dim_middle ** -0.5

        self.mlp_glu = mlp_glu
        if self.mlp_glu:
            self.gate_proj = nn.Linear(dim_in, dim_middle, bias=False)
            nn.init.xavier_uniform_(self.gate_proj.weight)

    def forward(self, x):
        if not self.mlp_glu:
            x = F.silu(self.up_proj(x))
        else:
            x = F.silu(self.up_proj(x)) * self.gate_proj(x)
        return self.down_proj(x)


class MLP(MLPCustom):
    def __init__(self, config):
        super().__init__(
            config.d_model,
            int(config.d_model * getattr(config, "mlp_ratio", 4)),
            config.d_model,
            mlp_glu=True
        )
