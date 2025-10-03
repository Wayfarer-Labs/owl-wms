from torch import nn
import torch.nn.functional as F

from .. import nn as owl_nn


class MLPCustom(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()

        self.fc1 = nn.Linear(dim_in, dim_middle, bias=False)
        self.fc2 = nn.Linear(dim_middle, dim_out, bias=False)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc1.weight.data *= dim_in ** -0.5
        self.fc2.weight.data *= dim_middle ** -0.5

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x


class MLP(MLPCustom):
    def __init__(self, config):
        super().__init__(
            config.d_model,
            config.d_model * getattr(config, "mlp_ratio", 4),
            config.d_model
        )


class QMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim_in = config.d_model
        dim_middle = config.d_model * getattr(config, "mlp_ratio", 4)
        dim_out = config.d_model

        self.fc1 = owl_nn.QLinear(dim_in, dim_middle, bias=False)
        self.fc2 = owl_nn.QLinear(dim_middle, dim_out, bias=False)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc1.weight.data *= dim_in ** -0.5
        self.fc2.weight.data *= dim_middle ** -0.5

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x
