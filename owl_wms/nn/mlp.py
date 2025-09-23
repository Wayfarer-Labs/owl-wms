from torch import nn
import torch.nn.functional as F
from .quantized_linear import QLinear


class MLPCustom(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out, fp8_mode=False):
        super().__init__()

        Linear = QLinear if fp8_mode else nn.Linear
        self.fc1 = Linear(dim_in, dim_middle)
        self.fc2 = Linear(dim_middle, dim_out)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc1.weight.data *= dim_in ** -0.5
        if fp8_mode:
            nn.init.zeros_(self.fc2.weight)
        else:
            self.fc2.weight.data *= dim_middle ** -0.5

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

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
            config.d_model,
            fp8_mode=getattr(config, "mlp_fp8", False),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x
