from torch import nn
import torch.nn.functional as F


class MLPCustom(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()

        self.fc1 = nn.Linear(dim_in, dim_middle)
        self.fc2 = nn.Linear(dim_middle, dim_out)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

        self.fc1.weight.data *= dim_in ** -0.5
        self.fc2.weight.data *= dim_middle ** -0.5

    def __post_init__(self):
        self.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x

class MLP(MLPCustom):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__(config.d_model, config.d_model * 4, config.d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x
