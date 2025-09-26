from torch import nn
import torch.nn.functional as F


class MLPCustom(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()

        self.fc1 = nn.Linear(dim_in, dim_middle)
        self.fc2 = nn.Linear(dim_middle, dim_out)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc1.weight.data *= dim_in ** -0.5
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
        dim_middle = config.d_model * getattr(config, "mlp_ratio", 4)

        super().__init__(config.d_model, dim_middle, config.d_model)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0 / dim_middle**0.5)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
