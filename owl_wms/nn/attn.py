import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP

import einops as eo

from .modulation import AdaLN, Gate
#from .embeddings import FlatVideoRoPE
from rotary_embedding_torch import RotaryEmbedding

torch.backends.cuda.enable_flash_sdp(enabled = True)

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)

    def forward(self, x):
        q,k,v = eo.rearrange(self.qkv(x), 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)
        q,k = self.qk_norm(q,k)

        x = F.scaled_dot_product_attention(q,k,v)
        x = eo.rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond):
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x)
        x = self.gate1(x, cond)
        x = res1 + x
        
        res2 = x.clone()
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = res2 + x

        return x

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(DiTBlock(config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)

        return x

class UViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(DiTBlock(config))
        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(nn.Linear(config.d_model * 2, config.d_model))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]
            
            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)
            
            x = self.blocks[i](x, cond)

        return x

# === VIT Specific Layers ===

class FinalLayer(nn.Module):
    def __init__(self, sample_size, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels*patch_size*patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)

        return x

if __name__ == "__main__":
    layer = PatchProjOut(64, 384, 3, 4).cuda().bfloat16()
    x = torch.randn(1,256,384).cuda().bfloat16()

    with torch.no_grad():
        z = layer(x)
        print(z.shape)