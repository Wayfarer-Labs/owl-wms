import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP


from .modulation import AdaLN, Gate
from .rope import FlatVideoRoPE, FrameRoPE

torch.backends.cuda.enable_flash_sdp(enabled = True)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

create_block_mask = torch.compile(create_block_mask, dynamic=True)
flex_attention = torch.compile(flex_attention, dynamic=True)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def create_causal_block_mask(tokens_per_frame: int, n_new_tokens: int, n_cached_tokens: int, device: torch.device):
    """
    Build a full (total_tokens × total_tokens) BlockMask that:
      - lets each token attend within its frame of size tokens_per_frame
      - lets it attend to any earlier frame
      - but in the final frame, no attention to frame 0
    Row index q is interpreted as absolute position q+offset.
    """
    total_tokens = n_cached_tokens + n_new_tokens
    n_frames = total_tokens // tokens_per_frame
    frame = torch.arange(total_tokens, device=device) // tokens_per_frame

    def mask_mod(b, h, q, k):
        abs_q = q + n_cached_tokens
        fq = frame[abs_q]
        fk = frame[k]
        return (fk <= fq) & ~((fq == n_frames - 1) & (fk == 0))

    return create_block_mask(mask_mod, None, None, n_new_tokens, total_tokens)


class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.layer_ind = None

        self.rope = FlatVideoRoPE(config)
        #self.rope = FrameRoPE(config)

        self.tokens_per_frame = config.tokens_per_frame
        self.causal = config.causal

    @torch.compile
    def forward(self, x, kv_cache=None):
        B, L, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.n_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.qk_norm(q, k)
        q, k = q.type_as(v), k.type_as(v)

        # prepend cache (if any) and record offset
        offset = kv_cache.length_at(self.layer_ind) if kv_cache is not None else 0
        if offset > 0:
            old_k, old_v = kv_cache.get(self.layer_ind)
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)
            if kv_cache.should_update:
                kv_cache.update(k.clone(), v.clone(), self.layer_ind)

        q, k = self.rope(q, k, offset)

        if not self.causal:
            block_mask = None  # default: full unmasked attention
        else:
            block_mask = create_causal_block_mask(self.tokens_per_frame, L, offset, device=x.device)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], L, -1)

        return self.out(x)


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond, kv_cache = None):
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x, kv_cache)
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
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond, kv_cache = None):
        for block in self.blocks:
            x = block(x, cond, kv_cache)

        return x

class UViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i

        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(nn.Linear(config.d_model * 2, config.d_model))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond, kv_cache = None):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]

            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)

            x = self.blocks[i](x, cond, kv_cache)

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

def test_attn_mask():
    total_tokens = 64
    tokens_per_frame = 8

    # Block causal mask
    mask = create_block_causal_mask(total_tokens, tokens_per_frame)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    plt.imshow(mask.float().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.title(f'Block Causal Mask ({total_tokens} tokens, {tokens_per_frame} per frame)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig('test_mask.png')
    plt.close()


@torch.no_grad()
def test_kv_cache():
    from .kv_cache import KVCache
    from ..configs import TransformerConfig

    # Create test configs
    config = TransformerConfig(
        n_layers=2,
        n_heads=8,
        d_model=64,
        tokens_per_frame=8
    )

    # Create model and cache
    model = DiT(config).cuda()
    cache = KVCache(config)
    cache.to('cuda')

    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model).cuda()
    cond = torch.randn(batch_size, seq_len//config.tokens_per_frame, config.d_model).cuda()

    # Test forward pass with cache
    cache.reset(batch_size)
    cache.enable_cache_updates()

    # First forward pass should populate cache
    out1 = model(x, cond, cache)

    # Second pass should use cached values
    cache.disable_cache_updates()
    out2 = model(x, cond, cache)

    # Outputs should match
    print("Max difference between outputs:", torch.max(torch.abs(out1 - out2)).item())
    print("Cache test complete")

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    test_attn_mask()
