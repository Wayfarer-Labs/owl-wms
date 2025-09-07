import torch
import einops
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch.nn.functional as F
from .normalization import rms_norm
from .mlp import MLP


from .modulation import AdaLN, Gate
from .rope import get_rope_cls

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def get_block_mask(
    n_tokens: int,
    tokens_per_frame: int,
    window_len: int | None = None,
    doc_id: torch.Tensor | None = None,
    q_offset: int = 0,
    is_causal: bool = True,
    device="cpu"
):
    assert 0 <= q_offset < n_tokens, "kv cache cannot exceed total tokens"
    if not is_causal:
        assert q_offset == 0, "kv caching not supported with bidirectional"

    frame_id = torch.arange(n_tokens, device=device, dtype=torch.int32) // tokens_per_frame
    n_frames = n_tokens // tokens_per_frame

    if window_len is None:
        window_len = n_frames

    def mask_mod(b, h, q, kv):
        abs_q = q + q_offset  # offset for kv caching
        frame_q, frame_kv = frame_id[abs_q], frame_id[kv]

        if is_causal:
            causal_mask = frame_kv <= frame_q
        else:
            causal_mask = True

        if doc_id is not None:
            same_doc_mask = doc_id[b, frame_q] == doc_id[b, frame_kv]
        else:
            same_doc_mask = True

        window_mask = torch.abs(frame_q - frame_kv) < window_len

        return causal_mask & window_mask & same_doc_mask

    q_len = n_tokens - q_offset
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=n_tokens, device=device)


class Attn(nn.Module):
    def __init__(self, config, layer_idx, local = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.rope = get_rope_cls(getattr(config, "rope_impl", "ortho"))(config)

        self.local = local
        self.local_offset = config.local_window * config.tokens_per_frame

    def forward(self, x, block_mask, kv_cache=None):
        B, L, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = einops.rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        q, k = rms_norm(q), rms_norm(k)

        # rotate new queries and keys (shared kv cache between modalities)
        offset = kv_cache.get_offset(self.layer_idx) if kv_cache is not None else 0

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if kv_cache is not None:
            if kv_cache.should_update:
                kv_cache.update(k, v, self.layer_idx)
                k, v = kv_cache.get(self.layer_idx)
            else:
                k, v = kv_cache.get(self.layer_idx, new_k = k, new_v = v)

        # prepend cached values
        #if offset > 0:
        #    old_k, old_v = kv_cache.get(self.layer_idx)
        #    k = torch.cat([old_k, k], dim=2)
        #    v = torch.cat([old_v, v], dim=2)

        # update cache
        #if kv_cache is not None and kv_cache.should_update:
        #    kv_cache.update(k, v, self.layer_idx)

        # NOTE: Using block_mask = None to mark decoding, probably need something more explicit in future
        if self.local and block_mask is None:
            k = k[:,:,-self.local_offset:]
            v = v[:,:,-self.local_offset:]

        if block_mask is None:
            attn_out = flex_attention(q, k, v)
        else:
            attn_out = flex_attention(q, k, v, block_mask=block_mask)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], L, -1)

        return self.out(attn_out)


class DiTBlock(nn.Module):
    def __init__(self, config, layer_idx, local = False):
        super().__init__()

        dim = config.d_model

        self.attn = Attn(config, layer_idx, local)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond, block_mask, kv_cache=None):
        residual = x
        x = self.adaln1(x, cond)
        # Profile attention if kv_cache exposes profiling helpers
        if kv_cache is not None and hasattr(kv_cache, "_can_profile") and kv_cache._can_profile():
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            x = self.attn(x, block_mask, kv_cache)
            e1.record(); torch.cuda.synchronize()
            if not hasattr(kv_cache, "attn_ms"):
                kv_cache.attn_ms = 0.0
            kv_cache.attn_ms += e0.elapsed_time(e1)
        else:
            x = self.attn(x, block_mask, kv_cache)
        x = self.gate1(x, cond)
        x = residual + x

        residual = x
        x = self.adaln2(x, cond)
        if kv_cache is not None and hasattr(kv_cache, "_can_profile") and kv_cache._can_profile():
            e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
            e2.record()
            x = self.mlp(x)
            e3.record(); torch.cuda.synchronize()
            if not hasattr(kv_cache, "mlp_ms"):
                kv_cache.mlp_ms = 0.0
            kv_cache.mlp_ms += e2.elapsed_time(e3)
        else:
            x = self.mlp(x)
        x = self.gate2(x, cond)
        x = residual + x

        return x


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if not hasattr(config, "local_idx"):
            config.local_idx = 4
        self.local_layers = [(layer_idx % config.local_idx != 0) for layer_idx in range(config.n_layers)]
        self.blocks = nn.ModuleList([DiTBlock(config, idx, local) for idx, local in enumerate(self.local_layers)])
        self.decoding = False

    def enable_decoding(self):
        self.decoding = True

    def disable_decoding(self):
        self.decoding = False

    def get_block_mask(self, seq_len, doc_id, window_len, q_offset, device):
        n_tokens = seq_len + q_offset
        return get_block_mask(
            n_tokens=n_tokens,
            tokens_per_frame=self.config.tokens_per_frame,
            window_len=window_len,
            doc_id=doc_id,
            q_offset=q_offset,
            is_causal=self.config.causal,
            device=device
        )

    def forward(self, x, cond, doc_id=None, kv_cache=None, local_block_mask = None, global_block_mask = None):
        seq_len, device = x.size(1), x.device
        q_offset = kv_cache.get_offset(0) if kv_cache is not None else 0

        if local_block_mask is None and not self.decoding:
            local_block_mask = self.get_block_mask(seq_len, doc_id, self.config.local_window, q_offset, device)
        if global_block_mask is None and not self.decoding:
            global_block_mask = self.get_block_mask(seq_len, doc_id, self.config.global_window, q_offset, device)

        for layer_idx, block in enumerate(self.blocks):
            block_mask = local_block_mask if self.local_layers[layer_idx] else global_block_mask

            if self.training and getattr(self.config, "gradient_checkpointing", False):
                x = checkpoint(block, x, cond, block_mask, kv_cache)
            else:
                x = block(x, cond, block_mask, kv_cache)
        return x


class SkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = AdaLN(config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, prev, cond):
        x = x + prev
        x = self.norm(x, cond)
        x = self.proj(x)

        return x


class UViT(nn.Module):
    get_block_mask = DiT.get_block_mask

    def __init__(self, config):
        super().__init__()

        self.tokens_per_frame = config.tokens_per_frame
        self.causal = config.causal

        blocks = []
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i

        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(SkipConnection(config))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond, kv_cache = None):
        block_mask = self.get_block_mask(x, kv_cache)

        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond, block_mask, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond, block_mask, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]

            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = self.skip_projs[skip_idx](x, early_feat, cond)

            # Block
            x = self.blocks[i](x, cond, block_mask, kv_cache)
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
    device = "cpu"

    block_mask = get_block_mask(total_tokens, tokens_per_frame, device=device)

    # Convert to dense grid
    idx = torch.arange(total_tokens, device=device, dtype=torch.int32)
    bool_mask = block_mask.mask_mod(0, 0, idx[:, None], idx[None, :])
    dense_mask = torch.where(
        bool_mask, torch.tensor(0., device=device),
        torch.tensor(float("-inf"), device=device)
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.imshow(dense_mask.float().cpu().numpy(), cmap='gray')
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
