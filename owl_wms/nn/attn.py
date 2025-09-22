import torch
import einops as eo
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .normalization import rms_norm
from .mlp import MLP


from .modulation import AdaLN, Gate
from .rope import get_rope

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def get_block_mask(
    t_pos: torch.Tensor,
    window_len: int | None = None,
    doc_id: torch.Tensor | None = None,
    q_offset: int = 0,
    is_causal: bool = True,
    device="cpu"
):
    kv_len = t_pos.shape[-1]
    q_len = kv_len - q_offset

    assert 0 <= q_offset < kv_len, "kv cache cannot exceed total tokens"
    if not is_causal:
        assert q_offset == 0, "kv caching not supported with bidirectional"

    def mask_mod(b, h, q, kv):
        abs_q = q + q_offset  # offset for kv caching
        t_q, t_kv = t_pos[b, abs_q], t_pos[b, kv]  # timestep of q / kv

        base_mask = (t_kv <= t_q) if is_causal else True  # causal / bidirectional
        window_mask = (t_q - t_kv).abs() < window_len if window_len is not None else True  # sliding window
        same_doc_mask = doc_id[b, abs_q] == doc_id[b, kv] if doc_id is not None else True

        return base_mask & window_mask & same_doc_mask

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)


class AttnMaskScheduler:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.global_period = getattr(self.config, "global_attn_period", 4)

    def __call__(self, seq_len, doc_id, kv_cache, device, t_pos):
        q_offset = t_pos.shape[-1] - seq_len
        torch._assert(q_offset >= 0, "negative q_offset")
        if kv_cache is not None:
            torch._assert((kv_cache.kv_offset == kv_cache.kv_offset[0]).all(), "Per-layer KV offsets diverged")
            torch._assert((kv_cache.kv_offset[0] == q_offset).all(), "cache offset disagrees with t_pos")
        torch._assert(doc_id is None or doc_id.size(1) == t_pos.size(1), "doc_id must be token-expanded to S tokens")

        kwargs = dict(
            t_pos=t_pos,
            doc_id=doc_id,
            q_offset=q_offset,
            is_causal=self.config.causal,
            device=device,
        )
        local_bm = get_block_mask(window_len=self.config.local_window, **kwargs)
        global_bm = get_block_mask(window_len=self.config.global_window, **kwargs)
        return [
            global_bm if (i % self.global_period) == 0 else local_bm
            for i in range(self.config.n_layers)
        ]


class Attn(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.n_heads = config.n_heads
        self.n_kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        self.d_head = config.d_model // self.n_heads
        assert config.d_model % self.n_heads == 0

        self.enable_gqa = self.n_heads != self.n_kv_heads

        qkv_out = (self.n_heads + 2 * self.n_kv_heads) * self.d_head
        self.qkv = nn.Linear(config.d_model, qkv_out, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = get_rope(config)

        self.gated_attn = getattr(config, "gated_attn", False)
        if self.gated_attn:
            self.gate = nn.Linear(config.d_model, config.d_model, bias=False)
            nn.init.zeros_(self.gate.weight)

    def forward(self, x, pos_ids, block_mask, kv_cache=None):
        q, k, v = eo.rearrange(self.qkv(x), "b t (g d) -> b g t d", d=self.d_head)\
                    .split([self.n_heads, self.n_kv_heads, self.n_kv_heads], dim=1)
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, pos_ids), self.rope(k, pos_ids)

        if kv_cache is not None:
            k, v = kv_cache.upsert(k, v, self.layer_idx)

        attn_out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=self.enable_gqa)
        attn_out = eo.rearrange(attn_out, "b h t d -> b t (h d)")
        attn_out = (attn_out * self.gate(x).sigmoid()) if self.gated_attn else attn_out
        attn_out = self.out(attn_out)
        return attn_out


class CrossAttention(nn.Module):
    def __init__(self, config, context_dim=None):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(context_dim or config.d_model, config.d_model * 2)
        self.o = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, context, context_pad_mask=None):
        q = eo.rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.n_heads)
        k, v = eo.rearrange(self.kv(context), "b m (two h d) -> two b h m d", two=2, h=self.n_heads)
        attn_mask = None if context_pad_mask is None else context_pad_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.o(out)


# HACK - clean up
class CrossAttentionSameFrame(nn.Module):
    def __init__(self, config, context_dim=None):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(context_dim or config.d_model, config.d_model * 2)
        self.o = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, context, context_pad_mask=None):
        q = eo.rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.n_heads)
        k, v = eo.rearrange(self.kv(context), "b m (two h d) -> two b h m d", two=2, h=self.n_heads)
        # Per-frame mask via flex_attention block mask: each query token attends only to its frame's controller token
        B, H, Lq, _ = q.shape
        M = k.size(2)
        assert Lq % M == 0, "query length must be an integer multiple of #context frames"
        tpf = Lq // M  # tokens per frame
        # int32 helps compiled block-mask perf/compat
        q_frame = (torch.arange(Lq, device=x.device, dtype=torch.int32) // tpf)  # [Lq]

        # Optional padding: keep only unpadded keys
        assert context_pad_mask is None
        #ctx_keep = None if context_pad_mask is None else (~context_pad_mask).to(device=x.device)

        def mask_mod(b, h, q_idx, kv_idx):
            same_frame = (q_frame[q_idx] == kv_idx)
            #if ctx_keep is not None:
            #    return same_frame & ctx_keep[b, kv_idx]
            return same_frame

        block_mask = create_block_mask(mask_mod, B=B, H=H, Q_LEN=Lq, KV_LEN=M, device=x.device)
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.o(out)



class DiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        dim = config.d_model

        self.attn = Attn(config, layer_idx)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond, block_mask, kv_cache=None):
        residual = x
        x = self.adaln1(x, cond)
        x = self.attn(x, block_mask, kv_cache)
        x = self.gate1(x, cond)
        x = residual + x

        residual = x
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = residual + x

        return x


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_masker = AttnMaskScheduler(config)
        self.blocks = nn.ModuleList([DiTBlock(config, idx) for idx in range(config.n_layers)])

    def forward(self, x, cond, doc_id=None, kv_cache=None):
        enable_ckpt = self.training and getattr(self.config, "gradient_checkpointing", False)
        block_masks = self.attn_masker(seq_len=x.size(1), doc_id=doc_id, kv_cache=kv_cache, device=x.device)
        for block, block_mask in zip(self.blocks, block_masks):
            if enable_ckpt:
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
        block_mask = self.get_block_mask(x, kv_cache)  # TODO: use AttnMaskScheduler or get_block_mask directly

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
    def __init__(self, d_model, channels, patch_size=1):
        super().__init__()
        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels * patch_size * patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)

        return x
