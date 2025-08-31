import torch
import einops as eo
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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


class AttnMaskScheduler:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.global_period = getattr(self.config, "global_attn_period", 4)

    def __call__(self, seq_len, doc_id, kv_cache, device):
        q_offset = kv_cache.length_at(0) if kv_cache is not None else 0
        n_tokens = seq_len + q_offset
        kwargs = dict(
            n_tokens=n_tokens,
            tokens_per_frame=self.config.tokens_per_frame,
            doc_id=doc_id,
            q_offset=q_offset,
            is_causal=self.config.causal,
            device=device
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
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.rope = get_rope_cls(getattr(config, "rope_impl", "ortho"))(config)

        self.use_attn_gate = getattr(config, "use_attn_gate", False)
        if self.use_attn_gate:
            self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -2.0)  # start mostly "closed"

    def forward(self, x, block_mask, kv_cache=None):
        qkv = self.qkv(x)
        q, k, v = eo.rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
        q, k = rms_norm(q), rms_norm(k)

        # rotate new queries and keys (shared kv cache between modalities)
        offset = kv_cache.length_at(self.layer_idx) if kv_cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # prepend cached values
        if offset > 0:
            old_k, old_v = kv_cache.get(self.layer_idx)
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)
        # update cache
        if kv_cache is not None and kv_cache.should_update:
            kv_cache.update(k, v, self.layer_idx)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)

        if self.use_attn_gate:
            gate = torch.sigmoid(self.gate_proj(x))
            gate = eo.rearrange(gate, "b t (h d) -> b h t d", h=self.n_heads)
            attn_out = attn_out * gate

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), -1)
        return self.out(attn_out)


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
    def __init__(self, d_model, channels, **conv_kw):
        super().__init__()

        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.ConvTranspose3d(d_model, channels, **conv_kw)

    def forward(self, x, cond, out_hw=None):
        """
        x: (B, D, N, s, s)    cond: (B, N, D)  # per-frame conditioning
        """
        B, D, N, H2, W2 = x.shape

        # token-wise AdaLN + SiLU (broadcast cond over spatial s×s)
        x_tok = eo.rearrange(x, 'b d n h w -> b (n h w) d')
        cond_tok = eo.repeat(cond, 'b n d -> b (n h w) d', h=H2, w=W2)
        x_tok = self.act(self.norm(x_tok, cond_tok))
        x = eo.rearrange(x_tok, 'b (n h w) d -> b d n h w', n=N, h=H2, w=W2)
        if out_hw is None:
            return self.proj(x)  # -> (B, C, N, s*ps[1], s*ps[2])
        else:
            H, W = out_hw
            return self.proj(x, output_size=(N, H, W))
