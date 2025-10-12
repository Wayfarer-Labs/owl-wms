from typing import Optional
from torch import Tensor

import torch
import einops as eo
from torch import nn
import torch.nn.functional as F

from .normalization import rms_norm
from .rope import get_rope

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def get_block_mask(
    t_pos: Tensor,
    window_len: Optional[Tensor] = None,
    doc_id: Optional[Tensor] = None,
    q_offset: int = 0,
    is_causal: bool = True,
    curr_frame_mask: Optional[Tensor] = None,
    device="cpu"
):
    kv_len = t_pos.shape[-1]
    q_len = kv_len - q_offset

    assert 0 <= q_offset < kv_len, "kv cache cannot exceed total tokens"
    if not is_causal:
        assert q_offset == 0, "kv caching not supported with bidirectional"

    # EXPERIMENTAL
    if curr_frame_mask is not None:
        assert q_offset == 0
    # ########

    def mask_mod(b, h, q, kv):
        abs_q = q + q_offset  # offset for kv caching
        t_q, t_kv = t_pos[b, abs_q], t_pos[b, kv]  # timestamp of q / kv

        base_mask = (t_kv <= t_q) if is_causal else True  # causal / bidirectional
        window_mask = (t_q - t_kv).abs() < window_len if window_len is not None else True  # sliding window
        same_doc_mask = doc_id[b, abs_q] == doc_id[b, kv] if doc_id is not None else True  # for sequence packing

        # EXPERIMENTAL
        ##############
        # current prev attn: previous frames are noised at a contant level, current frames noised at random level
        # matches inference behavior
        if curr_frame_mask is not None:
            is_curr_q = curr_frame_mask[b, abs_q]
            is_curr_kv = curr_frame_mask[b, kv]
            prev_curr_mask = (~is_curr_q & ~is_curr_kv) | (is_curr_q & ((t_kv == t_q) == is_curr_kv))
        else:
            prev_curr_mask = True
        # ########

        return base_mask & window_mask & same_doc_mask & prev_curr_mask

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)


class AttnMaskScheduler:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.causal = getattr(self.config, "causal", True)
        self.global_period = getattr(self.config, "global_attn_period", 4)

    def __call__(self, seq_len, doc_id, kv_cache, device, t_pos, curr_frame_mask=None, *, local_window, global_window):
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
            curr_frame_mask=curr_frame_mask,
            device=device,
        )
        local_bm = get_block_mask(window_len=local_window, **kwargs)
        global_bm = get_block_mask(window_len=global_window, **kwargs)
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

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = get_rope(config)

        self.sink_attn = getattr(config, "sink_attn", False)
        if self.sink_attn:
            self.attn_sinks = nn.Parameter(torch.zeros(self.n_heads))

        self.gated_attn = getattr(config, "gated_attn", False)
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)  # sparse attn gate
            nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x, pos_ids, bm, kv_cache=None):
        # Q, K, V proj -> QK-norm -> RoPE
        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads, d=self.d_head)
        k = eo.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)
        v = eo.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, pos_ids), self.rope(k, pos_ids)

        # Update KV-cache and K, V in-place
        if kv_cache is not None:
            k, v = kv_cache.upsert(k, v, self.layer_idx)

        # SDPA -> Attention Gate -> Out Proj
        if self.sink_attn:
            y, lse = flex_attention(q, k, v, block_mask=bm, enable_gqa=self.enable_gqa, return_lse=True)
            with torch.autocast("cuda", enabled=False):
                sink_scale = torch.sigmoid((lse.float() - self.attn_sinks.view(1, -1, 1))).to(y.dtype)
            y = y * sink_scale.unsqueeze(-1).to(y.dtype)
        else:
            y = flex_attention(q, k, v, block_mask=bm, enable_gqa=self.enable_gqa)

        if self.gated_attn:
            gates = torch.sigmoid(self.gate_proj(x[..., :self.n_heads]))
            y = y * gates.permute(0, 2, 1).unsqueeze(-1)
        y = eo.rearrange(y, "b h t d -> b t (h d)")
        y = self.out_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config, context_dim=None):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(context_dim or config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(context_dim or config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, context, context_pad_mask=None):
        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = eo.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        v = eo.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        attn_mask = None if context_pad_mask is None else context_pad_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.o(out)
