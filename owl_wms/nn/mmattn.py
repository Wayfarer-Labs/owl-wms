import torch
from torch import nn

from .normalization import layer_norm
from .mlp import MLP

import einops as eo

from .modulation import cond_adaln, cond_gate
from .rope import AVRoPE
from .attn import create_causal_block_mask

from torch.nn.attention.flex_attention import flex_attention

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


flex_attention = torch.compile(flex_attention, dynamic=True)

"""
This code makes the assumption that there are some
tokens from another modality that must always be attended to
"""


class MMAttn(nn.Module):
    """
    MMDiT style attention
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.tok_per_frame_mod = [self.config.sample_size ** 2, 1]

        self.qkv_projs = nn.ModuleList([nn.Linear(config.d_model, 3 * config.d_model) for _ in range(2)])
        self.out_projs = nn.ModuleList([nn.Linear(config.d_model, config.d_model)for _ in range(2)])
        self.rope = AVRoPE(config)

    def split_qkv(self, qkv, tok_per_frm):
        return eo.rearrange(qkv, 'b (f n) (three h d) -> three b h f n d', n=tok_per_frm, three=3, h=self.n_heads)

    def forward(self, x0, x1, block_mask=None, kv_cache=None):
        """MMDiT Attention: Calculate qkv separately per modality, interleave and concat, SDPA, separate"""
        # calculate qkvs for each modality: qkv is list of triplets
        qkvs = [
            self.split_qkv(self.qkv_projs[i](x), self.tok_per_frame_mod[i]).unbind(0)
            for i, x in enumerate([x0, x1])
        ]
        # concat along tok-per-frame of [(b, h, f, 64, d), (b, h, f, 1, d)] and flatten to interleave modalities
        q, k, v = [torch.cat(groups, dim=3) for groups in zip(*qkvs)]
        q, k, v = [eo.rearrange(x, 'b h f n d -> b h (f n) d') for x in [q, k, v]]

        q, k = layer_norm(q), layer_norm(k)

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
            kv_cache.update(k.clone(), v.clone(), self.layer_idx)

        # Attention & merge heads
        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = eo.rearrange(attn_out, 'b h n d -> b n (h d)')

        # Split into original modalities + out proj
        V = self.config.sample_size**2
        x0, x1 = eo.rearrange(attn_out, 'b (f n) d -> b f n d', n=V + 1).split([V, 1], dim=2)
        x0, x1 = x0.flatten(1, 2), x1.flatten(1, 2)

        x0 = self.out_projs[0](x0).contiguous()
        x1 = self.out_projs[1](x1).contiguous()

        return x0, x1


class MMDiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = MMAttn(config, layer_idx)
        self.mlps = nn.ModuleList([MLP(config) for _ in range(2)])

    def forward(self, x0, x1, cond0, cond1, block_mask=None, kv_cache=None):
        # Conditioning inputs
        attn_scale0, attn_bias0, attn_gate0, mlp_scale0, mlp_bias0, mlp_gate0 = cond0.chunk(6, dim=-1)
        attn_scale1, attn_bias1, attn_gate1, mlp_scale1, mlp_bias1, mlp_gate1 = cond1.chunk(6, dim=-1)

        # Conditioned Attention
        r0, r1 = x0, x1
        x0, x1 = cond_adaln(x0, attn_scale0, attn_bias0), cond_adaln(x1, attn_scale1, attn_bias1)
        x0, x1 = self.attn(x0, x1, block_mask, kv_cache)
        x0, x1 = cond_gate(x0, attn_gate0), cond_gate(x1, attn_gate1)
        x0, x1 = (r0 + x0), (r1 + x1)

        # Conditioned MLP
        r0, r1 = x0, x1
        x0, x1 = cond_adaln(x0, mlp_scale0, mlp_bias0), cond_adaln(x1, mlp_scale1, mlp_bias1)
        x0, x1 = self.mlps[0](x0), self.mlps[1](x1)
        x0, x1 = cond_gate(x0, mlp_gate0), cond_gate(x1, mlp_gate1)
        x0, x1 = (r0 + x0), (r1 + x1)

        return x0, x1


class MMDIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # layer attention pattern is [global, local, local, local, global, ...]
        self.local_layers = [(layer_idx % 4 != 0) for layer_idx in range(config.n_layers)]
        self.local_window = nn.Buffer(torch.tensor(self.config.local_window, dtype=torch.int32), persistent=False)
        self.global_window = nn.Buffer(torch.tensor(self.config.global_window, dtype=torch.int32), persistent=False)

        self.blocks = nn.ModuleList([MMDiTBlock(config, idx) for idx in range(config.n_layers)])

        # DiT-Air
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model * 2 * 2 * 3)
        )

    def get_block_mask(self, x0, x1, kv_cache, window_len):
        if not self.config.causal:
            return None
        seq_len = x0.shape[1] + x1.shape[1]
        offset = kv_cache.length_at(0) if kv_cache is not None else 0
        return create_causal_block_mask(
            n_tokens=seq_len + offset,
            tokens_per_frame=self.config.tokens_per_frame,
            n_cached_tokens=offset,
            window_len=window_len,
            device=x0.device
        )

    def forward(self, x0, x1, cond, kv_cache=None):
        local_block_mask = self.get_block_mask(x0, x1, kv_cache, self.local_window)
        global_block_mask = self.get_block_mask(x0, x1, kv_cache, self.global_window)
        cond0, cond1 = self.cond_proj(cond).chunk(2, dim=-1)
        for layer_idx, block in enumerate(self.blocks):
            block_mask = local_block_mask if self.local_layers[layer_idx] else global_block_mask
            x0, x1 = block(x0, x1, cond0, cond1, block_mask, kv_cache)
        return x0, x1
