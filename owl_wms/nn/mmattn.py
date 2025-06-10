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

"""
This code makes the assumption that there are some
tokens from another modality that must always be attended to
"""

def create_block_causal_mask_with_mm(tokens, context_tokens, tokens_per_frame):
    frames = tokens // tokens_per_frame
    
    # Create base causal mask, nothing is masked
    total_tokens = tokens + context_tokens
    mask = torch.zeros(total_tokens, total_tokens)
    
    # Allow attention within each frame
    for i in range(frames):
        start = i * tokens_per_frame
        end = (i + 1) * tokens_per_frame
        mask[start:end, end:tokens] = True # Can't see future frames
    
    # Context tokens can attend to everything (no masking needed)
    # Regular tokens can attend to all context tokens (no masking needed)
    # The zeros in mask[tokens:, :] allow context to attend to everything
    # The zeros in mask[:, tokens:] allow tokens to attend to all context
        
    return mask

class MMAttn(nn.Module):
    """
    MMDiT style attention
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv_1 = nn.Linear(config.d_model, 3 * config.d_model)
        self.qkv_2 = nn.Linear(config.d_model, 3 * config.d_model)

        self.out_1 = nn.Linear(config.d_model, config.d_model)
        self.out_2 = nn.Linear(config.d_model, config.d_model)

        self.qk_norm_1 = QKNorm(config.d_model // config.n_heads)
        self.qk_norm_2 = QKNorm(config.d_model // config.n_heads)

        self.config = config
        self.causal = config.causal

    def split(self, qkv):
        return eo.rearrange(qkv, 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)

    def merge(self, x):
        return eo.rearrange(x, 'b h n d -> b n (h d)')

    def forward(self, x_1, x_2, kv_cache=None):
        n1 = x_1.shape[1]

        q1,k1,v1 = self.split(self.qkv_1(x_1))
        q2,k2,v2 = self.split(self.qkv_2(x_2))

        q1,k1 = self.qk_norm_1(q1,k1)
        q2,k2 = self.qk_norm_2(q2,k2)

        if not self.causal or (kv_cache is not None and len(kv_cache) > 0):
            mask = None
        else:
            mask = create_block_causal_mask_with_mm(x_1.shape[1], x_2.shape[1], self.config.tokens_per_frame)
            mask = mask.to(device=x_1.device,dtype=x_1.dtype)
            mask = mask.unsqueeze(0).repeat(x_1.shape[0],1,1)

        if kv_cache is not None:
            if len(kv_cache) > 0:
                old_k, old_v = kv_cache.get(self.layer_ind)
                
                new_k = torch.cat([old_k, k1], dim=2).contiguous()
                new_v = torch.cat([old_v, v1], dim=2).contiguous()
            else:
                new_k = k1.contiguous()
                new_v = v1.contiguous()

            if kv_cache.should_update:
                kv_cache.update(new_k, new_v, self.layer_ind)

            k = torch.cat([new_k, k2], dim=-2)
            v = torch.cat([new_v, v2], dim=-2)
            q = torch.cat([q1, q2], dim=-2)

            x = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
            x = x[:,:,-q.shape[2]:] # Only keep latest outputs
            x = self.merge(x)
        else:
            q = torch.cat([q1,q2],dim=-2)
            k = torch.cat([k1,k2],dim=-2) 
            v = torch.cat([v1,v2],dim=-2)

            x = F.scaled_dot_product_attention(q,k,v, attn_mask = mask)
            x = self.merge(x)

        x_1, x_2 = x[:,:n1], x[:,n1:]
        x_1 = self.out_1(x_1)
        x_2 = self.out_2(x_2)

        return x_1, x_2

class MMDiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.attn = MMAttn(config)
        
        self.mlp_1 = MLP(config)
        self.mlp_2 = MLP(config)

        # Stream 1 - AdaLN and gating
        self.adaln1_1 = AdaLN(dim)
        self.gate1_1 = Gate(dim)
        self.adaln2_1 = AdaLN(dim)
        self.gate2_1 = Gate(dim)

        # Stream 2 - Standard LayerNorm
        self.ln1_2 = nn.LayerNorm(dim)
        self.ln2_2 = nn.LayerNorm(dim)

    def forward(self, x, y, cond, kv_cache = None):
        res1_x = x.clone()
        res1_y = y.clone()
        
        # First attention block
        x = self.adaln1_1(x, cond)
        y = self.ln1_2(y)
        
        x, y = self.attn(x, y, kv_cache)
        
        x = self.gate1_1(x, cond)
        
        x = res1_x + x
        y = res1_y + y
        
        # Second MLP block
        res2_x = x.clone()
        res2_y = y.clone()
        
        x = self.adaln2_1(x, cond)
        y = self.ln2_2(y)
        
        x = self.mlp_1(x)
        y = self.mlp_2(y)
        
        x = self.gate2_1(x, cond)
        
        x = res2_x + x
        y = res2_y + y

        return x, y

class MMUViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for i in range(config.n_layers):
            blocks.append(MMDiTBlock(config))
            blocks[-1].attn.layer_ind = i

        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(nn.Linear(config.d_model * 2, config.d_model))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, y, cond, kv_cache = None):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x,y = self.blocks[i](x, y, cond, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x,y = self.blocks[mid_idx](x, y, cond, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]
            
            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)
            
            x,y = self.blocks[i](x, y, cond, kv_cache)

        return x


def test_fwd_with_cache():
    from ..configs import TransformerConfig
    from .kv_cache import KVCache

    import matplotlib.pyplot as plt

    cfg = TransformerConfig(
        None,
        6,
        6,
        384,
        1,
        128,
        4,
        0.1,
        8,
        16,
        True
    )

    model = MMUViT(cfg).bfloat16().cuda()

    NUM_FRAMES = 10
    x = torch.randn(1,16*NUM_FRAMES,384).bfloat16().cuda()
    y = torch.randn(1,16,384).bfloat16().cuda()
    cond=torch.randn(1,16,384).bfloat16().cuda()

    cache = KVCache(cfg).to(device='cuda',dtype=torch.bfloat16)
    cache.reset(1)
    
    with torch.no_grad():
        cache.enable_cache_updates()
        out = model(x,y,cond,cache)

        new_x = torch.randn(1,16,384).bfloat16().cuda()
        cond = torch.randn(1,1,384).bfloat16().cuda()

        print(len(cache))
        print(cache.cache[0][0].shape)
        new_out = model(new_x, y, cond, cache)

        print(len(cache))
        print(cache.cache[0][0].shape)

def test_mask():
    import matplotlib.pyplot as plt

    n_frames = 10
    n_tok_per_frame = 16
    n_context = 16

    mask = create_block_causal_mask_with_mm(n_frames*n_tok_per_frame, n_context, n_tok_per_frame)

    plt.figure(figsize=(10,10))
    plt.imshow(mask.float().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.title(f'Block Causal Mask with MM ({n_frames*n_tok_per_frame} tokens, {n_context} context, {n_tok_per_frame} per frame)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position') 
    plt.savefig('test_mm_mask.png')
    plt.close()


if __name__ == "__main__":
    test_mask()