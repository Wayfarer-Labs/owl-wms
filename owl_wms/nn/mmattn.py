import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP

import einops as eo

from .modulation import AdaLN, Gate
from .rope import AVRoPE

torch.backends.cuda.enable_flash_sdp(enabled = True)

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

"""
This code makes the assumption that there are some
tokens from another modality that must always be attended to
"""

def create_block_causal_mask_with_mm(tokens, tokens_per_frame, audio_tokens = 1):
    """
    Assumes 1 token per frame for audio
    """
    tokens_per_frame_video = tokens_per_frame - audio_tokens
    tokens_per_frame_audio = audio_tokens

    frames = tokens // tokens_per_frame

    mask_tl = torch.zeros(frames*tokens_per_frame_video, frames*tokens_per_frame_video)
    mask_br = torch.zeros(frames*tokens_per_frame_audio, frames*tokens_per_frame_audio)

    for i in range(frames):
        start = i * tokens_per_frame_video
        end = (i + 1) * tokens_per_frame_video
        mask_tl[start:end, end:] = -float('inf')
        if i == frames - 1:
            mask_tl[start:end, :tokens_per_frame_video] = -float('inf')

    for i in range(frames):
        start = i * tokens_per_frame_audio
        end = (i + 1) * tokens_per_frame_audio
        mask_br[start:end, end:] = -float('inf')
        if i == frames - 1:
            mask_br[start:end, :tokens_per_frame_audio] = -float('inf')

    mask_bl = torch.zeros(frames*tokens_per_frame_audio, frames*tokens_per_frame_video)
    mask_tr = torch.zeros(frames*tokens_per_frame_video, frames*tokens_per_frame_audio)
    
    for i in range(frames):
        start = i * tokens_per_frame_audio
        end = (i + 1) * tokens_per_frame_audio
        start_video = i * tokens_per_frame_video
        end_video = (i + 1) * tokens_per_frame_video

        mask_bl[start:end, end_video:] = -float('inf')
        if i == frames - 1:
            mask_bl[start:end, :tokens_per_frame_video] = -float('inf')
    
    for i in range(frames):
        start = i * tokens_per_frame_video
        end = (i + 1) * tokens_per_frame_video
        start_audio = i * tokens_per_frame_audio
        end_audio = (i + 1) * tokens_per_frame_audio

        mask_tr[start:end, end_audio:] = -float('inf')
        if i == frames - 1:
            mask_tr[start:end, :tokens_per_frame_audio] = -float('inf')

    mask_top = torch.cat([mask_tl, mask_tr], dim=1)
    mask_bottom = torch.cat([mask_bl, mask_br], dim=1)
    mask = torch.cat([mask_top, mask_bottom], dim=0)

    # mask is now [n_query, n_key]
    # make it [b,h,n_query,n_key]
    mask = mask[None,None,:,:]
    
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

        self.rope = AVRoPE(config)

    def split(self, qkv):
        return eo.rearrange(qkv, 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)

    def merge(self, x):
        return eo.rearrange(x, 'b h n d -> b n (h d)')

    @torch.compile
    def forward(self, x_1, x_2, block_mask = None, kv_cache = None):
        """
        For MMDiT we assume kv_cache is a tuple of two caches
        """
        n1 = x_1.shape[1]

        q1,k1,v1 = self.split(self.qkv_1(x_1))
        q2,k2,v2 = self.split(self.qkv_2(x_2))

        q1,k1 = self.qk_norm_1(q1,k1)
        q2,k2 = self.qk_norm_2(q2,k2)

        if not self.causal or (kv_cache is not None and len(kv_cache[0]) > 0):
            mask = None

        if kv_cache is not None:
            if len(kv_cache[0]) > 0:
                old_k1, old_v1 = kv_cache[0].get(self.layer_ind)
                old_k2, old_v2 = kv_cache[1].get(self.layer_ind)
                
                new_k1 = torch.cat([old_k1, k1], dim=2).contiguous()
                new_v1 = torch.cat([old_v1, v1], dim=2).contiguous()
                new_k2 = torch.cat([old_k2, k2], dim=2).contiguous()
                new_v2 = torch.cat([old_v2, v2], dim=2).contiguous()
            else:
                new_k1 = k1.contiguous()
                new_v1 = v1.contiguous()
                new_k2 = k2.contiguous()
                new_v2 = v2.contiguous()

            if kv_cache.should_update:
                kv_cache[0].update(new_k1, new_v1, self.layer_ind)
                kv_cache[1].update(new_k2, new_v2, self.layer_ind)

            q1, q2 = self.rope(q1, q2)
            new_k1, new_k2 = self.rope(new_k1, new_k2)

            q = torch.cat([q1, q2], dim=-2)
            k = torch.cat([new_k1, new_k2], dim=-2)
            v = torch.cat([new_v1, new_v2], dim=-2)

            x = F.scaled_dot_product_attention(q, k, v, attn_mask = block_mask)
            x = x[:,:,-q.shape[2]:] # Only keep latest outputs
            x = self.merge(x)
        else:
            q1, q2 = self.rope(q1,q2)
            k1, k2 = self.rope(k1,k2)

            q = torch.cat([q1,q2],dim=-2)
            k = torch.cat([k1,k2],dim=-2) 
            v = torch.cat([v1,v2],dim=-2)

            x = F.scaled_dot_product_attention(q,k,v, attn_mask = block_mask)
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

    def forward(self, x, y, cond, block_mask = None, kv_cache = None):
        res1_x = x.clone()
        res1_y = y.clone()
        
        # First attention block
        x = self.adaln1_1(x, cond)
        y = self.ln1_2(y)
        
        x, y = self.attn(x, y, block_mask, kv_cache)
        
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

class MMDIT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList([MMDiTBlock(config) for _ in range(config.n_layers)])

    def get_block_mask(self, x, y, kv_cache):
        n_tokens = x.shape[1] + y.shape[1]
        n_tokens_per_frame = self.config.tokens_per_frame
        n_audio_tokens = self.config.tokens_per_frame - config.sample_size**2

        return create_block_causal_mask_with_mm(n_tokens, n_tokens_per_frame, n_audio_tokens)


    def forward(self, x, y, cond, kv_cache = None):
        block_mask = self.get_block_mask(x, y, kv_cache)
        for block in self.blocks:
            x,y = block(x, y, cond, block_mask, kv_cache)
        return x,y

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

    def forward(self, x, y, cond, block_mask = None, kv_cache = None):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x,y = self.blocks[i](x, y, cond, block_mask, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x,y = self.blocks[mid_idx](x, y, cond, block_mask, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]
            
            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)
            
            x,y = self.blocks[i](x, y, cond, block_mask, kv_cache)

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
    n_tok_per_frame = 17
    n_audio_tokens = 1
    total_tokens = n_frames * n_tok_per_frame

    mask = create_block_causal_mask_with_mm(total_tokens, n_tok_per_frame, n_audio_tokens)
    
    # Convert to visualization format: 1 = allowed (white), 0 = blocked (black)
    mask_vis = (mask != -float('inf')).float()

    plt.figure(figsize=(12, 10))
    plt.imshow(mask_vis.cpu().numpy(), cmap='gray', interpolation='nearest')
    plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
    plt.title(f'Block Causal Mask with MM\n({total_tokens} total tokens, {n_tok_per_frame} per frame, {n_audio_tokens} audio per frame)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add grid lines to show frame boundaries
    for i in range(1, n_frames):
        video_boundary = i * (n_tok_per_frame - n_audio_tokens)
        audio_boundary = n_frames * (n_tok_per_frame - n_audio_tokens) + i * n_audio_tokens
        plt.axhline(y=video_boundary - 0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=video_boundary - 0.5, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=audio_boundary - 0.5, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(x=audio_boundary - 0.5, color='blue', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('test_mm_mask.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Mask shape: {mask.shape}")
    print(f"Number of blocked positions: {(mask == -float('inf')).sum().item()}")
    print(f"Number of allowed positions: {(mask != -float('inf')).sum().item()}")


if __name__ == "__main__":
    test_mask()
