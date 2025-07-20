import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP

import einops as eo

from .modulation import AdaLN, Gate
from .rope import AVRoPE
from .mmattn import MMAttn

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

class MMDiTBlock2(nn.Module):
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

        # Stream 2 - AdaLN and gating (was LayerNorm)
        self.adaln1_2 = AdaLN(dim)
        self.gate1_2 = Gate(dim)
        self.adaln2_2 = AdaLN(dim)
        self.gate2_2 = Gate(dim)

    @torch.compile
    def forward(self, x, y, cond, block_mask = None, kv_cache = None):
        res1_x = x.clone()
        res1_y = y.clone()
        
        # First attention block
        x = self.adaln1_1(x, cond)
        y = self.adaln1_2(y, cond)
        
        x, y = self.attn(x, y, block_mask, kv_cache)
        
        x = self.gate1_1(x, cond)
        y = self.gate1_2(y, cond)
        
        x = res1_x + x
        y = res1_y + y
        
        # Second MLP block
        res2_x = x.clone()
        res2_y = y.clone()
        
        x = self.adaln2_1(x, cond)
        y = self.adaln2_2(y, cond)
        
        x = self.mlp_1(x)
        y = self.mlp_2(y)
        
        x = self.gate2_1(x, cond)
        y = self.gate2_2(y, cond)
        
        x = res2_x + x
        y = res2_y + y

        return x, y

class MMDIT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList([MMDiTBlock2(config) for _ in range(config.n_layers)])

    def get_block_mask(self, x, y, kv_cache):
        n_tokens = x.shape[1] + y.shape[1]
        n_tokens_per_frame = self.config.tokens_per_frame
        n_audio_tokens = self.config.tokens_per_frame - self.config.sample_size**2

        return create_block_causal_mask_with_mm(n_tokens, n_tokens_per_frame, n_audio_tokens).to(x.device,x.dtype)

    def forward(self, x, y, cond, kv_cache = None):
        block_mask = self.get_block_mask(x, y, kv_cache)
        for block in self.blocks:
            x,y = block(x, y, cond, block_mask, kv_cache)
        return x,y
 