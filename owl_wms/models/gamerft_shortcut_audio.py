"""
Shortcut simple with audio
"""

import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..nn.embeddings import (
    TimestepEmbedding,
    StepEmbedding,
    ControlEmbedding,
    LearnedPosEnc
)
from ..nn.attn import UViT, FinalLayer
from ..nn.mmattn import MMUViT
from ..utils import freeze

class ShortcutGameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = UViT(config)
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)

        self.step_embed = StepEmbedding(config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.audio_proj = nn.Linear(config.audio_channels, config.d_model, bias = False)
        self.audio_proj_out = nn.Linear(config.d_model, config.audio_channels, bias = False)

        self.config = config

    def sample(self, x, audio, mouse, btn, kv_cache = None, t = None, d = None):
        """
        This is a function that largely abstracts
        away most things for the specific case where
        you are only generating the one next token

        The return is one step sample always
        """

        b,n,c,h,w = x.shape
        if t is None:
            t = torch.ones_like(x[:,:,0,0,0])
        if d is None:
            d = torch.ones_like(x[:,:,0,0,0])

        pred_x, pred_audio = self.forward(x, audio, t, mouse, btn, d, kv_cache)
        return x - pred_x, audio - pred_audio

    def forward(self, x, audio, t, mouse, btn, d, kv_cache = None):
        # x is [b,n,c,h,w]
        # a is [b,c,n]
        # t is [b,n]
        # d is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)
        d_cond = self.step_embed(d)

        cond = ctrl_cond + t_cond + d_cond # [b,n,d]

        audio = self.audio_proj(audio.transpose(-1,-2))[:,:,None] # -> [b,n,1,d]
        
        b,n,c,h,w = x.shape
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')
        x = self.proj_in(x)
        x = eo.rearrange(x, 'b (n f) d -> b n f d', n = n)
        x = torch.cat([x, audio], dim=-2) # [b,n,f,d]
        x = eo.rearrange(x, 'b n f d -> b (n f) d')

        x = self.transformer(x, cond, kv_cache)

        x = eo.rearrange(x, 'b (n f) c -> b n f w', n=n)
        audio = x[:,:,-1] # [b,n,d]
        x = x[:,:,:-1] # [b,n,f,d]

        x = eo.rearrange(x, 'b n f d -> b (n f) d')
        x = self.proj_out(x, cond) # -> [b,n*hw,c]
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', n=n,h=h,w=w)
        audio = self.audio_proj_out(audio).transpose(-1,-2) # [b,d,n]

        return x, audio

def sample_discrete_timesteps(steps, eps = 1.0e-6):
    # steps is Tensor([1,4,2,64,16]) as an example
    b,n = steps.shape

    ts_list = []
    ts = torch.rand(b, n, device=steps.device, dtype=steps.dtype) * (steps - eps)
    ts = ts.clamp(eps).ceil() / steps
    """
    Example, if d was all 2, ts would be [0,2]
    so do clamp, then ceil will be 1 or 2 (0, 2]
    then do t / 2 and get 0.5 or 1.0, our desired timesteps
    """
    return ts

def sample_steps(b, n, device, dtype, min_val = 0):
    valid = torch.tensor([2**i for i in range(min_val, 8)]) # [1,2,...,128]
    inds = torch.randint(low=0,high=len(valid), size = (b,n))
    steps = valid[inds].to(device=device,dtype=dtype)
    return steps

#@torch.compile()
@torch.no_grad()
def get_sc_targets(ema, x, audio, mouse, btn, cfg_scale):
    steps_slow = sample_steps(x.shape[0], x.shape[1], x.device, x.dtype, min_val = 1)
    steps_fast = steps_slow / 2

    dt_slow = 1./steps_slow
    dt_fast = 1./steps_fast

    def expand(t):
        #b,c,h,w = x.shape
        #t = eo.repeat(t,'b -> b c h w',c=c,h=h,w=w)
        #return t
        return t[:,:,None,None,None]

    ts = sample_discrete_timesteps(steps_fast)
    cfg_mask = torch.isclose(steps_slow, torch.ones_like(steps_slow)*128)
    cfg_mask = expand(cfg_mask) # -> [b,n,1,1,1]

    null_mouse = torch.zeros_like(mouse)
    null_btn = torch.zeros_like(btn)

    pred_1_x_uncond, pred_1_a_uncond = ema(x, audio, ts, null_mouse, null_btn, steps_slow)
    pred_1_x_cond, pred_1_a_cond = ema(x, audio, ts, mouse, btn, steps_slow)
    pred_1_x_cfg = pred_1_x_uncond + cfg_scale * (pred_1_x_cond - pred_1_x_uncond)
    pred_1_a_cfg = pred_1_a_uncond + cfg_scale * (pred_1_a_cond - pred_1_a_uncond)
    pred_1_x = torch.where(cfg_mask, pred_1_x_cfg, pred_1_x_cond)
    pred_1_a = pred_1_a_cfg if cfg_mask.any() else pred_1_a_cond

    x_new = x - pred_1_x * expand(dt_slow)
    audio_new = audio - pred_1_a * dt_slow[:,None,:]
    ts_new = ts - dt_slow

    pred_2_x_uncond, pred_2_a_uncond = ema(x_new, audio_new, ts_new, null_mouse, null_btn, steps_slow)
    pred_2_x_cond, pred_2_a_cond = ema(x_new, audio_new, ts_new, mouse, btn, steps_slow)
    pred_2_x_cfg = pred_2_x_uncond + cfg_scale * (pred_2_x_cond - pred_2_x_uncond)
    pred_2_a_cfg = pred_2_a_uncond + cfg_scale * (pred_2_a_cond - pred_2_a_uncond)
    pred_2_x = torch.where(cfg_mask, pred_2_x_cfg, pred_2_x_cond)
    pred_2_a = pred_2_a_cfg if cfg_mask.any() else pred_2_a_cond

    pred_x = 0.5 * (pred_1_x + pred_2_x)
    pred_a = 0.5 * (pred_1_a + pred_2_a)
    return (pred_x, pred_a), steps_fast, ts

class ShortcutGameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = ShortcutGameRFTCore(config)
        self.cfg_prob = config.cfg_prob

        self.sc_frac = 0.25
        self.sc_max_steps = 128
        self.cfg_scale = 1.3

        self.config = config
        
    def get_sc_loss(self, x, audio, mouse, btn, ema):
        (target_x, target_a), steps, ts = get_sc_targets(ema, x, audio, mouse, btn, self.cfg_scale)
        pred_x, pred_a = self.core(x, audio, ts, mouse, btn, steps)
        sc_loss_x = F.mse_loss(pred_x, target_x)
        sc_loss_a = F.mse_loss(pred_a, target_a)
        return sc_loss_x + sc_loss_a

    def forward(self, x, audio, mouse, btn, ema):
        # x is [b,n,c,h,w]
        # audio is [b,c,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        with torch.no_grad():
            _,n,c,h,w = x.shape

            # Split batches between consistency/rf 
            b = int(len(x) * (1 - self.sc_frac))
            x,x_sc = x[:b], x[b:]
            audio,audio_sc = audio[:b], audio[b:]
            mouse,mouse_sc = mouse[:b], mouse[b:]
            btn,btn_sc = btn[:b], btn[b:]

            # Apply classifier-free guidance dropout
            if self.cfg_prob > 0.0:
                mask = torch.rand(b, device=x.device) <= self.cfg_prob
            null_mouse = torch.zeros_like(mouse)
            null_btn = torch.zeros_like(btn)
            
            # Where mask is True, replace with zeros
            mouse = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_mouse, mouse)
            btn = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_btn, btn)
        
            d = torch.ones_like(x[:,:,0,0,0])*self.sc_max_steps
            ts = sample_discrete_timesteps(d)
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            ts_exp = eo.repeat(ts, 'b n -> b n 1 1 1')
            z_x = torch.randn_like(x)
            z_a = torch.randn_like(audio)

            lerpd_x = x * (1. - ts_exp) + z_x * ts_exp
            lerpd_a = audio * (1. - ts[:,None,:]) + z_a * ts[:,None,:]
            target_x = z_x - x
            target_a = z_a - audio
        
        pred_x, pred_a = self.core(lerpd_x, lerpd_a, ts, mouse, btn, d)
        diff_loss_x = F.mse_loss(pred_x, target_x)
        diff_loss_a = F.mse_loss(pred_a, target_a)
        sc_loss = self.get_sc_loss(x_sc, audio_sc, mouse_sc, btn_sc, ema)

        return diff_loss_x + diff_loss_a, sc_loss