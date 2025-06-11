"""
Causal-First RFT With Shortcut objective
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

class ShortcutGameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = MMUViT(config)
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)

        self.step_embed = StepEmbedding(config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.proj_y_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.pos_enc_y = LearnedPosEnc(config.tokens_per_frame, config.d_model)

        self.config = config

    def sample(self, x, y, mouse, btn, kv_cache = None, t = None, d = None):
        """
        This is a function that largely abstracts
        away most things for the specific case where
        you are only generating the one next token

        The return is one step sample always
        """
        if x is None:
            x = torch.randn_like(y)

        b,n,c,h,w = x.shape
        if t is None:
            t = torch.ones_like(x[:,:,0,0,0])
        if d is None:
            d = torch.ones_like(x[:,:,0,0,0])

        return x - self.forward(x, y, t, mouse, btn, d, kv_cache)

    def forward(self, x, y, t, mouse, btn, d, kv_cache = None):
        # x is [b,n,c,h,w]
        # y is [b,1,c,h,w]
        # t is [b,n]
        # d is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)
        d_cond = self.step_embed(d)

        cond = ctrl_cond + t_cond + d_cond # [b,n,d]
        
        b,n,c,h,w = x.shape
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')
        y = eo.rearrange(y, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)

        y = self.proj_y_in(y)
        y = self.pos_enc_y(y)

        x = self.transformer(x, y, cond, kv_cache)
        x = self.proj_out(x, cond) # -> [b,n*hw,c]
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', n=n,h=h,w=w)

        return x

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

class ShortcutGameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = ShortcutGameRFTCore(config)
        self.cfg_prob = config.cfg_prob

        self.ema = None
        self.sc_frac = 0.25
        self.sc_max_steps = 128
        self.cfg_scale = 1.3

        self.config = config
    
    def set_ema(self, ema):
        if hasattr(ema.ema_model, 'module'):
            self.ema = ema.ema_model.module.core
        else:
            self.ema = ema.ema_model.core

    #@torch.compile()
    @torch.no_grad()
    def get_sc_targets(self, x, y, mouse, btn):
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

        pred_1_uncond = self.ema(x, y, ts, null_mouse, null_btn, steps_slow)
        pred_1_cond = self.ema(x, y, ts, mouse, btn, steps_slow)
        pred_1_cfg = pred_1_uncond + self.cfg_scale * (pred_1_cond - pred_1_uncond)
        pred_1 = torch.where(cfg_mask, pred_1_cfg, pred_1_cond)

        x_new = x - pred_1 * expand(dt_slow)
        ts_new = ts - dt_slow

        pred_2_uncond = self.ema(x_new, y, ts_new, null_mouse, null_btn, steps_slow)
        pred_2_cond = self.ema(x_new, y, ts_new, mouse, btn, steps_slow)
        pred_2_cfg = pred_2_uncond + self.cfg_scale * (pred_2_cond - pred_2_uncond)
        pred_2 = torch.where(cfg_mask, pred_2_cfg, pred_2_cond)

        pred = 0.5 * (pred_1 + pred_2)
        return pred, steps_fast, ts
    
    def get_sc_loss(self, x, y, mouse, btn):
        target, steps, ts = self.get_sc_targets(x, y, mouse, btn)
        pred = self.core(x, y, ts, mouse, btn, steps)
        sc_loss = F.mse_loss(pred, target)
        return sc_loss

    def forward(self, x, y, mouse, btn):
        # x is [b,n,c,h,w]
        # y (seed frame) is [b,1,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        _,n,c,h,w = x.shape

        # Split batches between consistency/rf 
        b = int(len(x) * (1 - self.sc_frac))
        x,x_sc = x[:b], x[b:]
        y,y_sc = y[:b], y[b:]
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
        
        with torch.no_grad():
            d = torch.ones_like(x[:,:,0,0,0])*self.sc_max_steps
            ts = sample_discrete_timesteps(d)
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            ts_exp = eo.repeat(ts, 'b n -> b n 1 1 1')
            z = torch.randn_like(x)

            lerpd = x * (1. - ts_exp) + z * ts_exp
            target = z - x
        
        pred = self.core(lerpd, y, ts, mouse, btn, d)
        diff_loss = F.mse_loss(pred, target)
        sc_loss = self.get_sc_loss(x_sc, y_sc, mouse_sc, btn_sc)

        return diff_loss, sc_loss

def test_inference_cache():
    from ..configs import TransformerConfig
    from ..nn.kv_cache import KVCache

    cfg = TransformerConfig(
        None,           # model_id
        6,             # n_layers
        6,             # n_heads
        384,           # d_model
        1,             # patch_size
        128,           # channels
        16,            # sample_size
        0.1,           # cfg_prob
        11,            # n_buttons
        16,            # tokens_per_frame
        10,            # n_frames
        True           # causal
    )

    model = ShortcutGameRFTCore(cfg).bfloat16().cuda()

    NUM_FRAMES = 10
    x = torch.randn(1, NUM_FRAMES, 128, 4, 4).bfloat16().cuda()
    y = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
    mouse = torch.randn(1, NUM_FRAMES, 2).bfloat16().cuda()
    btn = torch.randn(1, NUM_FRAMES, 11).bfloat16().cuda()
    t = torch.full((1, NUM_FRAMES), 0.25, device='cuda', dtype=torch.bfloat16)
    d = torch.full((1, NUM_FRAMES), 4, device='cuda', dtype=torch.bfloat16)

    cache = KVCache(cfg).to(device='cuda', dtype=torch.bfloat16)
    cache.reset(1)

    with torch.no_grad():
        # First pass - generate cache for all frames
        cache.enable_cache_updates()
        out = model(x, y, t, mouse, btn, d, cache)
        print(f"Initial cache length: {len(cache)}")
        print(f"Initial cache shape: {cache.cache[0][0].shape}")

        # Generate single new frame with t=1, d=1
        new_x = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
        new_mouse = torch.randn(1, 1, 2).bfloat16().cuda()
        new_btn = torch.randn(1, 1, 11).bfloat16().cuda()
        new_t = torch.ones(1, 1, device='cuda', dtype=torch.bfloat16)
        new_d = torch.ones(1, 1, device='cuda', dtype=torch.bfloat16)

        # Disable cache updates for inference
        cache.disable_cache_updates()
        new_out = model(new_x, y, new_t, new_mouse, new_btn, new_d, cache)
        print(f"After inference cache length: {len(cache)}")
        print(f"After inference cache shape: {cache.cache[0][0].shape}")

        # Re-enable cache updates and update cache with t=0.25, d=4
        cache.enable_cache_updates()
        new_t = torch.full((1, 1), 0.25, device='cuda', dtype=torch.bfloat16)
        new_d = torch.full((1, 1), 4, device='cuda', dtype=torch.bfloat16)
        new_out = model(new_x, y, new_t, new_mouse, new_btn, new_d, cache)
        print(f"Final cache length: {len(cache)}")
        print(f"Final cache shape: {cache.cache[0][0].shape}")
    
def test_wrapper():
    from ..configs import TransformerConfig
    from ema_pytorch import EMA
    from copy import deepcopy

    cfg = TransformerConfig(
        None,           # model_id
        6,             # n_layers
        6,             # n_heads
        384,           # d_model
        1,             # patch_size
        128,           # channels
        16,            # sample_size
        0.1,           # cfg_prob
        11,            # n_buttons
        16,            # tokens_per_frame
        10,            # n_frames
        True           # causal
    )

    model = ShortcutGameRFT(cfg).bfloat16().cuda()
    ema = EMA(model, beta=0.999,update_after_step=0,update_every=1)
    model.set_ema(ema)

    NUM_FRAMES = 10
    x = torch.randn(4, NUM_FRAMES, 128, 4, 4).bfloat16().cuda()
    y = torch.randn(4, 1, 128, 4, 4).bfloat16().cuda()
    mouse = torch.randn(4, NUM_FRAMES, 2).bfloat16().cuda()
    btn = torch.randn(4, NUM_FRAMES, 11).bfloat16().cuda()

    with torch.no_grad():
        loss_1, loss_2 = model(x, y, mouse, btn)
        print(loss_1, loss_2)

if __name__ == "__main__":
    test_wrapper()
