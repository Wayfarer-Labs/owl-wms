"""
Shortcut sampler, with cache!
"""

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache

def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha

class CacheShortcutSampler:
    """
    Shortcut CFG sampler builds cache with 4 step diffusion.
    Samples new frames in 1 step.

    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, window_length = 60, num_frames = 60, only_return_generated = False):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, history, keyframe, mouse, btn, decode_fn = None, scale = 1):
        # dummy_batch is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        history = history[:,:self.window_length]
        new_frames = []
        alpha = 0.25 # This number is special for our sampler

        # Extended fake controls to use during sampling
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, num_frames + self.window_length)

        # Generate cache over history
        noisy_history = zlerp(history.clone(), alpha)
        ts = torch.ones_like(noisy_history[:,:,0,0,0]) * alpha
        d = torch.ones_like(noisy_history[:,:,0,0,0]) * round(1./alpha)
        ts_single = ts[:,0].unsqueeze(1)
        d_single = d[:,0].unsqueeze(1)

        cache = KVCache(model.config)
        cache.reset(history.shape[0])

        cache.enable_cache_updates()
        _ = model.sample(noisy_history, keyframe, mouse, btn, cache, ts, d)
        cache.disable_cache_updates()

        # Cache is now built!
        
        for frame_idx in tqdm(range(num_frames)):
            cache.truncate(1) # Drop first frame

            # Generate new frame
            cache.disable_cache_updates()
            mouse = extended_mouse[:,self.window_length+frame_idx].unsqueeze(1)
            btn = extended_btn[:,self.window_length+frame_idx].unsqueeze(1)
            new_frame = model.sample(None, keyframe, mouse, btn, cache) # [b,1,c,h,w]
            new_frames.append(new_frame)
            
            # Add that frame to the cache
            cache.enable_cache_updates()
            new_frame_noisy = zlerp(new_frame, alpha)
            _ = model.sample(new_frame_noisy, keyframe, mouse, btn, cache, ts_single, d_single)

        new_frames = torch.cat(new_frames, dim = 1)
        x = torch.cat([history,new_frames], dim = 1)

        if self.only_return_generated:
            x = x[:,-num_frames:]
            extended_mouse = extended_mouse[:,-num_frames:]
            extended_btn = extended_btn[:,-num_frames:]

        if decode_fn is not None:
            x = x * scale 
            x = decode_fn(x)
    
        return x, extended_mouse, extended_btn

class WindowShortcutSampler:
    """
    Same as above but with no cache

    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, window_length = 60, num_frames = 60, only_return_generated = False):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, history, keyframe, mouse, btn, decode_fn = None, scale = 1):
        # dummy_batch is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        history = history[:,:self.window_length]
        new_frames = []
        alpha = 0.25 # This number is special for our sampler

        # Extended fake controls to use during sampling
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, num_frames + self.window_length)

        # Generate cache over history
        noisy_history = zlerp(history.clone(), alpha)
        ts = torch.ones_like(noisy_history[:,:,0,0,0]) * alpha
        d = torch.ones_like(noisy_history[:,:,0,0,0]) * round(1./alpha)
        ts_single = ts[:,0].unsqueeze(1)
        d_single = d[:,0].unsqueeze(1)

        cache = KVCache(model.config)
        cache.reset(history.shape[0])

        cache.enable_cache_updates()
        _ = model.sample(noisy_history, keyframe, mouse, btn, cache, ts, d)
        cache.disable_cache_updates()

        # Cache is now built!
        
        for frame_idx in tqdm(range(num_frames)):
            cache.truncate(1) # Drop first frame

            # Generate new frame
            cache.disable_cache_updates()
            mouse = extended_mouse[:,self.window_length+frame_idx].unsqueeze(1)
            btn = extended_btn[:,self.window_length+frame_idx].unsqueeze(1)
            new_frame = model.sample(None, keyframe, mouse, btn, cache) # [b,1,c,h,w]
            new_frames.append(new_frame)
            
            # Add that frame to the cache
            cache.enable_cache_updates()
            new_frame_noisy = zlerp(new_frame, alpha)
            _ = model.sample(new_frame_noisy, keyframe, mouse, btn, cache, ts_single, d_single)

        new_frames = torch.cat(new_frames, dim = 1)
        x = torch.cat([history,new_frames], dim = 1)

        if self.only_return_generated:
            x = x[:,-num_frames:]
            extended_mouse = extended_mouse[:,-num_frames:]
            extended_btn = extended_btn[:,-num_frames:]

        if decode_fn is not None:
            x = x * scale 
            x = decode_fn(x)
    
        return x, extended_mouse, extended_btn