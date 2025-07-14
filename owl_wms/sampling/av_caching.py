import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache

class AVCachingSampler:
    """
    Caching sampler that uses KV cache to avoid recomputing attention for previous frames.
    Samples new frames one by one, using cached keys/values from previous frames.

    :param n_steps: Number of diffusion steps for each frame (default 4)
    :param cfg_scale: CFG scale for each frame
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    :param cache_after_denoise: Whether to cache clean frame after denoising (vs caching final noisy frame)
    """
    def __init__(self, n_steps=4, num_frames=60, window_length = 60,only_return_generated=False):
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated
        self.window_length = window_length

    @torch.no_grad()
    def __call__(self, model, dummy_batch, audio, mouse, btn, decode_fn=None, audio_decode_fn=None, image_scale=1, audio_scale=1):
        # Initialize outputs
        clean_history = dummy_batch.clone()
        clean_audio_history = audio.clone()
        
        # Get extended controls
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, self.num_frames + dummy_batch.shape[1])

        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(dummy_batch.shape[0])

        # Cache context frames
        kv_cache.enable_cache_updates()
        ts = torch.zeros_like(clean_history[:,:,0,0,0])
        _ = model(clean_history, clean_audio_history, ts, mouse, btn, kv_cache=kv_cache)
        kv_cache.disable_cache_updates()

        dt = 1. / self.n_steps

        # Generate new frames
        for frame_idx in tqdm(range(self.num_frames)):
            # Initialize new frame with noise
            new_frame = torch.randn_like(clean_history[:,0:1])
            new_audio = torch.randn_like(clean_audio_history[:,0:1])

            curr_mouse = extended_mouse[:,frame_idx:frame_idx+1]
            curr_btn = extended_btn[:,frame_idx:frame_idx+1]

            b = new_frame.shape[0]
            ts = torch.ones_like(new_frame[:,0,0,0,0]).unsqueeze(1)

            if kv_cache.n_frames() >= self.window_length:
                kv_cache.truncate(1, front=False)

            # Denoise
            for step in range(self.n_steps):
                pred_video, pred_audio = model(new_frame, new_audio, ts, curr_mouse, curr_btn, kv_cache=kv_cache)

                # Update
                new_frame = new_frame - pred_video * dt
                new_audio = new_audio - pred_audio * dt
                ts = ts - dt

            # Cache clean frame with noise level 1.0
            kv_cache.enable_cache_updates()
            ts = torch.zeros_like(new_frame[:,0,0,0,0]).unsqueeze(1)
            _ = model(new_frame, new_audio, ts, curr_mouse, curr_btn, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()

            # Add to history
            clean_history = torch.cat([clean_history, new_frame], dim=1)
            clean_audio_history = torch.cat([clean_audio_history, new_audio], dim=1)

        # Return only generated frames if specified
        if self.only_return_generated:
            clean_history = clean_history[:,-self.num_frames:]
            clean_audio_history = clean_audio_history[:,-self.num_frames:]
            extended_mouse = extended_mouse[:,-self.num_frames:]
            extended_btn = extended_btn[:,-self.num_frames:]

        # Decode if decoders provided
        if decode_fn is not None:
            clean_history = clean_history * image_scale
            clean_history = decode_fn(clean_history)

        if audio_decode_fn is not None:
            clean_audio_history = clean_audio_history * audio_scale
            clean_audio_history = audio_decode_fn(clean_audio_history)

        return clean_history, clean_audio_history, extended_mouse, extended_btn

class AVCachingOneStepSampler:
    """
    Same as above but distinctly one step.

    :param num_frames: Number of new frames to sample
    :param window_length: Number of frames in sliding window
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, num_frames=60, window_length = 60,only_return_generated=False):
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated
        self.window_length = window_length

    @torch.no_grad()
    def __call__(self, model, dummy_batch, audio, mouse, btn, decode_fn=None, audio_decode_fn=None, image_scale=1, audio_scale=1):
        # Mostly identical to rollout manager
        kv_cache = KVCache(model.config)
        kv_cache.reset(dummy_batch.shape[0])

        # === Prepare generation inputs ===
        video = dummy_batch
        context_frames = video.shape[1]
        ext_mouse, ext_btn = batch_permute_to_length(mouse, btn, self.num_frames + context_frames)
        rollout_mouse = ext_mouse[:,context_frames:]
        rollout_btn = ext_btn[:,context_frames:]
        ts = torch.zeros_like(video[:,:,0,0,0])

        # === cache context frames ===
        kv_cache.enable_cache_updates()
        model(video, audio, ts, mouse, btn, kv_cache=kv_cache)
        kv_cache.disable_cache_updates()

        # === rollouts! ===
        for frame_idx in tqdm(range(self.num_frames)):
            # inputs for this frame
            new_frame = torch.randn_like(video[:,0:1])
            new_audio = torch.randn_like(audio[:,0:1])
            new_mouse = rollout_mouse[:,frame_idx:frame_idx+1]
            new_btn = rollout_btn[:,frame_idx:frame_idx+1]
            ts = torch.ones_like(new_frame[:,0:1,0,0,0])

            # Eject oldest frame
            kv_cache.truncate(1, front=False)

            # Denoise
            pred_video, pred_audio = model(new_frame, new_audio, ts, new_mouse, new_btn, kv_cache=kv_cache)

            # Update
            new_frame = new_frame - pred_video
            new_audio = new_audio - pred_audio

            # Update cache
            kv_cache.enable_cache_updates()
            model(new_frame, new_audio, ts * 0.0, new_mouse, new_btn, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()

            # Add to history
            video = torch.cat([video, new_frame], dim=1)
            audio = torch.cat([audio, new_audio], dim=1)
        
        if self.only_return_generated:
            video = video[:,-self.num_frames:]
            audio = audio[:,-self.num_frames:]
            ext_mouse = ext_mouse[:,-self.num_frames:]
            ext_btn = ext_btn[:,-self.num_frames:]
        
        if decode_fn is not None:
            video = video * image_scale
            video = decode_fn(video)
        
        if audio_decode_fn is not None:
            audio = audio * audio_scale
            audio = audio_decode_fn(audio)

        return video, audio, ext_mouse, ext_btn