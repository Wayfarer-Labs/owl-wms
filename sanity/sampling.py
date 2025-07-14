from owl_wms.configs import Config
from owl_wms.models.gamerft_audio import GameRFTAudioCore
from owl_wms.data import get_loader
from owl_wms.sampling.schedulers import get_sd3_euler
from owl_wms.sampling.av_window import zlerp
from owl_wms.nn.kv_cache import KVCache

cfg  = Config.from_yaml("configs/av_v2_4x4.yml")
ckpt_path = "/mnt/data/checkpoints/owl-wms/backup/step_100000.pt"

import torch
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)['ema']

def prefix_filter(ckpt, prefix):
    return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

ckpt = prefix_filter(ckpt, "ema_model.module.core.")
torch.save(ckpt, "core.pt")
core = GameRFTAudioCore(cfg.model)
core.load_state_dict(ckpt)

from owl_wms.data import get_loader

loader = get_loader(cfg.train.data_id, batch_size = 1, **cfg.train.data_kwargs)

import torchvision.utils as vutils

import torch

# Get a batch from the loader
batch_vid, batch_audio, batch_mouse, batch_btn = next(iter(loader))
batch_vid = batch_vid.bfloat16() / cfg.train.vae_scale
batch_audio = batch_audio.bfloat16() / cfg.train.audio_vae_scale
batch_mouse = batch_mouse.bfloat16()
batch_btn = batch_btn.bfloat16()

# Move to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
core = core.to(device).eval().bfloat16()
batch_vid = batch_vid.to(device)
batch_audio = batch_audio.to(device)
batch_mouse = batch_mouse.to(device)
batch_btn = batch_btn.to(device)

window_length = 60
n_steps = 10

# Truncate to window_length if needed
batch_vid = batch_vid[:, :window_length]
batch_audio = batch_audio[:, :window_length]
batch_mouse = batch_mouse[:, :window_length]
batch_btn = batch_btn[:, :window_length]

# Replace last frame and last audio with noise
batch_vid = batch_vid.clone()
batch_audio = batch_audio.clone()
batch_vid[:, -1] = torch.randn_like(batch_vid[:, -1])
batch_audio[:, -1] = torch.randn_like(batch_audio[:, -1])

# Prepare time steps
ts = torch.ones(batch_vid.shape[0], window_length, device=device, dtype=batch_vid.dtype)

dt = get_sd3_euler(n_steps)

cache_cond = KVCache(core.config)
cache_uncond = KVCache(core.config)

cache_cond.reset(1)
cache_uncond.reset(1)

# Denoising loop for the last frame only
with torch.no_grad():
    # Noise history frames using zlerp
    batch_vid[:,:-1] = zlerp(batch_vid[:,:-1], 0.2)
    batch_audio[:,:-1] = zlerp(batch_audio[:,:-1], 0.2)
    
    # Set timesteps for history to noise level
    ts[:,:-1] = 0.2

    # Create masks for conditional and unconditional branches
    b = batch_vid.shape[0]
    uncond_mask = torch.zeros(b, dtype=torch.bool, device=batch_vid.device)
    cond_mask = torch.ones(b, dtype=torch.bool, device=batch_vid.device)

    
    # First get prediction without any caching
    cache_cond.enable_cache_updates()
    pred_video_no_cache, _ = core(
        batch_vid, 
        batch_audio, 
        ts, 
        batch_mouse, 
        batch_btn, 
        has_controls=cond_mask,
        kv_cache=cache_cond
    )
    cache_cond.disable_cache_updates()
    cache_cond.truncate(1, front = True)
    """

    cache_uncond.enable_cache_updates()
    pred_video_no_cache, _ = core(
        batch_vid, 
        batch_audio, 
        ts, 
        batch_mouse, 
        batch_btn, 
        has_controls=uncond_mask,
        kv_cache=cache_uncond
    #)
    #cache_uncond.disable_cache_updates()
    #cache_uncond.truncate(1, front = True)
    """

    #cache_cond.truncate(1, front = True)
    # Save last frame prediction
    last_frame_pred_no_cache = pred_video_no_cache[:,-1].clone()

    # Get prediction for last frame using cache
    pred_video_with_cache, _ = core(
        batch_vid[:,-1:],
        batch_audio[:,-1:], 
        ts[:,-1:],
        batch_mouse[:,-1:],
        batch_btn[:,-1:],
        has_controls=cond_mask,
        kv_cache=cache_cond
    )
    last_frame_pred_with_cache = pred_video_with_cache[:,-1]

    # Check if predictions match
    diff = (last_frame_pred_no_cache - last_frame_pred_with_cache).abs().mean()
    print(f"Difference between cached and non-cached predictions: {diff}")
    exit()

    #cache_cond.truncate(1)
    #cache_uncond.truncate(1)

    for step_idx in range(n_steps):
        # Get unconditional predictions
        pred_video_uncond, pred_audio_uncond = core(
            batch_vid[:,-1:], batch_audio[:,-1:], ts[:,-1:], batch_mouse[:,-1:], batch_btn[:,-1:], has_controls=uncond_mask, kv_cache=cache_uncond
        )
        
        # Get conditional predictions
        pred_video_cond, pred_audio_cond = core(
            batch_vid[:,-1:], batch_audio[:,-1:], ts[:,-1:], batch_mouse[:,-1:], batch_btn[:,-1:], has_controls=cond_mask, kv_cache=cache_cond
        )

        # Apply CFG
        cfg_scale = 1.3
        pred_video = pred_video_uncond + cfg_scale * (pred_video_cond - pred_video_uncond)
        pred_audio = pred_audio_uncond + cfg_scale * (pred_audio_cond - pred_audio_uncond)
        
        # Only update the last frame
        batch_vid[:, -1] = batch_vid[:, -1] - pred_video[:, -1] * dt[step_idx]
        batch_audio[:, -1] = batch_audio[:, -1] - pred_audio[:, -1] * dt[step_idx]
        ts[:, -1] = ts[:, -1] - dt[step_idx]

# Decode the denoised last frame
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

decoder = get_decoder_only(cfg.train.vae_id, cfg.train.vae_cfg_path, cfg.train.vae_ckpt_path)
decoder = decoder.to(device).eval().bfloat16()
decode_fn = make_batched_decode_fn(decoder, batch_size=4)

# The decoded image expects [b, n, c, h, w], so select the last and second last frames
last_frame = batch_vid[:, -1:]
prev_frame = batch_vid[:, -2:-1]

# Decode last frame
print(last_frame.min(), last_frame.max(), last_frame.std(), last_frame.mean())
decoded_last = decode_fn(last_frame * cfg.train.vae_scale)  # shape: [b, 1, c, h, w]
img_last = decoded_last[0, 0].float().cpu()
print(img_last.min(), img_last.max(), img_last.std(), img_last.mean())
img_last = (img_last + 1) / 2
img_last = img_last.clamp(0, 1)
vutils.save_image(img_last, "sample.png")

# Decode second last frame
print(prev_frame.min(), prev_frame.max(), prev_frame.std(), prev_frame.mean())
decoded_prev = decode_fn(prev_frame * cfg.train.vae_scale)  # shape: [b, 1, c, h, w]
img_prev = decoded_prev[0, 0].float().cpu()
print(img_prev.min(), img_prev.max(), img_prev.std(), img_prev.mean())
img_prev = (img_prev + 1) / 2
img_prev = img_prev.clamp(0, 1)
vutils.save_image(img_prev, "prev_frame.png")

