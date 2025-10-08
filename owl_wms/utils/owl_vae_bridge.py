import sys
import os

import torch
from diffusers import AutoencoderDC

sys.path.append("./owl-vaes")
from owl_vaes.utils.proxy_init import load_proxy_model
from owl_vaes.models import get_model_cls
from owl_vaes.configs import Config

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only(vae_id, cfg_path, ckpt_path):
    assert vae_id is None
    cfg = Config.from_yaml(cfg_path).model
    model = get_model_cls(cfg.model_id)(cfg)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
    except:
        model.decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
    del model.encoder
    model = model.decoder
    model = model.bfloat16().cuda().eval()
    return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c,h,w]
        b,n,c,h,w = x.shape
        x = x.view(b*n,c,h,w).contiguous()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,c,h,w]
        _,c,h,w = x.shape
        x = x.view(b,n,c,h,w).contiguous()

        return x
    return decode

@torch.no_grad()
def make_batched_audio_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c] audio samples
        x = x.transpose(1,2)
        b,c,n = x.shape

        batches = x.contiguous().split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b,c,n]
        x = x.transpose(-1,-2).contiguous() # [b,n,2]

        return x
    return decode

@torch.no_grad()
def make_batched_decode_fn_temporal_vae(decoder, batch_size = 8, window_size = 4):
    def decode(latents):
        # Input is [b,n,c,h,w]
        # We assume window size of 4, batch size is ignored
        B, N, C, H, W = latents.shape
        assert N >= window_size, f"{N=} < {window_size=}"

        # Optional but helps if any layers are mode-sensitive
        was_training = decoder.training
        decoder.eval()

        # First window: full decode
        rec0 = decoder(latents[:, :window_size], ignore_nonterminal_frames=False)  # [B, W, C, H, W]
        out = latents.new_empty((B, N, rec0.shape[2], rec0.shape[3], rec0.shape[4]))
        out[:, :window_size] = rec0

        # Subsequent windows: terminal-only
        for i in range(1, N - window_size + 1):
            rec = decoder(latents[:, i:i+window_size], ignore_nonterminal_frames=True)  # [B, 1, C, H, W]
            out[:, window_size + i - 1] = rec[:, -1]  # or rec[:, 0]

        if was_training:
            decoder.train()
        return out
    return decode