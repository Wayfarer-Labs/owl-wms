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
        if vae_id == "dcae":
            model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
            model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
            del model.encoder
            return model.decoder
        else:
            cfg = Config.from_yaml(cfg_path).model
            model = get_model_cls(cfg.model_id)(cfg)
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            del model.encoder
            model = model.decoder
            model = model.bfloat16().cuda().eval()
            return model

def get_encoder_only(vae_id, cfg_path, ckpt_path):
    if vae_id == "dcae":
        model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
        model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
        del model.decoder  # Keep encoder only
        return model.encoder
    elif vae_id == "720pr3dc":
        cfg = Config.from_yaml(cfg_path).model
        model = get_model_cls(cfg.model_id)(cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
        del model.decoder  # Keep encoder only
        model = model.encoder
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
<<<<<<< HEAD
def make_batched_encode_fn(encoder, batch_size=8):
    def encode(x):
        # x is [b,n,c,h,w] RGB frames
        b,n,c,h,w = x.shape
        x = x.view(b*n,c,h,w).contiguous()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(encoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,latent_c,latent_h,latent_w]
        _,latent_c,latent_h,latent_w = x.shape
        x = x.view(b,n,latent_c,latent_h,latent_w).contiguous()

        return x
    return encode
=======
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
>>>>>>> causvid
