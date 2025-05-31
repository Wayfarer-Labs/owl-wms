import sys
import os

import torch
from diffusers import AutoencoderDC

sys.path.append("./owl-vaes")
from owl_vaes.utils.proxy_init import load_proxy_model

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only():
        model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
        model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
        del model.encoder
        return model.decoder

@torch.no_grad()
def _make_batched_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,m,d]
        b,n,m,d = x.shape
        x = x.view(b*n,m,d).contiguous()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,3,256,256]
        _,c,h,w = x.shape
        x = x.view(b,n,c,h,w).contiguous()

        return x
    return decode

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