import sys
import os

import torch

# Add owl-vaes submodule path to Python path
owl_vae_path = os.path.join(os.path.dirname(__file__), "..", "owl-vaes")
sys.path.append(owl_vae_path)

from owl_vaes.utils.proxy_init import load_proxy_model

def get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,m,d]
        b,n,m,d = x.shape
        x = x.view(b*n,m,d).contiguos()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,3,256,256]
        x = x.view(b,n,-1,-1,-1).contiguous()

        return x