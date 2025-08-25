from owl_wms.data import get_loader
from owl_wms.configs import Config
from owl_wms import from_pretrained

from owl_wms.utils import batch_permute_to_length
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn
from owl_wms.utils.logging import LogHelper, to_wandb_av
from owl_wms.nn.rope import RoPE

import torch
import gc

cfg_path = "configs/dit_v4_dmd.yml"
ckpt_path = "vid_dit_v4_dmd_7k.pt"

model, decoder = from_pretrained(cfg_path, ckpt_path, return_decoder=True)
model = model.core.eval().cuda().bfloat16()

# Find any RoPE modules in the model and cast their cos and sin back to fp32
def cast_rope_buffers_to_fp32(module):
    for submodule in module.modules():
        if isinstance(submodule, RoPE):
            if hasattr(submodule, "cos"):
                submodule.cos = submodule.cos.float()
            if hasattr(submodule, "sin"):
                submodule.sin = submodule.sin.float()

cast_rope_buffers_to_fp32(model)

decoder = decoder.eval().cuda().bfloat16()

cfg = Config.from_yaml(cfg_path)
train_cfg = cfg.train

print("Done Model Loading")

decode_fn = make_batched_decode_fn(decoder, 8)

print("Done Decoder Loading")

import wandb

wandb.init(
    project="video_models",
    entity="shahbuland",
    name="video_dit_v4"
)

from owl_wms.sampling import get_sampler_cls
import os

sampler = get_sampler_cls(train_cfg.sampler_id)(**train_cfg.sampler_kwargs)

data = torch.load("data_cache.pt")
vid = data["vid"]
mouse = data["mouse"]
btn = data["btn"]

vid = vid[:1]
mouse = mouse[:1]
btn = btn[:1]

with torch.no_grad():

    latent_vid = sampler(model, vid, mouse, btn, compile_on_decode = True)

    latent_vid = latent_vid[:, vid.size(1):]
    mouse = mouse[:, vid.size(1):]
    btn = btn[:, vid.size(1):]

    del model
    # Clear cuda cachce and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    
    video = decode_fn(latent_vid * train_cfg.vae_scale)

wandb_av_out = to_wandb_av(video, None, mouse, btn)

if len(wandb_av_out) == 3:
    video, depth_gif, flow_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif, flow_gif=flow_gif)
elif len(wandb_av_out) == 2:
    video, depth_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif)
else:
    eval_wandb_dict = dict(samples=wandb_av_out)

wandb.log(eval_wandb_dict)

print("Done")