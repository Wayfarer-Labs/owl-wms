#### TESTING SAMPLERS - Goal:
## I suspect our sampling is broken because generation is completely nonsensical even though the causal
## student is initialized with bidirectional teacher's weights.
## 1. A) Load bidirectional model.
## 1. B) Load causal model as causal.
## 2. For each model:
## 2. A) Test Sampler as in av_trainer by saving .png's.
## 2. B) Test RF Sampler as in GameRFT by saving .png's.
import torch
from owl_wms.configs import Config
from owl_wms.utils.ddp import setup
from owl_wms.trainers.self_forcing import SelfForcingTrainer

CONFIG_PATH = "configs/self_forcing.yaml"

config = Config.from_yaml(CONFIG_PATH)
# config.model.context_length = 32
# config.model.n_frames = 35

global_rank, local_rank, world_size = setup()

trainer = SelfForcingTrainer(
    config.train, config.wandb, config.model, global_rank, local_rank, world_size
)
MODEL = trainer.bidirectional_model

sf_frames = trainer.test_self_forcing_sampler(model=MODEL)
av_frames = trainer.test_av_window_sampler(model=MODEL)


target_av_frames = av_frames[0, :6]
target_sf_frames = sf_frames[0, :6]

import os
import cv2
from PIL import Image

# TODO Save last 3 frames as png, vertically stacked
# Helper to take the last N frames and stack them vertically into one PNG
def save_vertical_stack(frames, path, ):
    imgs = []
    for f in frames:
        # unwrap any extra batch/frame dims into a [C,H,W] tensor
        # now f is [C,H,W], want as HWC:
        array = (f.permute(1,2,0).cpu().float().numpy() * 255).clip(0,255).astype("uint8")

        imgs.append(Image.fromarray(array))

    # compute stacked canvas size
    widths, heights = zip(*(im.size for im in imgs))
    canvas = Image.new("RGB", (max(widths), sum(heights)))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.size[1]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)

# Save the last 3 frames for both samplers
save_vertical_stack(target_av_frames, "output_frames/av_last3.png")
save_vertical_stack(target_sf_frames, "output_frames/sf_last3.png")
print("Saved stacked frames in output_frames/") 