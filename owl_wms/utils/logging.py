import torch.distributed as dist
import wandb
import torch

import einops as eo

import numpy as np
from .vis import draw_frames
from moviepy.editor import ImageSequenceClip, CompositeVideoClip
from moviepy.audio.AudioClip import AudioArrayClip

import os

class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out). 
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        
        self.data = {}
    
    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}
        
        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)
            
            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced
            
        self.data = {}
        return final

@torch.no_grad()
def to_wandb(x, batch_mouse, batch_btn, gather = False, max_samples = 8):
    # x is [b,n,c,h,w]
    x = x.clamp(-1, 1)
    x = x[:max_samples]

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    # Get labels on them
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np

    if max_samples == 8:
        x = eo.rearrange(x, '(r c) n d h w -> n d (r h) (c w)', r = 2, c = 4)

    return wandb.Video(x, format='gif',fps=60)

@torch.no_grad()
def _to_wandb_av(x, audio, batch_mouse, batch_btn, gather = False, max_samples = 8):
    # x is [b,n,c,h,w]
    # audio is [b,n,2]
    x = x.clamp(-1, 1)
    x = x[:max_samples]
    audio = audio[:max_samples]

    if dist.is_initialized() and gather:
        gathered_x = [None for _ in range(dist.get_world_size())]
        gathered_audio = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x, x)
        dist.all_gather(gathered_audio, audio)
        x = torch.cat(gathered_x, dim=0)
        audio = torch.cat(gathered_audio, dim=0)

    # Get labels on frames
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np
    
    # Convert audio to numpy float32 [-1,1]
    audio = audio.cpu().float().numpy()

    # Create grid of videos like in to_wandb
    if max_samples == 8:
        x = eo.rearrange(x, '(r c) n d h w -> n d (r h) (c w)', r = 2, c = 4)

    # Create video and audio objects
    video = wandb.Video(x, format='gif', fps=60)
    audio_samples = [wandb.Audio(audio[i], sample_rate=44100) for i in range(len(audio))]

    return video, audio_samples

@torch.no_grad()
def to_wandb_av(x, audio, batch_mouse, batch_btn, gather = False, max_samples = 8):
    # x is [b,n,c,h,w]
    # audio is [b,n,2]
    x = x.clamp(-1, 1)
    x = x[:max_samples]
    audio = audio[:max_samples].cpu().float().detach().numpy()

    if dist.is_initialized() and gather:
        gathered_x = [None for _ in range(dist.get_world_size())]
        gathered_audio = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x, x)
        dist.all_gather(gathered_audio, audio)
        x = torch.cat(gathered_x, dim=0)
        audio = torch.cat(gathered_audio, dim=0)

    # Get labels on frames
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np
    
    # Convert both to list of [n,h,w,c] and [n,2] numpy arrays
    x = [np.moveaxis(x[i], 1, -1) for i in range(len(x))]
    audio = [audio[i] for i in range(len(audio))]

    os.makedirs("temp_vids", exist_ok = True)
    paths = [f'temp_vids/temp_{i}.mp4' for i in range(len(x))]
    for i, path in enumerate(paths):
        write_video_with_audio(path, x[i], audio[i])

    return [wandb.Video(path, format='mp4') for path in paths]

def write_video_with_audio(path, vid, audio, fps=60,audio_fps=44100):
    """
    Writes videos with audio to a path at given fps and sample rate

    :param video: [n,h,w,c] [0,255] uint8 np array
    :param audio: [n,2] stereo audio as np array norm to [-1,1]
    """
    # Create video clip from image sequence
    video_clip = ImageSequenceClip(list(vid), fps=fps)
    
    # Create audio clip from array
    audio_clip = AudioArrayClip(audio, fps=audio_fps)
    
    # Combine video with audio
    video_with_audio = video_clip.set_audio(audio_clip)
    
    # Write to file
    video_with_audio.write_videofile(
        path, 
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    