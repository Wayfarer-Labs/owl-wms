import torch.distributed as dist
import wandb
import torch

import einops as eo

import numpy as np
from .vis import draw_frames

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

    import os
    import tempfile
    import subprocess
    import numpy as np

    tmp_paths = []
    for i in range(len(x)):
        # Create temporary directory if it doesn't exist
        tmp_dir = os.path.join(tempfile.gettempdir(), 'wandb_videos')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Create temporary paths for video, audio and final output
        tmp_video = os.path.join(tmp_dir, f'video_{i}_temp.mp4')
        tmp_audio = os.path.join(tmp_dir, f'audio_{i}_temp.wav')
        tmp_final = os.path.join(tmp_dir, f'video_{i}.mp4')
        
        # Save frames as raw video using ffmpeg
        p = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{x[i].shape[2]}x{x[i].shape[1]}',  # width x height
            '-pix_fmt', 'rgb24',
            '-r', '60',  # fps
            '-i', '-',  # Read from pipe
            '-c:v', 'libx264',
            '-preset', 'fast',
            tmp_video
        ], stdin=subprocess.PIPE)
        
        p.stdin.write(x[i].tobytes())
        p.stdin.close()
        p.wait()

        # Save audio as wav
        audio_data = (audio[i] * 32767).astype(np.int16)  # Convert to 16-bit PCM
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 's16le',  # 16-bit signed little endian
            '-ar', '44100',  # sample rate
            '-ac', '2',  # stereo
            '-i', '-',  # Read from pipe
            tmp_audio
        ], input=audio_data.tobytes())

        # Combine video and audio
        subprocess.run([
            'ffmpeg', '-y',
            '-i', tmp_video,
            '-i', tmp_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            tmp_final
        ])

        # Cleanup temp files
        os.remove(tmp_video)
        os.remove(tmp_audio)
        
        tmp_paths.append(tmp_final)

    return [wandb.Video(path, format='mp4') for path in tmp_paths]
