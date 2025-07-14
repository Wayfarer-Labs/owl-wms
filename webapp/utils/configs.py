from __future__ import annotations

import io
import os
import tarfile
import yaml
import torch
import random
from dataclasses import dataclass

from owl_wms.configs import Config as RunConfig

@dataclass
class WebappConfig:
    run_config : RunConfig
    stream_config : StreamingConfig
    sampling_config : SamplingConfig
    run_config_path : os.PathLike
    device : str = 'cuda'

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> WebappConfig:
        #  
        with open(path, 'r') as wcp:
            config                      = yaml.safe_load(wcp)
            config['run_config']        = RunConfig.from_yaml(config['run_config_path'])
            config['sampling_config']   = SamplingConfig(**config['sampling_config'])
            config['stream_config']     = StreamingConfig(**config['stream_config'])

        return cls(**config)


@dataclass    
class SamplingConfig:
    sampling_steps : int = 20
    vae_scale: float = 1.0
    cfg_scale : float = 1.3
    window_length : int = 60
    num_frames : int = 60
    noise_prev : float = 0.2


@dataclass
class StreamingConfig:
    model_checkpoint_path : os.PathLike
    fps: int = 20
    frames_per_batch: int = 1
    device: str = 'cuda'
    n_buttons: int = 11
    n_mouse_axes: int = 2
    window_size: int = 30
    mouse_range: tuple[float, float] = (-1.0, 1.0)
    history_tar_path: os.PathLike = None
    tar_base_idx: int = 0
    video_latent_history_path: os.PathLike = None
    audio_latent_history_path: os.PathLike = None
    mouse_history_path: os.PathLike = None
    button_history_path: os.PathLike = None
    with_audio: bool = False


    @property
    def frame_interval(self) -> float: return 1.0 / self.fps

    @property
    def batch_duration(self) -> float: return self.frames_per_batch / self.fps

    @property
    def video_latent_history(self) -> torch.Tensor:
        if self.video_latent_history_path is None:  raise ValueError("video_latent_history_path is not set")
        return torch.load(self.video_latent_history_path)
    
    @property
    def audio_latent_history(self) -> torch.Tensor:
        if self.audio_latent_history_path is None:  raise ValueError("audio_latent_history_path is not set")
        return torch.load(self.audio_latent_history_path)
    
    @property
    def mouse_history(self) -> torch.Tensor:
        if self.mouse_history_path is None:         raise ValueError("mouse_history_path is not set")
        return torch.load(self.mouse_history_path)
    
    @property
    def button_history(self) -> torch.Tensor:
        if self.button_history_path is None:         raise ValueError("button_history_path is not set")
        return torch.load(self.button_history_path)
    
if __name__ == '__main__':
    window_size = 60
    with tarfile.open('/home/louis/owl-wms/webapp/static/0000.tar') as tar:
        members = tar.getmembers()
        for member in members:
            if member.name.endswith('.latent.pt'):
                f = tar.extractfile(member.name)
                tensor_data = f.read()
                tensor = torch.load(io.BytesIO(tensor_data))
                num_base_indices = tensor.shape[0] // window_size
                print(tensor.shape)
                break