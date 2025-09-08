from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from omegaconf import OmegaConf

@dataclass
class TransformerConfig:
    model_id : str = None

    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384
    
    patch_size : int = 1
    channels : int = 128
    audio_channels : int = 64
    sample_size : int = 16

    cfg_prob : float = 0.1
    n_buttons : int = 8
    tokens_per_frame : int = 16
    audio_tokens : int = 0
    n_frames : int = 120

    causal : bool = False

@dataclass
class TrainingConfig:
    trainer_id : str = None
    data_id : str = None

    target_batch_size : int = 128
    batch_size : int = 2

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : dict = None

    loss_weights : dict = None

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : str = None

    # Distillation related
    teacher_ckpt : str = None
    teacher_cfg : str = None

    sample_interval : int = 1000
    save_interval : int = 1000

    n_samples: int = 8 # For sampling

    sampler_id : str = None
    sampler_kwargs : dict = None

    vae_id : str = None
    vae_cfg_path : str = None
    vae_ckpt_path : str = None
    vae_scale : float = 0.34
    vae_batch_size: int = 4

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None 

@dataclass
class InferenceConfig:
    # Compilation & profiling
    compile: Optional[bool] = True
    profile_kv: Optional[bool] = False
    profile_kv_every: Optional[int] = 30
    profile_kv_first: Optional[int] = 3
    sampling_steps: Optional[int] = 4

    # Quantized KV cache
    fp8_kv: Optional[bool] = True
    k_fp8: Optional[bool] = True              # Keys quantized
    kv_late_layers: Optional[int] = 12         # apply on last N layers

    # TensorRT decoder
    trt_decoder: Optional[bool] = True
    trt_decoder_force: Optional[bool] = False
    trt_decoder_slow_pct: Optional[float] = 10.0

    # MX formats for Linear layers (TorchAO)
    mxfp_enable: Optional[bool] = True        # best so far: MXFP8 enabled
    mxfp_bits: Optional[int] = 8              # 8-bit by default
    mxfp_scope: Optional[str] = "all"         # apply to all Linear layers
    mxfp_kernel: Optional[str] = "cutlass"    # prefer CUTLASS kernels

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig
    inference: Optional[InferenceConfig] = None

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)
        
        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))