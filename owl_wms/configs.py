import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TransformerConfig:
    model_id : Optional[str] = None
    channels : int = 128
    sample_size : int = 16
    patch_size : int = 1

    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384
    
    audio_channels : int = 64

    cfg_prob : float = 0.1
    n_buttons : int = 8
    tokens_per_frame : int = 16
    audio_tokens : int = 0
    n_frames : int = 120

    causal : bool = False
    # -- self forcing stuff:
    context_length : int = 48 # number of frames of context in kv cache


@dataclass
class TrainingConfig:
    trainer_id : Optional[str] = None
    data_id : Optional[str] = None
    ode_init_steps : int = 1000
    n_steps : int = 4
    target_batch_size : int = 128
    batch_size : int = 2
    max_grad_norm : float = 5.0
    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : Optional[dict] = None

    loss_weights : Optional[dict] = None

    scheduler : Optional[str] = None
    scheduler_kwargs : Optional[dict] = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : Optional[str] = None

    # -- self forcing only:
    student_ckpt : Optional[str] = None
    critic_ckpt: Optional[str] = None
    # -- 

    # Distillation related
    teacher_ckpt : Optional[str] = None
    teacher_cfg : Optional[str] = None

    log_interval : int = 100
    sample_interval : int = 1000
    save_interval : int = 1000

    n_samples: int = 8 # For sampling

    sampler_id : Optional[str] = None
    sampler_kwargs : Optional[dict] = None

    vae_id : Optional[str] = None
    vae_cfg_path : Optional[str] = None
    vae_ckpt_path : Optional[str] = None
    vae_scale : float = 0.34
    vae_batch_size: int = 4

    audio_vae_id : Optional[str] = None
    audio_vae_cfg_path : Optional[str] = None
    audio_vae_ckpt_path : Optional[str] = None
    audio_vae_scale : float = 0.17

    # -- self forcing stuff:
    frame_gradient_cutoff : int = 20 # number of frames from the end to start gradient computation for
    t_schedule : list[int] = field(default_factory=lambda: [1000, 750, 500, 250]) # timesteps to sample for DMD loss
    latent_shape : tuple[int, int, int] = (128, 4, 4)


@dataclass
class WANDBConfig:
    name : Optional[str] = None
    project : Optional[str] = None
    run_name : Optional[str] = None 

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)
        
        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))

