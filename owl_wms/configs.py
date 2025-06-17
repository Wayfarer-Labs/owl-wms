import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    model_id : str = None
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
    # -- self forcing only:
    frame_gradient_cutoff : int = 12
    student_ckpt : str = None
    teacher_ckpt : str = None
    # -- 

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

    audio_vae_id : str = None
    audio_vae_cfg_path : str = None
    audio_vae_ckpt_path : str = None
    audio_vae_scale : float = 0.17

    # -- self forcing stuff:
    frame_gradient_cutoff : int = 20 # number of frames from the end to start gradient computation for
    t_schedule : list[int] = field(default_factory=lambda: [1000, 750, 500, 250]) # timesteps to sample for DMD loss
    latent_shape : tuple[int, int, int] = (128, 4, 4)


@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None 

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)
        
        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))

