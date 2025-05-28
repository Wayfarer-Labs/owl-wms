from .rft_trainer import RFTTrainer
from .vae_rft_trainer import VAERFTTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "rft":
        return RFTTrainer