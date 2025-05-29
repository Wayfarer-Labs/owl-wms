from .gamerft_trainer import RFTTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "rft":
        return RFTTrainer