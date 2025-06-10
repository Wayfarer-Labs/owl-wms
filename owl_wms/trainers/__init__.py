from .gamerft_trainer import RFTTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "rft":
        return RFTTrainer
    if trainer_id == "causvid":
        from .causvid import CausVidTrainer
        return CausVidTrainer
    if trainer_id == "shortcut":
        from .shortcut_trainer import ShortcutTrainer
        return ShortcutTrainer