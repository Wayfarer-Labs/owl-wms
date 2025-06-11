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
    if trainer_id == "shortcut_2":
        from .shortcut_trainer_2 import ShortcutTrainer
        return ShortcutTrainer
    if trainer_id == "av":
        from .av_trainer import AVRFTTrainer
        return AVRFTTrainer