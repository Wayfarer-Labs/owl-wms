def get_trainer_cls(trainer_id):
    if trainer_id == "rft":
        from .gamerft_trainer import RFTTrainer
        return RFTTrainer
    if trainer_id == "causvid":
        from .causvid import CausVidTrainer
        return CausVidTrainer
    if trainer_id == "av":
        from .av_trainer import AVRFTTrainer
        return AVRFTTrainer
    if trainer_id == "mixed_av":
        from .mixed_av_trainer import MixedAVRFTTrainer
        return MixedAVRFTTrainer
    if trainer_id == "sforce1":
        from .self_forcing_stage_1 import CausalDistillationTrainer
        return CausalDistillationTrainer
    if trainer_id == "sforce2":
        from .self_forcing_stage_2 import SelfForceTrainer
        return SelfForceTrainer
    if trainer_id == "sforce3":
        from .sf_trainer import SelfForceTrainer
        return SelfForceTrainer
    if trainer_id == "ode_distill":
        from .ode_regression import DistillODETrainer
        return DistillODETrainer