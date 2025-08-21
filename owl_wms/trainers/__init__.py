def get_trainer_cls(trainer_id):
    if trainer_id == "craft":
        from .rft_pair_distill_trainer import RFTPairDistillTrainer
        return RFTPairDistillTrainer
    if trainer_id == "rft_pair_distill":
        from .rft_pair_distill_trainer import RFTPairDistillTrainer
        return RFTPairDistillTrainer
    if trainer_id == "causvid":
        """
        CausvidTrainer does DMD to create few-step causal student
        """
        from .causvid_v2 import CausVidTrainer
        return CausVidTrainer
    if trainer_id == "av":
        """
        Most basic trainer. Does audio + video training.
        """
        from .av_trainer import AVRFTTrainer
        return AVRFTTrainer
    if trainer_id == "rft":
        """
        Most basic trainer. Does audio + video training.
        """
        from .rft_trainer import RFTTrainer
        return RFTTrainer
    if trainer_id == "mixed_av":
        """
        Allows for datasets that are a mix of unlabelled (wrt controls) and labelled
        """
        from .mixed_av_trainer import MixedAVRFTTrainer
        return MixedAVRFTTrainer
    if trainer_id == "sforce":
        """
        Self force trainer does clean context + DMD to enable few step with KV caching (broken rn)
        """
        from .sf_trainer_v2 import SelfForceTrainer
        return SelfForceTrainer
    if trainer_id == "ode_distill":
        """
        ODE regression matches student trajectories to teacher trajectories
        """
        from .ode_regression import DistillODETrainer
        return DistillODETrainer
