# TODO Constant LR scheduler
from torch.optim.lr_scheduler import ConstantLR

def get_scheduler_cls(scheduler_id):
    if scheduler_id == 'constant':
        return ConstantLR
    else:
        raise ValueError(f"Scheduler {scheduler_id} not found")