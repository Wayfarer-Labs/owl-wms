# TODO Constant LR scheduler
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler_cls(scheduler_id):
    if scheduler_id == 'constant':
        return LambdaLR
    else:
        raise ValueError(f"Scheduler {scheduler_id} not found")