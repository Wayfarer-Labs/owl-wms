from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np

def get_sd3_euler(n_steps):
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=3,
        num_train_timesteps=n_steps
    )
    ts = scheduler.timesteps / n_steps
    ts = torch.cat([ts, torch.zeros(1, dtype=ts.dtype, device=ts.device)])
    dt = ts[:-1] - ts[1:]
    return dt

if __name__ == "__main__":
    scheduler = get_sd3_euler(10)
    print(scheduler)