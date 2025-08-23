from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
import torch


def get_sd3_euler(n_steps):
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    scheduler.set_timesteps(n_steps)
    t = scheduler.timesteps.float() / float(scheduler.config.num_train_timesteps - 1)
    return t


if __name__ == "__main__":
    scheduler = get_sd3_euler(10)
    print(scheduler)
