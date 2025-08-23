from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
import torch


def get_sd3_euler(n_steps):
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(num_inference_steps=n_steps)

    # t = scheduler.timesteps.float() / float(scheduler.config.num_train_timesteps - 1)
    # dt = t - torch.cat([t[1:], t.new_zeros(1)])

    sigmas = scheduler.sigmas.to(torch.float32)  # populated by set_timesteps
    dt_sigma = sigmas - torch.cat([sigmas[1:], sigmas.new_zeros(1)])  # Δσ_i = σ_i - σ_{i+1}

    return dt_sigma


if __name__ == "__main__":
    scheduler = get_sd3_euler(10)
    print(scheduler)
