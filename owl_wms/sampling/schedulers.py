from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler


def get_sd3_euler(n_steps, num_train_timesteps=1000):
    # TODO: keep old fn for backwards compatability, rename to def get_dsigmas
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
        num_train_timesteps=num_train_timesteps,  # granularity
    )
    scheduler.set_timesteps(num_inference_steps=n_steps)
    sigmas = scheduler.sigmas.float()
    return sigmas[:-1] - sigmas[1:]


if __name__ == "__main__":
    scheduler = get_sd3_euler(10)
    print(scheduler)
