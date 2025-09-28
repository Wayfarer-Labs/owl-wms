from diffusers import FlowMatchEulerDiscreteScheduler


def get_sd3_euler(n_steps):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(n_steps)
    ts = scheduler.sigmas
    dt = ts[:-1] - ts[1:]
    return dt


if __name__ == "__main__":
    scheduler = get_sd3_euler(10)
    print(scheduler)
