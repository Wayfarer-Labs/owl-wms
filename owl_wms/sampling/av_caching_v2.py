import torch
from tqdm import tqdm
import gc

from ..nn.kv_cache import KVCache, InferenceKVCache

from .schedulers import get_sd3_euler

def n_tokens(vid):
    return vid.size(1) * vid.size(3) * vid.size(4)

def get_deltas(custom_schedule):
    if custom_schedule[-1] != 0.0:
        custom_schedule.append(0.0)

    deltas = []
    crnt = custom_schedule[0]
    for nxt in custom_schedule[1:]:
        deltas.append(abs(nxt - crnt))
        crnt = nxt

    return deltas

class AVCachingSamplerV2:
    """
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param cfg_scale: Must be 1.0
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.3, num_frames: int = 60, noise_prev: float = 0.2, max_window = None, custom_schedule = None) -> None:
        self.cfg_scale = cfg_scale
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.max_window = max_window
        self.custom_schedule = custom_schedule

    @staticmethod
    def zlerp(x, alpha):
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def __call__(self, model, x, mouse, btn, compile_on_decode = False):
        batch_size, init_len = x.size(0), x.size(1)

        if self.custom_schedule is None:
            dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
        else:
            dt = get_deltas(self.custom_schedule)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        # At the start this is the whole video
        latents = [x.clone()]
        prev_x = x
        prev_mouse, prev_btn = mouse[:, :init_len], btn[:, :init_len]

        # ==== STEP 1: Cache context ====

        prev_x_noisy = self.zlerp(prev_x, self.noise_prev)
        prev_t = prev_x.new_full((batch_size, prev_x.size(1)), self.noise_prev)

        kv_cache.enable_cache_updates()
        _ = model(
            prev_x_noisy,
            prev_t,
            prev_mouse,
            prev_btn,
            kv_cache=kv_cache
        )
        kv_cache.disable_cache_updates()
        kv_cache = InferenceKVCache(kv_cache)

        def new_xt():
            return torch.randn_like(prev_x[:,:1]), prev_t.new_ones(batch_size, 1)

        # START FRAME LOOP
        num_frames = min(self.num_frames, mouse.size(1) - init_len)

        if compile_on_decode:
            model = torch.compile(model)

        model.transformer.enable_decoding()
        
        for idx in tqdm(range(num_frames), desc = "Sampling Frames..."):
            curr_x, curr_t = new_xt()

            start = init_len + idx
            curr_mouse, curr_btn = mouse[:,start:start+1], btn[:,start:start+1]
            null_mouse = torch.zeros_like(curr_mouse)
            null_btn = torch.zeros_like(curr_btn)

            # ==== STEP 2: Denoise the new frame ====
            for t_idx in range(self.n_steps):
                # now we have [b,1,c,h,w] vid, [b,1,2] mouse, [b,1,nb] btn
                pred_v = model(
                    curr_x,
                    curr_t,
                    curr_mouse,
                    curr_btn,
                    kv_cache=kv_cache
                ).clone()
                if self.cfg_scale != 1.0:
                    pred_v_uncond = model(
                        curr_x,
                        curr_t,
                        null_mouse,
                        null_btn,
                        kv_cache=kv_cache
                    ).clone()
                    pred_v = pred_v_uncond + self.cfg_scale * (pred_v - pred_v_uncond)

                curr_x = curr_x - dt[t_idx] * pred_v   
                curr_t = curr_t - dt[t_idx]


            
            # ==== STEP 3: New frame generated, append and cache ====
            latents.append(curr_x.clone()) # Append
            curr_x_noisy = self.zlerp(curr_x, self.noise_prev)
            curr_t_noisy = torch.ones_like(curr_t) * self.noise_prev

            kv_cache.enable_cache_updates()
            _ = model(
                curr_x_noisy,
                curr_t_noisy,
                curr_mouse,
                curr_btn,
                kv_cache=kv_cache
            )
            kv_cache.disable_cache_updates()

            gc.collect()
            torch.cuda.empty_cache()

        model.transformer.disable_decoding()

        return torch.cat(latents, dim = 1)

                


            
            


