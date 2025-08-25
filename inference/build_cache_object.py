import joblib

import torch
from owl_wms.configs import Config
from owl_wms import from_pretrained
from owl_wms.nn.rope import RoPE
from owl_wms.nn.kv_cache import KVCache

def cast_rope_buffers_to_fp32(module):
    """Cast RoPE buffers to fp32 for numerical stability"""
    for submodule in module.modules():
        if isinstance(submodule, RoPE):
            if hasattr(submodule, "cos"):
                submodule.cos = submodule.cos.float()
            if hasattr(submodule, "sin"):
                submodule.sin = submodule.sin.float()

@torch.no_grad()
def build_cache():
    # Configuration
    cfg_path = "configs/dit_v4_dmd.yml"
    ckpt_path = "vid_dit_v4_dmd_7k.pt"
    
    # Load model (no decoder needed)
    print("Loading model...")
    model = from_pretrained(cfg_path, ckpt_path, return_decoder=False)
    model = model.core.eval().cuda().bfloat16()
    
    # Cast RoPE buffers to fp32
    cast_rope_buffers_to_fp32(model)
    
    print("Model loaded successfully!")
    
    # Load data cache
    data = torch.load("data_cache.pt")
    vid = data["vid"]
    mouse = data["mouse"]
    btn = data["btn"]
    
    batch_size = vid.size(0)
    init_len = vid.size(1)
    
    print(f"Cache input shapes:")
    print(f"  vid: {vid.shape}")
    print(f"  mouse: {mouse.shape}")
    print(f"  btn: {btn.shape}")
    
    # Initialize KV cache
    kv_cache = KVCache(model.config)
    kv_cache.reset(batch_size)
    
    # Build cache with context frames
    noise_prev = 0.2
    vid_noisy = vid * (1. - noise_prev) + torch.randn_like(vid) * noise_prev
    t_noisy = vid.new_full((batch_size, init_len), noise_prev)

    init_len = 60
    vid_noisy = vid_noisy[:,:init_len]
    t_noisy = t_noisy[:vid_noisy.size(0),:init_len]
    mouse = mouse[:vid_noisy.size(0),:init_len]
    btn = btn[:vid_noisy.size(0),:init_len]
    
    
    print("Building KV cache...")
    kv_cache.enable_cache_updates()
    _ = model(
        vid_noisy,
        t_noisy,
        mouse,
        btn,
        kv_cache=kv_cache
    )
    kv_cache.disable_cache_updates()
    
    # Save the cache object
    cache_object = kv_cache
    
    print("Saving cache object...")
    joblib.dump(cache_object, 'kv_cache_object.pkl')
    print("Cache object saved to 'kv_cache_object.pkl'")

if __name__ == "__main__":
    build_cache()

