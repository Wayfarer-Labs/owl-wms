import torch

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from profiling.timing import time_fn

import torch

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

wm_cfg = "configs/av.yml"
vae_cfg = "configs/owl_vaes/cod_128x.yml"
audio_vae_cfg = "configs/owl_vaes/cod_audio.yml"

world_model = load_from_config(wm_cfg).core.bfloat16().cuda()
img_dec = get_decoder_only(None, vae_cfg)
audio_dec = get_decoder_only(None, audio_vae_cfg)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"World model parameters: {count_parameters(world_model):,}")

dummy_x = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
dummy_audio = torch.randn(1, 1, 64).bfloat16().cuda()
ts = torch.ones(1,1).bfloat16().cuda()
mouse = torch.randn(1,1,2).bfloat16().cuda()
btn = torch.randint(0, 1, (1,1,11)).bfloat16().cuda()

dummy = (dummy_x, dummy_audio, ts, mouse, btn)

world_model = torch.compile(world_model, mode='max-autotune', dynamic = False, fullgraph=True)
img_dec = torch.compile(img_dec,mode='max-autotune', dynamic = False, fullgraph=True)
audio_dec = torch.compile(audio_dec,mode='max-autotune', dynamic = False, fullgraph=True)

res = time_fn(world_model, dummy)
print(f"Mean: {res['mean']:.2f}ms")
print(f"Min: {res['min']:.2f}ms")
print(f"Max: {res['max']:.2f}ms")
print(f"Avg FPS: {1000./res['mean']:.2f}FPS")

dummy_audio_2 = torch.randn(1, 64, 120).bfloat16().cuda()

res = time_fn(img_dec, dummy_x[0])
print(f"Mean: {res['mean']:.2f}ms")
print(f"Min: {res['min']:.2f}ms")
print(f"Max: {res['max']:.2f}ms")
print(f"Avg FPS: {1000./res['mean']:.2f}FPS")

res = time_fn(audio_dec, dummy_audio_2)
print(f"Mean: {res['mean']:.2f}ms")
print(f"Min: {res['min']:.2f}ms")
print(f"Max: {res['max']:.2f}ms")
print(f"Avg FPS: {1000./res['mean']:.2f}FPS")

"""
1B Model notes (for 1 step with KV cache)
Mean: 85.35ms | 10 FPS
Min: 78.28ms
Max: 93.37ms

"""
