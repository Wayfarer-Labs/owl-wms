import torch
from fvcore.nn import FlopCountAnalysis
import time
from typing import Optional, Dict, Any

class MFUProfiler:
    """
    Model FLOPs Utilization (MFU) profiler for training loops.
    
    Usage:
        profiler = MFUProfiler(model, example_inputs, world_size=world_size, rank=rank)
        
        # In training loop:
        profiler.start_step()
        # ... training step ...
        metrics = profiler.end_step(batch_size)
        wandb_dict.update(metrics)
    """
    
    def __init__(self, model, example_inputs, world_size: int = 1, rank: int = 0, 
                 precision: str = "bf16", enabled: bool = True):
        self.world_size = world_size
        self.rank = rank
        self.enabled = enabled and (rank == 0)  # Only profile on rank 0
        
        if not self.enabled:
            return
            
        # Compute FLOPs once
        self.flops_fwd = self._get_model_flops(model, example_inputs)
        self.flops_per_sample = self.flops_fwd * 2  # forward + backward
        self.peak_tflops = self._get_device_peak_tflops(precision == "bf16")
        
        print(f"MFU Profiler initialized:")
        print(f"  • FLOPs/sample: {self.flops_per_sample/1e9:.1f} GFLOPs")
        print(f"  • Device peak: {self.peak_tflops:.1f} TFLOPs")
        
        self.step_start_time = None
        
    def start_step(self):
        """Call at the start of each training step"""
        if not self.enabled:
            return
        torch.cuda.synchronize()  # Ensure accurate timing
        self.step_start_time = time.time()
        
    def end_step(self, global_batch_size: int) -> Dict[str, float]:
        """
        Call at the end of each training step.
        Returns dict of metrics to log.
        """
        if not self.enabled:
            return {}
            
        torch.cuda.synchronize()
        step_time = time.time() - self.step_start_time
        
        samples_per_sec = global_batch_size / step_time
        achieved_tflops = (samples_per_sec * self.flops_per_sample) / 1e12
        mfu = achieved_tflops / self.peak_tflops
        
        return {
            'samples_per_sec': samples_per_sec,
            'step_time': step_time,
            'tflops': achieved_tflops,
            'mfu': mfu
        }
    
    def _get_model_flops(self, model, example_inputs) -> int:
        """Compute FLOPs for one forward pass"""
        flops = FlopCountAnalysis(model, example_inputs).total()
        return int(flops)
    
    def _get_device_peak_tflops(self, bf16: bool = True) -> float:
        """Rough theoretical peak TFLOPs"""
        #fp_ops_per_sm = 256 if bf16 else 128
        #prop = torch.cuda.get_device_properties(0)
        #clock_rate = torch.cuda.clock_rate()
        #clock_rate = 1785
        #peak = 2 * prop.multi_processor_count * fp_ops_per_sm * clock_rate * 1e3

        return 1979
        #return peak / 1e12


if __name__ == "__main__":
    cfg_path = "configs/av_v5_8x8_mixed.yml"

    from ..configs import Config
    from ..models import get_model_cls

    cfg = Config.from_yaml(cfg_path)
    model = get_model_cls(cfg.model.model_id)(cfg.model)
    model = model.cuda().bfloat16()

    # Create dummy batch with expected input shapes
    n = cfg.model.n_frames
    b = 1
    c = cfg.model.channels
    h = w = cfg.model.sample_size
    audio_c = cfg.model.audio_channels
    n_buttons = cfg.model.n_buttons
    n_mouse_axes = cfg.model.n_mouse_axes
    
    # Create dummy inputs matching the expected shapes
    profiler_sample = (
        torch.randn(b, n, c, h, w, device='cuda', dtype=torch.bfloat16),  # video
        torch.randn(b, n, audio_c, device='cuda', dtype=torch.bfloat16),  # audio
        torch.randn(b, n, n_mouse_axes, device='cuda', dtype=torch.bfloat16),  # mouse
        torch.randn(b, n, n_buttons, device='cuda', dtype=torch.bfloat16),  # buttons
    )
    model = torch.compile(model, mode='max-autotune', dynamic=False, fullgraph=True)
    profiler = MFUProfiler(model, profiler_sample, world_size=1, rank=0, precision="bf16", enabled=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    res = []
    for _ in range(10):
        with torch.no_grad():
            profiler.start_step()
            loss = model(*profiler_sample)
            #opt.zero_grad()
        # loss.backward()
            #opt.step()
            res.append(profiler.end_step(cfg.train.batch_size))
        
    # Res is a dict of metrics, so we want to get min,max,mean for each key
    for key in res[0].keys():
        print(f"{key} (min): {min([r[key] for r in res])}")
        print(f"{key} (max): {max([r[key] for r in res])}")
        print(f"{key} (mean): {sum([r[key] for r in res]) / len(res)}")


    