from .simple import SimpleSampler
from .cfg import CFGSampler
from .window import WindowCFGSampler

def get_sampler_cls(sampler_id):
    if sampler_id == "simple":
        return SimpleSampler
    elif sampler_id == "cfg":
        return CFGSampler
    elif sampler_id == "window":
        return WindowCFGSampler
    elif sampler_id == "av_window":
        from .av_window import AVWindowSampler
        return AVWindowSampler
    elif sampler_id == "av_caching":
        from .av_caching import AVCachingSampler
        return AVCachingSampler
    elif sampler_id == "av_causal":
        from .av_window import CausalAVWindowSampler
        return CausalAVWindowSampler
    elif sampler_id == "av_causal_no_cfg":
        from .av_window import CausalAVWindowSamplerNoCFG
        return CausalAVWindowSamplerNoCFG
    elif sampler_id == "av_caching_one_step":
        from .av_caching import AVCachingOneStepSampler
        return AVCachingOneStepSampler