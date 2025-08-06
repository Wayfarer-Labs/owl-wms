def get_sampler_cls(sampler_id):
    if sampler_id == "av_window":
        """
        Most basic Audio+Video sampler with CFG
        """
        from .av_window import AVWindowSampler
        return AVWindowSampler
    if sampler_id == "video_window":
        """
        Most basic Audio+Video sampler with CFG
        """
        from .video_window import VideoWindowSampler
        return VideoWindowSampler
    elif sampler_id == "av_caching":
        """
        Audio+Video sampler with KV caching.
        """
        from .av_caching import AVCachingSampler
        return AVCachingSampler
    elif sampler_id == "av_causal":
        """
        Audio+Video sampler with causal sampling, caches noisy history on first diffusion step.
        """
        from .av_window import CausalAVWindowSampler
        return CausalAVWindowSampler
    elif sampler_id == "av_causal_no_cfg":
        """
        Identical to av causal but skips cfg (ideal for distilled models)
        """
        from .av_window import CausalAVWindowSamplerNoCFG
        return CausalAVWindowSamplerNoCFG
    elif sampler_id == "av_caching_one_step":
        """
        Identical to av_caching but with a hard assumption for one step to simplify
        """
        from .av_caching import AVCachingOneStepSampler
        return AVCachingOneStepSampler
