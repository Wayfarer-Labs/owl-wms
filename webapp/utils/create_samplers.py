from torch import Tensor
from typing import Literal, Callable
from functools import partial, cache
from multimethod import multimethod
from owl_wms.sampling.cfg import CFGSampler
from owl_wms.sampling.simple import SimpleSampler, InpaintSimpleSampler
from owl_wms.sampling.window import WindowCFGSampler
from owl_wms.sampling.av_window import Inference_AV_WindowSampler
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn, make_batched_audio_decode_fn

SAMPLING_STEPS = 60
SCALE = 2.17
CFG_SCALE = 1.3

MouseData   = Tensor
ButtonData  = Tensor
VideoData   = Tensor
LatentData  = Tensor

@multimethod
def create_sampler(sampler_id: Literal['cfg'], encoder, decoder,
                   batch_size: int = 8,
                   n_steps: int = 20,
                   cfg_scale: float = 1.3,
                   vae_scale: float = 1.0,
                   **kwargs) -> Callable:
    """Create CFG sampler with its specific parameters."""
    
    @cache
    def _sampler(): 
        return CFGSampler(n_steps=n_steps, cfg_scale=cfg_scale)

    return partial(
        _sampler().__call__,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=vae_scale,
        model=encoder
    )


@multimethod  
def create_sampler(sampler_id: Literal['simple'], encoder, decoder,
                   batch_size: int = 8,
                   n_steps: int = 64,
                   vae_scale: float = 1.0,
                   **kwargs) -> Callable:
    """Create Simple sampler with its specific parameters."""
    
    @cache
    def _sampler():
        return SimpleSampler(n_steps=n_steps)
    
    return partial(
        _sampler().__call__,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=vae_scale,
        model=encoder
    )


@multimethod
def create_sampler(sampler_id: Literal['inpaint_simple'], encoder, decoder,
                   batch_size: int = 8,
                   n_steps: int = 64,
                   vae_scale: float = 1.0,
                   **kwargs) -> Callable:
    """Create Inpaint Simple sampler with its specific parameters."""
    
    @cache
    def _sampler():
        return InpaintSimpleSampler(n_steps=n_steps)
    
    return partial(
        _sampler().__call__,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=vae_scale,
        model=encoder
    )


@multimethod
def create_sampler(sampler_id: Literal['window'], encoder, decoder,
                   batch_size: int = 8,
                   n_steps: int = 20,
                   cfg_scale: float = 1.3,
                   window_length: int = 60,
                   num_frames: int = 60,
                   noise_prev: float = 0.2,
                   only_return_generated: bool = False,
                   vae_scale: float = 1.0,
                   **kwargs) -> Callable:
    """Create Window CFG sampler with its specific parameters."""
    
    @cache
    def _sampler():
        return WindowCFGSampler(
            n_steps=n_steps,
            cfg_scale=cfg_scale,
            window_length=window_length,
            num_frames=num_frames,
            noise_prev=noise_prev,
            only_return_generated=only_return_generated
        )
    
    return partial(
        _sampler().__call__,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=vae_scale,
        model=encoder
    )


@multimethod
def create_sampler(sampler_id: Literal['av_window'],
                   encoder, decoder, audio_decoder,
                   batch_size: int = 8,
                   n_steps: int = 20,
                   cfg_scale: float = 1.3,
                   window_length: int = 60,
                   num_frames: int = 60,
                   noise_prev: float = 0.2,
                   only_return_generated: bool = True,
                   vae_scale: float = 1.0,
                   **kwargs) -> Callable:

    @cache
    def _sampler():
        return Inference_AV_WindowSampler(
            n_steps=n_steps,
            cfg_scale=cfg_scale,
            window_length=window_length,
            num_frames=num_frames,
            noise_prev=noise_prev,
            only_return_generated=only_return_generated
        )

    # TODO `dummy_batch` is history, `audio` is audio history, mouse is mouse history, btn is button history
    return partial(
        _sampler().__call__,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        audio_decode_fn=make_batched_audio_decode_fn(audio_decoder, batch_size=batch_size),
        scale=vae_scale,
        model=encoder
    )


if __name__ == "__main__":
    # Each sampler type can be created with its specific parameters
    import webapp.utils.models
    encoder, decoder, model_config = webapp.utils.models.load_models()

    # CFG sampler
    cfg_sampler = create_sampler(
        'cfg', encoder, decoder,
        n_steps=25, cfg_scale=1.5
    )
    
    # Simple sampler  
    simple_sampler = create_sampler(
        'simple', encoder, decoder,
        n_steps=50
    )
    
    # Window sampler with all its specific params
    window_sampler = create_sampler(
        'window', encoder, decoder,
        n_steps=30, cfg_scale=1.4,
        window_length=80, num_frames=120,
        noise_prev=0.3, only_return_generated=True
    )

    # AV Window sampler with all its specific params
    av_window_sampler = create_sampler(
        'av_window', encoder, decoder,
        n_steps=10, cfg_scale=1.3,
        window_length=30, num_frames=60,
        noise_prev=0.2, only_return_generated=False
    )