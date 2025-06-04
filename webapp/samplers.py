import torch
from torch import nn, Tensor
from typing import Literal, Callable
from functools import partial, cache
from multimethod import multimethod
from owl_wms.sampling.cfg import CFGSampler, InpaintCFGSampler  # TODO is there 'window' sampler?
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn

SAMPLING_STEPS = 60
SCALE = 2.17
CFG_SCALE = 1.3

MouseData   = Tensor
ButtonData  = Tensor
VideoData   = Tensor

@multimethod
def create_sampler(sampler_id: Literal['cfg'], encoder: nn.Module, decoder: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    @cache # simple singleton
    def _sampler(): return CFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=SAMPLING_STEPS,
        decode_fn=make_batched_decode_fn(decoder),
        scale=SCALE,
        cfg_scale=CFG_SCALE,
        model=encoder
    )

@multimethod
def create_sampler(sampler_id: Literal['inpaint_cfg'], encoder: nn.Module, decoder: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    @cache
    def _sampler(): return InpaintCFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=SAMPLING_STEPS,
        decode_fn=make_batched_decode_fn(decoder),
        scale=SCALE,
        cfg_scale=CFG_SCALE,
        model=encoder
    )

@multimethod
def create_sampler(sampler_id: Literal['window'], encoder: nn.Module, decoder: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    raise NotImplementedError("Window sampler not implemented")

