import torch
from torch import nn, Tensor
from typing import Literal, Callable
from functools import partial, cache
from multimethod import multimethod
from owl_wms.sampling.cfg import CFGSampler, InpaintCFGSampler  # TODO is there 'window' sampler?
from webapp.models import load_model

SAMPLING_STEPS = 64
BATCH_SHAPE = torch.empty(1, 128, 16, 256, 256) # TODO fix this shit
DECODE_FN = None
SCALE = 1
CFG_SCALE = 1.3

MouseData   = Tensor
ButtonData  = Tensor
VideoData   = Tensor

@multimethod
def create_sampler(sampler_id: Literal['cfg'], model: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    @cache # simple singleton
    def _sampler(): return CFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=SAMPLING_STEPS,
        decode_fn=DECODE_FN,
        scale=SCALE,
        cfg_scale=CFG_SCALE,
        model=model
    )

@multimethod
def create_sampler(sampler_id: Literal['inpaint_cfg'], model: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    @cache
    def _sampler(): return InpaintCFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=SAMPLING_STEPS,
        decode_fn=DECODE_FN,
        scale=SCALE,
        cfg_scale=CFG_SCALE,
        model=model
    )

@multimethod
def create_sampler(sampler_id: Literal['window'], model: nn.Module) -> Callable[[MouseData, ButtonData], VideoData]:
    raise NotImplementedError("Window sampler not implemented")


model = load_model()
cfg_sampler = create_sampler('cfg', model)


