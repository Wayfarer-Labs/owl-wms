import torch

from .configs import Config
from .models import get_model_cls
from .utils import versatile_load

from .utils.owl_vae_bridge import get_decoder_only

def from_pretrained(cfg_path, ckpt_path, return_decoder=False):
    cfg = Config.from_yaml(cfg_path)
    model = get_model_cls(cfg.model.model_id)(cfg.model)

    try:
        model.load_state_dict(versatile_load(ckpt_path))
    except:
        model.core.load_state_dict(versatile_load(ckpt_path))

    if not return_decoder:
        return model

    # Decoder load as well
    decoder = get_decoder_only(
        cfg.train.vae_id,
        cfg.train.vae_cfg_path,
        cfg.train.vae_ckpt_path
    )

    return model, decoder