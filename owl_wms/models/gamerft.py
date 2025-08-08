import torch
from torch import nn
import torch.nn.functional as F

from ..nn.embeddings import TimestepEmbedding, ControlEmbedding
from ..nn.attn import DiT, FinalLayer

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


class GameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.backbone == "dit"
        self.transformer = DiT(config)

        if not config.uncond:
            self.control_embed = ControlEmbedding(config.n_buttons, config.n_mouse_axes, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        assert self.config.tokens_per_frame == self.config.sample_size**2

        self.uncond = config.uncond

    def forward(self, x, t, mouse, btn, doc_id=None, has_controls=None, kv_cache=None):
        """
        x: [b,n,c,h,w]
        t: [b,n]
        mouse: [b,n,2]
        btn: [b,n,n_buttons]
        """
        b, n, c, h, w = x.shape

        t_cond = self.t_embed(t)

        if not self.uncond:
            ctrl_cond = self.control_embed(mouse, btn)  # [b,n,d]
            if has_controls is not None:
                ctrl_cond = torch.where(has_controls[:, None, None], ctrl_cond, torch.zeros_like(ctrl_cond))
            cond = t_cond + ctrl_cond  # [b,n,d]
        else:
            cond = t_cond

        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)
        x = self.transformer(x, cond, doc_id, kv_cache)
        x = self.proj_out(x, cond)

        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', h=h, w=w)
        return x


class GameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = GameRFTCore(config)

    def handle_cfg(self, has_controls=None, cfg_prob=None):
        if cfg_prob is None:
            cfg_prob = self.config.cfg_prob
        if cfg_prob <= 0.0 or has_controls is None:
            return has_controls

        # Calculate current percentage without controls
        pct_without = 1.0 - has_controls.float().mean()

        # Only apply CFG if we need more negatives
        if pct_without < cfg_prob:
            # Calculate how many more we need
            needed = cfg_prob - pct_without
            needed_frac = needed / has_controls.float().mean()

            # Only drop controls where has_controls is True
            b = has_controls.shape[0]
            mask = (torch.rand(b, device=has_controls.device) <= needed_frac) & has_controls

            # Update has_controls based on mask
            has_controls = has_controls & (~mask)

        return has_controls

    def noise(self, tensor, ts):
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts
        return lerp, z - tensor, z

    def forward(self, x, mouse=None, btn=None, doc_id=None, return_dict=False, cfg_prob=None, has_controls=None):
        B, S = x.size(0), x.size(1)

        # Apply classifier-free guidance dropout
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        if mouse is None or btn is None:
            has_controls = torch.zeros_like(has_controls)
        has_controls = self.handle_cfg(has_controls, cfg_prob)
        with torch.no_grad():
            ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_video, target_video, z_video = self.noise(x, ts[:, :, None, None, None])

        pred_video = self.core(lerpd_video, ts, mouse, btn, doc_id, has_controls)
        loss = F.mse_loss(pred_video, target_video)

        if not return_dict:
            return loss
        else:
            return {
                'diffusion_loss': loss,
                'video_loss': loss,
                'lerpd_video': lerpd_video,
                'pred_video': pred_video,
                'ts': ts,
                'z_video': z_video,
                'cfg_mask': has_controls
            }
