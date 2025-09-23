from typing import Optional, List
from torch import Tensor

import einops as eo
from tensordict import TensorDict

import torch
from torch import nn

from .. import nn as owl_nn

from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy


class PromptEncoder(nn.Module):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    """Callable for text -> UMT5 embedding"""
    def __init__(self, model_id="google/umt5-xl", dtype=torch.bfloat16):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.encoder = UMT5EncoderModel.from_pretrained(model_id, torch_dtype=dtype).eval()

    @torch.compile
    def encode(self, inputs):
        return self.encoder(**inputs).last_hidden_state

    @torch.inference_mode()
    def forward(self, texts: List[str]):
        texts = [ftfy.fix_text(t) for t in texts]
        inputs = self.tok(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.encoder.device)
        emb = self.encode(inputs)
        pad_mask = ~inputs["attention_mask"].bool()  # True = PAD (ignore)
        return TensorDict({"emb": emb, "pad_mask": pad_mask}, batch_size=[emb.size(0)])


class ControllerInputEmbedding(nn.Module):
    def __init__(self, n_inputs, dim_out, dim=512):
        super().__init__()
        self.mlp = owl_nn.MLPCustom(n_inputs, dim * 4, dim_out)

    def forward(self, controller_input: Tensor):
        return self.mlp(controller_input)


class WorldDiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = owl_nn.Attn(config, layer_idx)
        self.cross_attn = owl_nn.CrossAttention(config)
        self.mlp = owl_nn.MLP(config)

        dim = config.d_model
        self.adaln = nn.ModuleList([owl_nn.AdaLN(dim) for _ in range(3)])
        self.gate = nn.ModuleList([owl_nn.Gate(dim) for _ in range(3)])

    def forward(self, x, pos_ids, cond, prompt_emb, ctrl_emb, block_mask, kv_cache=None):
        """
        0) Causal Frame Attention
        1) Frame->Text Cross Attention
        2) MLP
        """
        if self.config.noise_conditioning == "wan":
            cond = cond + self.conditioning_bias

        residual = x
        x = self.adaln[0](x, cond)
        x = self.attn(x, pos_ids, block_mask, kv_cache)
        x = self.gate[0](x, cond)
        x = x + residual

        """
        if prompt_emb is not None:
            residual = x
            x = self.adaln1(x, cond)
            x = self.cross_attn(x, context=prompt_emb["emb"], context_pad_mask=prompt_emb["pad_mask"])
            x = self.gate1(x, cond) + residual

        if ctrl_emb is not None:
            residual = x
            x = self.adaln1(x, cond)
            x = self.cross_attn_same_frame(x, context=ctrl_emb)
            x = self.gate1(x, cond) + residual
        """

        residual = x
        x = self.adaln[2](x, cond)
        x = self.mlp(x)
        x = self.gate[2](x, cond)
        x = x + residual

        return x


class WorldDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_masker = owl_nn.AttnMaskScheduler(config)
        self.blocks = nn.ModuleList([WorldDiTBlock(config, idx) for idx in range(config.n_layers)])

        if self.config.noise_conditioning in ("dit_air", "wan"):
            ref = self.blocks[0]
            for blk in self.blocks[1:]:
                blk.adaln, blk.gate = ref.adaln, ref.gate

        if self.config.noise_conditioning == "wan":
            self.conditioning_bias = nn.ModuleList([
                nn.Parameter(torch.zeros(config.d_model)) for _ in range(config.n_layers)
            ])
        else:
            self.conditioning_bias = [None] * config.n_layers

    def forward(self, x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id=None, kv_cache=None):
        ####
        # TODO: REMOVE, just an experiment
        if ctrl_emb is not None:
            cond = cond + ctrl_emb
        ####

        t_pos = pos_ids["t_pos"]
        if kv_cache is not None:
            t_pos = kv_cache.upsert_t_pos(t_pos)

        # generate block masks for each layer
        block_masks = self.attn_masker(
            seq_len=x.size(1),
            doc_id=doc_id,
            kv_cache=kv_cache,
            t_pos=t_pos,
            device=x.device
        )
        for block, block_mask, cond_bias in zip(self.blocks, block_masks, self.conditioning_bias):
            cond_layer = cond + cond_bias if cond_bias is not None else cond
            x = block(x, pos_ids, cond_layer, prompt_emb, ctrl_emb, block_mask, kv_cache)
        return x


class WorldModel(nn.Module):
    """
    WORLD: Wayfarer Operator-driven Rectified-flow Long-context Diffuser

    Denoise a frame given
    - All previous frames
    - The prompt embedding
    - The controller input embedding
    - The current noise level
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        assert config.tokens_per_frame == config.height * config.width

        self.denoise_step_emb = owl_nn.TimestepEmbedding(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config.n_controller_inputs, config.d_model)

        self.transformer = WorldDiT(config)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = owl_nn.FinalLayer(config.d_model, config.channels)

    def flat_forward(
        self,
        x: Tensor,
        pos_ids: TensorDict,
        sigma: Tensor,
        prompt_emb: Optional[TensorDict] = None,
        controller_inputs: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None
    ):
        assert doc_id is None or kv_cache is None, "Cannot use sequence packing with kv caching"
        assert x.ndim == 3, "Requires x to be [B, S, C]"

        # embed
        cond = self.denoise_step_emb(sigma)  # [B, N, d]
        ctrl_emb = self.ctrl_emb(controller_inputs) if controller_inputs is not None else None

        # patchify, fwd, unpatchify
        x = self.proj_in(x)
        x = self.transformer(x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id, kv_cache)
        x = self.proj_out(x, cond)
        return x

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_timestamp: Optional[Tensor] = None,
        fps: float = None,
        prompt_emb: Optional[TensorDict] = None,
        controller_inputs: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None
    ):
        """
        x: [B, N, C, H, W],
        sigma: [B, N]
        frame_timestamp: [B, N]
        prompt_emb: [B, P, D]
        controller_inputs: [B, N, I]
        doc_id: [B, N]
        """
        B, N, C, H, W = x.shape

        assert (fps is None) != (frame_timestamp is None), "Must specify fps or frame timestamps"
        if frame_timestamp is None:
            frame_timestamp = self.get_frame_timestamps(fps, N, x.device)

        pos_ids = self.get_pos_ids(frame_timestamp, H, W)
        if doc_id is not None:
            doc_id = doc_id.repeat_interleave(H * W, dim=1)

        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')
        x = self.flat_forward(x, pos_ids, sigma, prompt_emb, controller_inputs, doc_id, kv_cache)
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', h=H, w=W)
        return x

    def get_frame_timestamps(self, fps: torch.Tensor, num_frames: int, device):
        assert fps.dim() == 1 and fps.dtype == torch.long and torch.all(self.config.base_fps % fps == 0)
        scale = (self.config.base_fps // fps).unsqueeze(1)
        return torch.arange(num_frames, device=device).unsqueeze(0) * scale

    @staticmethod
    def get_pos_ids(seq_ts: torch.Tensor, H: int, W: int) -> TensorDict:
        """Positions for [B, F*H*W]; seq_ts is [B,F] (long)."""
        B, F = seq_ts.shape
        device = seq_ts.device
        y = torch.arange(H, device=device, dtype=torch.long).repeat_interleave(W)  # [H*W]
        x = torch.arange(W, device=device, dtype=torch.long).repeat(H)             # [H*W]
        return TensorDict(
            {
                "t_pos": seq_ts.repeat_interleave(H * W, 1),  # [B,F*H*W]
                "y_pos": y.repeat(F).expand(B, -1),           # [B,F*H*W]
                "x_pos": x.repeat(F).expand(B, -1),           # [B,F*H*W]
            },
            batch_size=[B, F * H * W],
        )
