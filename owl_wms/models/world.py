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
        self.attn = owl_nn.Attn(config, layer_idx)
        self.cross_attn = owl_nn.CrossAttention(config)
        self.mlp = owl_nn.MLP(config)

        dim = config.d_model
        self.adaln0, self.gate0 = owl_nn.AdaLN(dim), owl_nn.Gate(dim)
        self.adaln1, self.gate1 = owl_nn.AdaLN(dim), owl_nn.Gate(dim)
        self.adaln2, self.gate2 = owl_nn.AdaLN(dim), owl_nn.Gate(dim)

    def forward(self, x, cond, prompt_emb, ctrl_emb, block_mask, kv_cache=None):
        """
        0) Causal Frame Attention
        1) Frame->Text Cross Attention
        2) MLP
        """
        residual = x
        x = self.adaln0(x, cond)
        x = self.attn(x, block_mask, kv_cache)
        x = self.gate0(x, cond) + residual

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
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond) + residual

        return x


class WorldDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_masker = owl_nn.AttnMaskScheduler(config)
        self.blocks = nn.ModuleList([WorldDiTBlock(config, idx) for idx in range(config.n_layers)])

    def forward(self, x, cond, prompt_emb, ctrl_emb, doc_id=None, kv_cache=None):
        enable_ckpt = self.training and getattr(self.config, "gradient_checkpointing", False)

        # generate block masks for each layer
        block_masks = self.attn_masker(
            seq_len=x.size(1),
            doc_id=doc_id,
            kv_cache=kv_cache,
            device=x.device
        )
        for block, block_mask in zip(self.blocks, block_masks):
            if enable_ckpt:
                x = owl_nn.checkpoint(block, x, cond, prompt_emb, ctrl_emb, block_mask, kv_cache)
            else:
                x = block(x, cond, prompt_emb, ctrl_emb, block_mask, kv_cache)
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

        self.timestep_emb = owl_nn.TimestepEmbedding(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config.n_controller_inputs, config.d_model)

        self.transformer = WorldDiT(config)

        P = getattr(config, "patch_size", [1, 1, 1])
        assert P[0] == 1, "Temporal patching not supported; use (1,*,*)."
        self.proj_in = nn.Conv3d(config.channels, config.d_model, kernel_size=P, stride=P, bias=False)
        self.proj_out = owl_nn.FinalLayer(config.d_model, config.channels, kernel_size=P, stride=P, bias=True)

    def get_conditioning_vectors(self, ts_emb):
        # placeholder until we have Dit-Air and move ctrl_emb to cross attn
        return ts_emb

    def forward(
        self,
        x: Tensor,
        ts: Tensor,
        prompt_emb: Optional[TensorDict] = None,
        controller_inputs: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None
    ):
        """
        x: [B, N, C, H, W],
        denoising_ts: [B, N]
        prompt_emb: [???, ]
        controller_inputs: [B, N, I]
        doc_id: [???, ]
        """
        B, N, C, H, W = x.shape

        # embed
        cond = self.get_conditioning_vectors(self.timestep_emb(ts))  # [B, N, d]
        ctrl_emb = self.ctrl_emb(controller_inputs) if controller_inputs is not None else None

        # patchify
        x = self.proj_in(eo.rearrange(x, 'b n c h w -> b c n h w'))
        _, _, n, h, w = x.shape
        assert (self.config.height, self.config.width) == (h, w), f"{h}, {w}"

        # transformer fwd
        x = eo.rearrange(x, 'b d n h w -> b (n h w) d')
        x = self.transformer(x, cond, prompt_emb, ctrl_emb, doc_id, kv_cache)
        x = eo.rearrange(x, 'b (n h w) d -> b d n h w', n=n, h=h, w=w)

        # unpatchify
        x = self.proj_out(x, cond)
        return eo.rearrange(x, 'b c n h w -> b n c h w')
