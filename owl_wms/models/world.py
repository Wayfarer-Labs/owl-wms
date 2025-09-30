from typing import Optional, List
from torch import Tensor

import einops as eo
from tensordict import TensorDict

import torch
from torch import nn
import torch.nn.functional as F

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




class CondHead(nn.Module):
    """Per-layer conditioning head: bias_in → SiLU → Linear → chunk(n_cond)."""
    n_cond = 6

    def __init__(self, config):
        super().__init__()
        self.bias_in = nn.Parameter(torch.zeros(config.d_model)) if config.noise_conditioning == "wan" else None
        self.cond_proj = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(self.n_cond)]
        )

        # AdaLN-Zero
        if self.bias_in is not None:
            self.bias_in.detach().zero_()
        for p in self.cond_proj:
            p.weight.detach().zero_()

    def forward(self, cond):
        cond = cond + self.bias_in if self.bias_in is not None else cond
        h = F.silu(cond)
        return tuple(p(h) for p in self.cond_proj)


class WorldDiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = owl_nn.Attn(config, layer_idx)
        self.cross_attn = owl_nn.CrossAttention(config)
        self.mlp = owl_nn.MLP(config)
        self.cond_head = CondHead(config)

    def forward(self, x, pos_ids, cond, prompt_emb, ctrl_emb, block_mask, kv_cache=None):
        """
        0) Causal Frame Attention
        1) Frame->Text Cross Attention (TODO)
        2) MLP
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        residual = x
        x = owl_nn.ada_rmsnorm(x, s0, b0)
        x = self.attn(x, pos_ids, block_mask, kv_cache)
        x = owl_nn.ada_gate(x, g0)
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
        def cond_mlp(xm, sm, bm, gm):
            y = self.mlp(owl_nn.ada_rmsnorm(xm, sm, bm))
            return xm + owl_nn.ada_gate(y, gm)

        do_ckpt = self.config.gradient_checkpointing and self.training
        x = owl_nn.maybe_ckpt(do_ckpt, cond_mlp, x, s1, b1, g1)

        return x


class WorldDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_masker = owl_nn.AttnMaskScheduler(config)
        self.blocks = nn.ModuleList([WorldDiTBlock(config, idx) for idx in range(config.n_layers)])

        if self.config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

        # Shared RoPE buffers
        ref_rope = self.blocks[0].attn.rope
        for blk in self.blocks[1:]:
            blk.attn.rope = ref_rope

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
        for block, block_mask, in zip(self.blocks, block_masks):
            x = block(x, pos_ids, cond, prompt_emb, ctrl_emb, block_mask, kv_cache)
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

        self.denoise_step_emb = owl_nn.NoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config.n_controller_inputs, config.d_model)

        self.transformer = WorldDiT(config)

        self.patch = (ph, pw) = tuple(getattr(config, "patch", (1, 1)))

        if self.patch == (1, 1):
            self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        else:
            self.proj_in = nn.Conv2d(config.channels, config.d_model, kernel_size=(ph, pw), stride=(ph, pw), bias=False)

        self.out_norm = owl_nn.AdaLN(config.d_model)
        self.proj_out = nn.Linear(config.d_model, config.channels * ph * pw, bias=True)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_timestamp: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        prompt_emb: Optional[TensorDict] = None,
        controller_inputs: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None
    ):
        """
        x: [B, N, C, H, W],
        sigma: [B, N]
        frame_timestamp: [B, N]
        fps: [B]
        prompt_emb: [B, P, D]
        controller_inputs: [B, N, I]
        doc_id: [B, N]
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        assert (H % ph == 0) and (W % pw == 0), "H, W must be divisible by patch"
        Hp, Wp = H // ph, W // pw

        assert (fps is None) != (frame_timestamp is None), "Must specify fps or frame timestamps"
        if frame_timestamp is None:
            frame_timestamp = self.get_frame_timestamps(fps, N, x.device)
        pos_ids = self.get_pos_ids(frame_timestamp, Hp, Wp)

        assert doc_id is None or kv_cache is None, "Cannot use sequence packing with kv caching"
        if doc_id is not None:
            doc_id = doc_id.repeat_interleave(Hp * Wp, dim=1)

        # embed
        cond = self.denoise_step_emb(sigma)  # [B, N, d]
        ctrl_emb = self.ctrl_emb(controller_inputs) if controller_inputs is not None else None

        if self.patch == (1, 1):
            x = eo.rearrange(x, 'b n c h w -> b (n h w) c')
            x = self.proj_in(x)
            x = self.transformer(x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id, kv_cache)
            x = self.proj_out(F.silu(self.out_norm(x, cond)))
            x = eo.rearrange(x, 'b (n h w) c -> b n c h w', h=H, w=W)
        else:
            # patchify
            x = eo.rearrange(x, 'b n c h w -> (b n) c h w')
            x = self.proj_in(x)
            x = eo.rearrange(x, '(b n) d h w -> b (n h w) d', b=B, n=N)
            # backbone fwd
            x = self.transformer(x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id, kv_cache)
            # unpatchify
            x = self.proj_out(F.silu(self.out_norm(x, cond)))
            x = eo.rearrange(x, 'b (n h w) (c ph pw) -> b n c (h ph) (w pw)', n=N, h=Hp, w=Wp, ph=ph, pw=pw)

        return x

    def get_frame_timestamps(self, fps: torch.Tensor, num_frames: int, device):
        assert fps.dim() == 1 and fps.dtype == torch.long
        if not (self.config.base_fps % fps).eq(0).all():
            raise ValueError(f"base_fps={int(self.config.base_fps)} seen_fps={torch.unique(fps.detach().cpu()).tolist()}")
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
