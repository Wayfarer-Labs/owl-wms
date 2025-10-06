from typing import Optional, List
from torch import Tensor

import einops as eo
from einops.layers.torch import Rearrange
from tensordict import TensorDict
import math

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

        def cond_mlp(xm, sm, bm, gm):
            res = xm
            xm = self.mlp(owl_nn.ada_rmsnorm(xm, sm, bm))
            return owl_nn.ada_gate(xm, gm) + res

        do_ckpt = self.config.gradient_checkpointing and self.training
        x = owl_nn.maybe_ckpt(do_ckpt, cond_mlp, x, s1, b1, g1)

        return x


class WorldDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_masker = owl_nn.AttnMaskScheduler(config)
        self.local_window = nn.Buffer(torch.tensor(config.local_window, dtype=torch.int32), persistent=False)
        self.global_window = nn.Buffer(torch.tensor(config.global_window, dtype=torch.int32), persistent=False)

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

    def forward(self, x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id=None, kv_cache=None, curr_frame_mask=None):
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
            curr_frame_mask=curr_frame_mask,
            device=x.device,
            local_window=self.local_window,
            global_window=self.global_window,
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

        self.sink_emb = nn.Parameter(torch.zeros(1, 1, config.d_model, dtype=torch.float32))
        self.sink_emb.detach().normal_()

        self.transformer = WorldDiT(config)

        self.patch = tuple(getattr(config, "patch", (1, 1)))

        C, D = config.channels, config.d_model
        self.patchify = nn.Sequential(
            Rearrange('b n c h w -> (b n) c h w'),
            nn.Conv2d(C, D, kernel_size=self.patch, stride=self.patch, bias=False),
        )
        self.unpatchify = nn.Linear(D, C * math.prod(self.patch), bias=True)
        self.out_norm = owl_nn.AdaLN(config.d_model)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_timestamp: Tensor,
        prompt_emb: Optional[TensorDict] = None,
        controller_inputs: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None,
        curr_frame_mask: Optional[Tensor] = None,
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
        ph, pw = self.patch
        assert (H % ph == 0) and (W % pw == 0), "H, W must be divisible by patch"
        Hp, Wp = H // ph, W // pw

        pos_ids = self.get_pos_ids(frame_timestamp, Hp, Wp)

        if curr_frame_mask is not None:
            torch._assert(curr_frame_mask.size(1) == N, "curr_frame_mask must be frame-length")
            curr_frame_mask = curr_frame_mask.repeat_interleave(Hp * Wp, 1)

        assert doc_id is None or kv_cache is None, "Cannot use sequence packing with kv caching"
        if doc_id is not None:
            doc_id = doc_id.repeat_interleave(Hp * Wp, dim=1)

        # embed
        cond = self.denoise_step_emb(sigma)  # [B, N, d]
        ctrl_emb = self.ctrl_emb(controller_inputs) if controller_inputs is not None else None

        x = eo.rearrange(self.patchify(x), '(b n) d hp wp -> b (n hp wp) d', b=B, n=N)

        assert curr_frame_mask is None
        x, pos_ids, doc_id = self.apply_attention_sink(x, pos_ids, doc_id, kv_cache)

        x = self.transformer(x, pos_ids, cond, prompt_emb, ctrl_emb, doc_id, kv_cache, curr_frame_mask)

        if not (kv_cache is not None and torch.any(getattr(kv_cache, "kv_offset", 0) > 0)):
            x = x[:, 1:]  # remove sink token we just added

        x = F.silu(self.out_norm(x, cond))
        x = eo.rearrange(
            self.unpatchify(x),
            'b (n hp wp) (c ph pw) -> b n c (hp ph) (wp pw)',
            n=N, hp=Hp, wp=Wp, ph=ph, pw=pw
        )
        return x

    def apply_attention_sink(self, x, pos_ids, doc_id, kv_cache):
        B = x.size(0)

        if kv_cache is not None and torch.any(kv_cache.kv_offset > 0):
            # if kv cache is populated, then the sink token already is present
            return x, pos_ids, doc_id

        if doc_id is not None:
            sink_id = torch.full((B, 1), -42, dtype=torch.long, device=x.device)
            doc_id = torch.cat([sink_id, doc_id], dim=1)

        x = torch.cat([owl_nn.layer_norm(self.sink_emb).expand(B, 1, -1), x], dim=1)

        pos_ids["t_pos"] = pos_ids["t_pos"] + 1
        zeros = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        sink_pos = TensorDict({"t_pos": zeros, "y_pos": zeros, "x_pos": zeros}, batch_size=[B, 1])
        pos_ids = torch.cat([sink_pos, pos_ids], dim=1)

        return x, pos_ids, doc_id

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
