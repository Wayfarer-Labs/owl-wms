import torch
import os
from ..configs import TransformerConfig
from ..quant.fp8_kv import (
    quantize_per_head,
    dequantize_per_head,
    quantize_per_head_timewise,
    dequantize_per_head_timewise,
)


def KVCache(config : TransformerConfig):
    if config.backbone in ("dit", "mmdit"):
        return SingleKVCache(config)
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")


class SingleKVCache:
    def __init__(self, config: TransformerConfig):
        self.config = config

        self.cache = None
        self.device = 'cuda'
        self.dtype = torch.bfloat16

        self.should_update = False

        self.noise_caches = 0.0
        self.offsets = [0] * self.config.n_layers

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        shape = (batch_size, self.config.n_heads, 0, self.config.d_model//self.config.n_heads)
        dummy = torch.empty(*shape, device = self.device, dtype = self.dtype)
        self.cache = [[torch.empty_like(dummy), torch.empty_like(dummy)] for _ in range(self.config.n_layers)]
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind, new_k = None, new_v = None):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        if new_k is not None:
            k = torch.cat([k, new_k], dim=2)
            v = torch.cat([v, new_v], dim=2)
        return k,v

    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"

        new_len = new_k.shape[2]

        self.offsets[layer_ind] += new_len

        self.cache[layer_ind] = [
            torch.cat([self.cache[layer_ind][0], new_k], dim=2),
            torch.cat([self.cache[layer_ind][1], new_v], dim=2)
        ]

    def eject(self):
        for i in range(self.config.n_layers):
            self.cache[i][0] = self.cache[i][0][:,:,self.config.tokens_per_frame:]
            self.cache[i][1] = self.cache[i][1][:,:,self.config.tokens_per_frame:]

    def length_at(self, idx):
        return self.cache[idx][0].shape[2]
    
    def get_offset(self, idx=0):
        return self.offsets[idx]

    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0][0].shape[2]

    def n_frames(self):
        assert len(self) % self.config.tokens_per_frame == 0
        return len(self) // self.config.tokens_per_frame
    
    def clone(self):
        # Clones all tensors for max-autotune to work properly
        for i in range(self.config.n_layers):
            self.cache[i] = (self.cache[i][0].clone(), self.cache[i][1].clone())
        return self
    
    def detach(self):
        for i in range(self.config.n_layers):
            self.cache[i] = (self.cache[i][0].detach(), self.cache[i][1].detach())
        return self

    @property
    def shape(self):
        return self.cache[0][0].shape

class StaticCache:
    def __init__(
        self,
        config: TransformerConfig,
        max_length = 120,
        batch_size = 1,
        device = 'cuda',
        dtype = torch.bfloat16
    ):
        self.config = config

        self.cache = None
        self.device = 'cuda'
        self.dtype = torch.bfloat16

        self.should_update = False

        self.max_length = max_length
        self.batch_size = batch_size

        self.k_cache = [torch.empty(
            batch_size,
            config.n_heads,
            max_length * config.tokens_per_frame,
            config.d_model//config.n_heads,
            device = device,
            dtype = dtype
        ) for _ in range(config.n_layers)]
        self.v_cache = [torch.empty(
            batch_size,
            config.n_heads,
            max_length * config.tokens_per_frame,
            config.d_model//config.n_heads,
            device = device,
            dtype = dtype
        ) for _ in range(config.n_layers)]

        self.tokens_per_frame = config.tokens_per_frame
        self.offsets = [0] * self.config.n_layers

    def clone(self):
        new_cache = StaticCache(self.config, self.max_length, self.batch_size, self.device, self.dtype)
        new_cache.k_cache = [k.clone() for k in self.k_cache]
        new_cache.v_cache = [v.clone() for v in self.v_cache]
        new_cache.offsets = self.offsets.copy()
        return new_cache

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind, new_k = None, new_v = None):
        if new_k is not None and new_v is not None:
            old_k = self.k_cache[layer_ind].clone()
            old_v = self.v_cache[layer_ind].clone()

            old_k = torch.roll(old_k, shifts = -new_k.shape[2], dims = 2)
            old_v = torch.roll(old_v, shifts = -new_v.shape[2], dims = 2)

            old_k[:,:,-new_k.shape[2]:] = new_k
            old_v[:,:,-new_v.shape[2]:] = new_v

            return old_k, old_v
        else:
            return self.k_cache[layer_ind], self.v_cache[layer_ind]

    def update(self, new_k, new_v, layer_ind):
        new_len = new_k.shape[2]
        if new_len > self.tokens_per_frame: # More than one frame, full cache
            self.k_cache[layer_ind][:,:,:,:new_len] = new_k
            self.v_cache[layer_ind][:,:,:,:new_len] = new_v
            self.offsets[layer_ind] = new_len
        else: # step forward one
            self.k_cache[layer_ind] = torch.roll(self.k_cache[layer_ind], shifts = -new_len, dims = 2)
            self.v_cache[layer_ind] = torch.roll(self.v_cache[layer_ind], shifts = -new_len, dims = 2)
            self.k_cache[layer_ind][:,:,-new_len:] = new_k
            self.v_cache[layer_ind][:,:,-new_len:] = new_v
            self.offsets[layer_ind] += new_len

    def truncate(self, truncate_amt=1, front = False):
        """
        Truncate/eject frames from the KV cache
        This isn't needed in a static cache
        """
        return
    
    def get_offset(self, idx=0):
        return self.offsets[idx]

    def clone(self):
        # Clones all tensors for max-autotune to work properly
        for i in range(self.config.n_layers):
            self.k_cache[i] = self.k_cache[i].clone()
            self.v_cache[i] = self.v_cache[i].clone()
        return self
    
    def detach(self):
        for i in range(self.config.n_layers):
            self.k_cache[i] = self.k_cache[i].detach()
            self.v_cache[i] = self.v_cache[i].detach()
        return self

    @property
    def shape(self):
        return self.k_cache[0].shape


class QuantizedStaticCache:
    def __init__(
        self,
        config: TransformerConfig,
        max_length = 120,
        batch_size = 1,
        device = 'cuda',
        dtype = torch.bfloat16,
        fmt_k: str = 'e5m2',
        fmt_v: str = 'e4m3',
        ema_momentum: float = 0.99,
    ):
        self.config = config

        self.device = device
        self.dtype = dtype
        self.should_update = False

        self.max_length = max_length
        self.batch_size = batch_size
        self.tokens_per_frame = config.tokens_per_frame

        # Match StaticCache semantics: allocate tokens = frames * tokens_per_frame
        # Caller passes frames (init_len from pipeline); convert to tokens.
        T = max_length * config.tokens_per_frame
        H = config.n_heads
        D = config.d_model // H
        L = config.n_layers

        # Layer-wise enablement: allow FP8 only for late layers; and optionally keep K in BF16
        late_layers = int(os.environ.get("OWL_KV_LATE_LAYERS", "0"))
        start_fp8 = max(0, L - late_layers) if late_layers > 0 else 0
        k_fp8_global = bool(int(os.environ.get("OWL_K_FP8", "1")))
        self.k_use_fp8 = [(k_fp8_global and (li >= start_fp8)) for li in range(L)]
        self.v_use_fp8 = [(li >= start_fp8) for li in range(L)]

        # int8 storage for layers where FP8 is used; None otherwise
        self.k_i8 = [
            (torch.empty(batch_size, H, T, D, device=device, dtype=torch.int8) if self.k_use_fp8[li] else None)
            for li in range(L)
        ]
        self.v_i8 = [
            (torch.empty(batch_size, H, T, D, device=device, dtype=torch.int8) if self.v_use_fp8[li] else None)
            for li in range(L)
        ]

        # BF16 storage for layers not using FP8; None otherwise
        self.k_bf16 = [
            (torch.empty(batch_size, H, T, D, device=device, dtype=torch.bfloat16) if not self.k_use_fp8[li] else None)
            for li in range(L)
        ]
        self.v_bf16 = [
            (torch.empty(batch_size, H, T, D, device=device, dtype=torch.bfloat16) if not self.v_use_fp8[li] else None)
            for li in range(L)
        ]

        # scales per token (timewise) for layers using FP8: [B,H,T]; None otherwise
        self.scale_k = [
            (torch.ones(batch_size, H, T, device=device, dtype=torch.bfloat16) if self.k_use_fp8[li] else None)
            for li in range(L)
        ]
        self.scale_v = [
            (torch.ones(batch_size, H, T, device=device, dtype=torch.bfloat16) if self.v_use_fp8[li] else None)
            for li in range(L)
        ]

        self.offsets = [0] * config.n_layers
        self.fmt_k = fmt_k
        self.fmt_v = fmt_v
        self.ema_momentum = ema_momentum

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        # logical offsets reset; underlying slabs reused
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind, new_k = None, new_v = None):
        # Return BF16 K/V for current window (dequant if FP8)
        if self.k_use_fp8[layer_ind]:
            k = dequantize_per_head_timewise(self.k_i8[layer_ind], self.scale_k[layer_ind])
        else:
            k = self.k_bf16[layer_ind]
        if self.v_use_fp8[layer_ind]:
            v = dequantize_per_head_timewise(self.v_i8[layer_ind], self.scale_v[layer_ind])
        else:
            v = self.v_bf16[layer_ind]
        if new_k is not None:
            k = torch.cat([k, new_k], dim=2)
            v = torch.cat([v, new_v], dim=2)
        return k, v

    def update(self, new_k, new_v, layer_ind):
        # Quantize and write newest positions into ring (roll window by new_len tokens)
        new_len = new_k.shape[2]

        # Select buffers per tensor kind (K/V) depending on FP8 enablement
        if self.k_use_fp8[layer_ind]:
            qk, sk = quantize_per_head_timewise(new_k, fmt=self.fmt_k)
            k_buf = self.k_i8[layer_ind]
        else:
            k_slice = new_k
            k_buf = self.k_bf16[layer_ind]
        if self.v_use_fp8[layer_ind]:
            qv, sv = quantize_per_head_timewise(new_v, fmt=self.fmt_v)
            v_buf = self.v_i8[layer_ind]
        else:
            v_slice = new_v
            v_buf = self.v_bf16[layer_ind]

        # Write K
        T_cur_k = k_buf.shape[2]
        if T_cur_k == 0 or new_len >= T_cur_k:
            write_len_k = min(new_len, T_cur_k) if T_cur_k > 0 else 0
            if write_len_k > 0:
                if self.k_use_fp8[layer_ind]:
                    k_buf[:, :, -write_len_k:, :] = qk[:, :, -write_len_k:, :]
                    self.scale_k[layer_ind][:, :, -write_len_k:] = sk[:, :, -write_len_k:]
                else:
                    k_buf[:, :, -write_len_k:, :] = k_slice[:, :, -write_len_k:, :]
        else:
            if self.k_use_fp8[layer_ind]:
                self.k_i8[layer_ind] = torch.roll(k_buf, shifts=-new_len, dims=2)
                self.k_i8[layer_ind][:, :, -new_len:, :] = qk
                self.scale_k[layer_ind] = torch.roll(self.scale_k[layer_ind], shifts=-new_len, dims=2)
                self.scale_k[layer_ind][:, :, -new_len:] = sk
            else:
                self.k_bf16[layer_ind] = torch.roll(k_buf, shifts=-new_len, dims=2)
                self.k_bf16[layer_ind][:, :, -new_len:, :] = k_slice

        # Write V
        T_cur_v = v_buf.shape[2]
        if T_cur_v == 0 or new_len >= T_cur_v:
            write_len_v = min(new_len, T_cur_v) if T_cur_v > 0 else 0
            if write_len_v > 0:
                if self.v_use_fp8[layer_ind]:
                    v_buf[:, :, -write_len_v:, :] = qv[:, :, -write_len_v:, :]
                    self.scale_v[layer_ind][:, :, -write_len_v:] = sv[:, :, -write_len_v:]
                else:
                    v_buf[:, :, -write_len_v:, :] = v_slice[:, :, -write_len_v:, :]
        else:
            if self.v_use_fp8[layer_ind]:
                self.v_i8[layer_ind] = torch.roll(v_buf, shifts=-new_len, dims=2)
                self.v_i8[layer_ind][:, :, -new_len:, :] = qv
                self.scale_v[layer_ind] = torch.roll(self.scale_v[layer_ind], shifts=-new_len, dims=2)
                self.scale_v[layer_ind][:, :, -new_len:] = sv
            else:
                self.v_bf16[layer_ind] = torch.roll(v_buf, shifts=-new_len, dims=2)
                self.v_bf16[layer_ind][:, :, -new_len:, :] = v_slice

        self.offsets[layer_ind] += new_len

    def eject(self):
        # Maintain same API; no-op since ring behavior already ejects oldest tokens.
        return

    def length_at(self, idx):
        # Full logical window length in tokens
        return self.k_i8[idx].shape[2]

    def get_offset(self, idx=0):
        return self.offsets[idx]

    @property
    def shape(self):
        return self.k_i8[0].shape