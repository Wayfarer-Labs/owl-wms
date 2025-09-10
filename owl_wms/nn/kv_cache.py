import torch
import os
from ..configs import TransformerConfig
from ..quant.fp8_kv import (
    quantize_per_head,
    dequantize_per_head,
    quantize_per_head_timewise,
    dequantize_per_head_timewise,
)
from ..quant.kv_triton import fused_rotate_dequant_i8_to_bf16 as _triton_rot_dq


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

            return old_k.contiguous(), old_v.contiguous()
        else:
            return self.k_cache[layer_ind].contiguous(), self.v_cache[layer_ind].contiguous()

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
        kv_storage: str | None = None,   # None|'i8_scale'|'mxfp'
        kv_bits: int = 8,                # 8 or 4 when kv_storage='mxfp'
        ema_momentum: float = 0.99,
        kv_late_layers: int | None = None,
        k_fp8: bool | None = None,
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
        late_layers = kv_late_layers if kv_late_layers is not None else int(os.environ.get("OWL_KV_LATE_LAYERS", "0"))
        start_fp8 = max(0, L - late_layers) if late_layers > 0 else 0
        k_fp8_global = (k_fp8 if k_fp8 is not None else bool(int(os.environ.get("OWL_K_FP8", "1"))))
        self.k_use_fp8 = [(k_fp8_global and (li >= start_fp8)) for li in range(L)]
        self.v_use_fp8 = [(li >= start_fp8) for li in range(L)]

        # int8 storage for layers where FP8/MXFP is used; None otherwise
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
        self.kv_storage = kv_storage or os.environ.get("OWL_KV_STORAGE", "i8_scale")
        self.kv_bits = int(os.environ.get("OWL_KV_BITS", str(kv_bits)))
        self.ema_momentum = ema_momentum

        # Profiling (quant/dequant) when enabled via OWL_PROFILE_KV
        import os as _os
        self._profile = bool(int(_os.environ.get("OWL_PROFILE_KV", "0")))
        # Ring debugging/guards (Stage A)
        self._ring_mode = _os.environ.get("OWL_RING_MODE", "roll")
        self._ring_guard = bool(int(_os.environ.get("OWL_RING_GUARD", "0")))
        self._window_tokens = T
        self._filled_tokens = [0] * L
        # Pointer/compile policy
        self._compile_enabled = bool(int(_os.environ.get("OWL_COMPILE", "1")))
        self._pointer_allow_compile = bool(int(_os.environ.get("OWL_RING_POINTER_COMPILE", "0")))
        # Disable pointer mode under compile unless explicitly allowed
        self._use_pointer = (self._ring_mode == "pointer") and ((not self._compile_enabled) or self._pointer_allow_compile)
        self._use_scratch = bool(int(_os.environ.get("OWL_RING_SCRATCH", "1"))) if self._use_pointer else False
        # Write pointers per layer (next write index in time dimension)
        self._wp = [0] * L
        # Optional BF16 scratch buffers for assembled contiguous K/V (pointer mode)
        if self._use_pointer and self._use_scratch:
            self._scratch_k = [torch.empty(batch_size, H, T, D, device=device, dtype=torch.bfloat16) for _ in range(L)]
            self._scratch_v = [torch.empty(batch_size, H, T, D, device=device, dtype=torch.bfloat16) for _ in range(L)]
        else:
            self._scratch_k = [None] * L
            self._scratch_v = [None] * L
        # Blockwise LUT indices for pointer mode (compiler-friendly per-phase gather)
        self._block_len = config.tokens_per_frame
        assert T % self._block_len == 0, "T must be divisible by tokens_per_frame"
        self._num_blocks = T // self._block_len
        t_arange = torch.arange(T, device=device, dtype=torch.long)
        # 1D indices (preferred for performance via index_select)
        self._lut_idx_1d = [((t_arange + p * self._block_len) % T).to(torch.long) for p in range(self._num_blocks)]
        if self._ring_mode == "pointer" and self._compile_enabled and (not self._pointer_allow_compile) and self._profile:
            try:
                print("[KV-Profile] pointer mode disabled under compile; using roll for stability")
            except Exception:
                pass
        self.quant_ms = 0.0
        self.dequant_ms = 0.0

        # Helper to avoid profiling inside torch.compile graphs
        try:
            import torch._dynamo as _dynamo
            self._is_compiling = _dynamo.is_compiling
        except Exception:
            self._is_compiling = lambda: False

    def _can_profile(self) -> bool:
        return self._profile and not self._is_compiling()

    def set_local_layers(self, local_layers: list[bool], local_tail_len_tokens: int):
        # No-op in legacy mode; keeping signature for compatibility
        return

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
        if not self._use_pointer:
            if self.k_use_fp8[layer_ind]:
                if self._can_profile():
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                k = dequantize_per_head_timewise(self.k_i8[layer_ind], self.scale_k[layer_ind])
                if self._can_profile():
                    e1.record(); torch.cuda.synchronize(); self.dequant_ms += e0.elapsed_time(e1)
            else:
                k = self.k_bf16[layer_ind]
            if self.v_use_fp8[layer_ind]:
                if self._can_profile():
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                v = dequantize_per_head_timewise(self.v_i8[layer_ind], self.scale_v[layer_ind])
                if self._can_profile():
                    e1.record(); torch.cuda.synchronize(); self.dequant_ms += e0.elapsed_time(e1)
            else:
                v = self.v_bf16[layer_ind]
        else:
            # Pointer mode (LUT): compiler-friendly contiguous view using precomputed blockwise indices
            filled = self._filled_tokens[layer_ind]
            T = self._window_tokens
            H = self.config.n_heads
            D = self.config.d_model // H
            if filled == 0:
                k = self.k_bf16[layer_ind][:, :, :0, :] if not self.k_use_fp8[layer_ind] else self.k_i8[layer_ind][:, :, :0, :].new_empty((self.batch_size, H, 0, D), dtype=torch.bfloat16)
                v = self.v_bf16[layer_ind][:, :, :0, :] if not self.v_use_fp8[layer_ind] else self.v_i8[layer_ind][:, :, :0, :].new_empty((self.batch_size, H, 0, D), dtype=torch.bfloat16)
            else:
                head = (self._wp[layer_ind] - filled) % T
                # Enforce block alignment; if misaligned, fall back to roll assembly
                if (filled % self._block_len) != 0 or (head % self._block_len) != 0:
                    # Fallback: roll-based contiguous window
                    if self.k_use_fp8[layer_ind]:
                        k_full = dequantize_per_head_timewise(self.k_i8[layer_ind], self.scale_k[layer_ind])
                    else:
                        k_full = self.k_bf16[layer_ind]
                    if self.v_use_fp8[layer_ind]:
                        v_full = dequantize_per_head_timewise(self.v_i8[layer_ind], self.scale_v[layer_ind])
                    else:
                        v_full = self.v_bf16[layer_ind]
                    k = torch.roll(k_full, shifts=-head, dims=2).contiguous()
                    v = torch.roll(v_full, shifts=-head, dims=2).contiguous()
                else:
                    offset_t = int(head)
                    if self.k_use_fp8[layer_ind]:
                        k = _triton_rot_dq(self.k_i8[layer_ind], self.scale_k[layer_ind], offset_t)
                    else:
                        idx1d = (self._lut_idx_1d[0] + offset_t) % T
                        k = torch.index_select(self.k_bf16[layer_ind], dim=2, index=idx1d).to(torch.bfloat16)
                    if self.v_use_fp8[layer_ind]:
                        v = _triton_rot_dq(self.v_i8[layer_ind], self.scale_v[layer_ind], offset_t)
                    else:
                        idx1d = (self._lut_idx_1d[0] + offset_t) % T
                        v = torch.index_select(self.v_bf16[layer_ind], dim=2, index=idx1d).to(torch.bfloat16)
                # Maintain contiguity
                if not k.is_contiguous():
                    k = k.contiguous()
                if not v.is_contiguous():
                    v = v.contiguous()
        # Enforce contiguity for attention inputs (Stage A invariant)
        if new_k is not None:
            # Match StaticCache.get logical concat for attention read
            # Important: concat the unfilled tail of current window (not truncated) and new_k
            k = torch.cat([k, new_k], dim=2)
            v = torch.cat([v, new_v], dim=2)
        if self._can_profile():
            try:
                wp_print = (self._wp[layer_ind] if self._use_pointer else (self._filled_tokens[layer_ind] % self._window_tokens))
                print(f"[KV-Profile] get: layer={layer_ind}, logical_len={k.shape[2]}, filled={self._filled_tokens[layer_ind]}, wp={wp_print}")
            except Exception:
                pass
        return k, v

    def update(self, new_k, new_v, layer_ind):
        # Quantize and write newest positions into ring (roll window by new_len tokens)
        new_len = new_k.shape[2]
        prev_offset = self.offsets[layer_ind]
        prev_filled = self._filled_tokens[layer_ind]
        if self._ring_guard and not self._is_compiling():
            # Allow large prefill writes; otherwise expect frame-sized updates
            tpf = self.tokens_per_frame
            if new_len < self._window_tokens:
                assert (new_len % tpf) == 0, f"update new_len {new_len} not multiple of tokens_per_frame {tpf}"

        # Select buffers per tensor kind (K/V) depending on FP8 enablement
        if self.k_use_fp8[layer_ind]:
            if self._can_profile():
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
            if self.kv_storage == "mxfp":
                qk, sk = quantize_per_head_timewise(new_k, fmt=self.fmt_k, bits=self.kv_bits)
            else:
                qk, sk = quantize_per_head_timewise(new_k, fmt=self.fmt_k)
            if self._can_profile():
                e1.record(); torch.cuda.synchronize(); self.quant_ms += e0.elapsed_time(e1)
            k_buf = self.k_i8[layer_ind]
        else:
            k_slice = new_k
            k_buf = self.k_bf16[layer_ind]
        if self.v_use_fp8[layer_ind]:
            if self._can_profile():
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
            if self.kv_storage == "mxfp":
                qv, sv = quantize_per_head_timewise(new_v, fmt=self.fmt_v, bits=self.kv_bits)
            else:
                qv, sv = quantize_per_head_timewise(new_v, fmt=self.fmt_v)
            if self._can_profile():
                e1.record(); torch.cuda.synchronize(); self.quant_ms += e0.elapsed_time(e1)
            v_buf = self.v_i8[layer_ind]
        else:
            v_slice = new_v
            v_buf = self.v_bf16[layer_ind]

        if not self._use_pointer:
            # Roll mode (Stage A): constant shift, then tail write
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
        else:
            # Pointer mode (Stage C): write at wp, advance pointer, do not roll
            T = self._window_tokens
            wp = self._wp[layer_ind]
            # Write helper for possibly wrapped slices
            def _write_ring(buf, src, start_idx, is_fp8: bool, is_scale: bool = False):
                if src is None:
                    return
                total = src.shape[2] if not is_scale else src.shape[2]
                # For scales in timewise mode, src shape is [B,H,T_new]
                dim_time = 2 if not is_scale else 2
                first = min(total, T - start_idx)
                second = total - first
                if self._can_profile():
                    try:
                        what = "scale" if is_scale else "tensor"
                        print(f"[KV-Profile] write_ring: layer={layer_ind}, {what}, wp={start_idx}, total={total}, first={first}, second={second}")
                    except Exception:
                        pass
                if not is_scale:
                    buf[:, :, start_idx:start_idx+first, :] = src[:, :, :first, :]
                    if second > 0:
                        buf[:, :, :second, :] = src[:, :, first:first+second, :]
                else:
                    # scale tensors have [B,H,T]
                    buf[:, :, start_idx:start_idx+first] = src[:, :, :first]
                    if second > 0:
                        buf[:, :, :second] = src[:, :, first:first+second]

            if self.k_use_fp8[layer_ind]:
                _write_ring(self.k_i8[layer_ind], qk, wp, is_fp8=True, is_scale=False)
                _write_ring(self.scale_k[layer_ind], sk, wp, is_fp8=True, is_scale=True)
            else:
                _write_ring(self.k_bf16[layer_ind], k_slice, wp, is_fp8=False, is_scale=False)
            if self.v_use_fp8[layer_ind]:
                _write_ring(self.v_i8[layer_ind], qv, wp, is_fp8=True, is_scale=False)
                _write_ring(self.scale_v[layer_ind], sv, wp, is_fp8=True, is_scale=True)
            else:
                _write_ring(self.v_bf16[layer_ind], v_slice, wp, is_fp8=False, is_scale=False)
            self._wp[layer_ind] = (wp + new_len) % T

        self.offsets[layer_ind] += new_len
        self._filled_tokens[layer_ind] = min(self._window_tokens, prev_filled + new_len)
        if self._ring_guard and not self._is_compiling():
            try:
                assert self.offsets[layer_ind] == prev_offset + new_len, "offset not advanced by new_len"
            except AssertionError as e:
                raise
        if self._can_profile():
            try:
                wp = (self._wp[layer_ind] if self._use_pointer else (self._filled_tokens[layer_ind] % self._window_tokens))
                print(f"[KV-Profile] update_done: layer={layer_ind}, new_len={new_len}, offset={self.offsets[layer_ind]}, filled={self._filled_tokens[layer_ind]}, wp={wp}")
            except Exception:
                pass

    def eject(self):
        # Maintain same API; no-op since ring behavior already ejects oldest tokens.
        return

    def length_at(self, idx):
        # Full logical window length in tokens
        buf = self.k_i8[idx] if self.k_i8[idx] is not None else self.k_bf16[idx]
        return buf.shape[2]

    def get_offset(self, idx=0):
        return self.offsets[idx]

    @property
    def shape(self):
        return self.k_i8[0].shape