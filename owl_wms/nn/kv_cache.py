from torch import Tensor

import torch
from torch import nn
from ..configs import TransformerConfig


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
        self.offset = 0

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
        self.cache = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]
        self.offset = 0

    @torch.no_grad()
    def get(self, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        if self.noise_caches > 0.0:
            k = k + torch.randn_like(k) * self.noise_caches
            v = v + torch.randn_like(v) * self.noise_caches
        return k,v

    @torch.no_grad()
    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"

        self.cache[layer_ind] = (new_k,new_v)

    @torch.no_grad()
    def truncate(self, truncate_amt, front = False):
        """
        Truncate/eject frames from the KV cache
        """
        truncate_amt = truncate_amt * self.config.tokens_per_frame
        def tuple_truncate(k, v):
            if front:
                k = k[:,:,:-truncate_amt]
                v = v[:,:,:-truncate_amt]
            else:
                k = k[:,:,truncate_amt:]
                v = v[:,:,truncate_amt:]
            return k, v

        for i in range(self.config.n_layers):
            self.cache[i] = tuple_truncate(*self.cache[i])
        if not front:
            # When ejecting first frame, window slides forward, offset increases
            self.offset += truncate_amt // self.config.tokens_per_frame

    def length_at(self, idx):
        return self.cache[idx][0].shape[2]

    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0][0].shape[2]

    def n_frames(self):
        assert len(self) % self.config.tokens_per_frame == 0
        return len(self) // self.config.tokens_per_frame

    @property
    def shape(self):
        return self.cache[0][0].shape


class StaticKVCache(nn.Module):
    def __init__(self, config, batch_size, dtype):
        super().__init__()

        # Exclude last N tokens from caching
        self.n_uncached = config.tokens_per_frame

        B = batch_size
        H = config.n_heads
        L = config.n_frames * config.tokens_per_frame
        Dh = config.d_model // config.n_heads
        NL = config.n_layers

        self.k = nn.Buffer(torch.empty(NL, B, H, L, Dh, dtype=dtype), persistent=False)
        self.v = nn.Buffer(torch.empty(NL, B, H, L, Dh, dtype=dtype), persistent=False)
        self.offset = nn.Buffer(torch.zeros(NL, dtype=torch.long), persistent=False)

    @torch.inference_mode()
    def upsert(self, k: Tensor, v: Tensor, layer: int):
        T = k.size(2)
        start = self.offset[layer]
        end = start + T

        torch._assert(end <= self.k.size(3), "KV cache overflow")
        torch._assert(T >= self.n_uncached, "chunk shorter than n_uncached")

        self.k[layer, :, :, start:end, :].copy_(k)
        self.v[layer, :, :, start:end, :].copy_(v)
        self.offset[layer].fill_(end - self.n_uncached)

        return self.k[layer, :, :, :end, :], self.v[layer, :, :, :end, :]  # TODO: make static kv
