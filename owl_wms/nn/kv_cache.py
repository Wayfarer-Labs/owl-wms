import torch
from torch import nn

from ..configs import TransformerConfig

class KVCache:
    def __init__(self, config : TransformerConfig, rank: int = 0):
        self.shape = None
        self.config = config

        self.cache = None
        self.device = torch.device('cuda', rank) # NOTE: This is the device that the cache will be on
        self.dtype = torch.bfloat16
        
        self.should_update = False

        self.max_length = config.tokens_per_frame * config.n_frames
        self.noise_caches = 0.0

    def enable_cache_updates(self):
        self.should_update = True
    
    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16, rank: int = 0):
        self.device = torch.device(device, rank)
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        self.shape = (batch_size, self.config.n_heads, 0, self.config.d_model//self.config.n_heads)
        dummy = torch.empty(*self.shape, device = self.device, dtype = self.dtype)
        self.cache = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]

    @torch.no_grad()
    def get(self, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        if self.noise_caches > 0.0:
            k = k + torch.randn_like(k) * self.noise_caches
            v = v + torch.randn_like(v) * self.noise_caches
        return k,v
    
    @torch.no_grad()
    def push(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind] # each [b,h,n,d]
        k = torch.cat([k,new_k],dim=2)
        v = torch.cat([v,new_v],dim=2)
        self.cache[layer_ind] = (k,v)
    
    @torch.no_grad()
    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"

        def tuple_truncate(k, v):
            k = k[:,:,-self.max_length:]
            v = v[:,:,-self.max_length:]
            return k, v

        self.cache[layer_ind] = tuple_truncate(new_k,new_v)
    
    @torch.no_grad()
    def truncate(self, truncate_amt):
        """
        Truncate frames from the KV cache
        """
        truncate_amt = truncate_amt * self.config.tokens_per_frame
        def tuple_truncate(k, v):
            k = k[:,:,truncate_amt:]
            v = v[:,:,truncate_amt:]
            return k, v

        for i in range(self.config.n_layers):
            self.cache[i] = tuple_truncate(*self.cache[i])

    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0][0].shape[2]

    def shape(self):
        return self.shape
