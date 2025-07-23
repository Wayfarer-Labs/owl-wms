import torch
from torch import nn
from copy import deepcopy

from ..configs import TransformerConfig

def KVCache(config : TransformerConfig):
    if config.backbone == "dit":
        return SingleKVCache(config)
    elif config.backbone == "mmdit":
        return MMKVCache(config)
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")

class SingleKVCache:
    def __init__(self, config : TransformerConfig):
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
        return len(self) // self.config.tokens_per_frame

    @property
    def shape(self):
        return self.cache[0][0].shape

class MMKVCache(SingleKVCache):
    def __init__(self, config : TransformerConfig):
        super().__init__(config)

        self.cache_2 = None

    def reset(self, batch_size = 1):
        shape = (batch_size, self.config.n_heads, 0, self.config.d_model//self.config.n_heads)
        dummy = torch.empty(*shape, device = self.device, dtype = self.dtype)
        self.cache = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]
        self.cache_2 = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]
        self.offset = 0

    @torch.no_grad()
    def get(self, layer_ind):
        k,v = self.cache[layer_ind]
        k_2,v_2 = self.cache_2[layer_ind]
        if self.noise_caches > 0.0:
            k = k + torch.randn_like(k) * self.noise_caches
            v = v + torch.randn_like(v) * self.noise_caches
            k_2 = k_2 + torch.randn_like(k_2) * self.noise_caches
            v_2 = v_2 + torch.randn_like(v_2) * self.noise_caches
        return (k,v), (k_2,v_2)
    
    @torch.no_grad()
    def update(self, new_k, new_v, new_k_2, new_v_2, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        self.cache[layer_ind] = (new_k,new_v)
        self.cache_2[layer_ind] = (new_k_2,new_v_2)

    @torch.no_grad()
    def truncate(self, truncate_amt, front = False):
        """
        Truncate/eject frames from the KV cache
        """
        truncate_amt_1 = self.config.sample_size**2
        truncate_amt_2 = self.config.tokens_per_frame - truncate_amt_1

        def tuple_truncate(k, v, truncate_amt):
            if front:
                k = k[:,:,:-truncate_amt]
                v = v[:,:,:-truncate_amt] 
            else:
                k = k[:,:,truncate_amt:]
                v = v[:,:,truncate_amt:]
            return k, v

        for i in range(self.config.n_layers):
            self.cache[i] = tuple_truncate(*self.cache[i], truncate_amt_1)
            self.cache_2[i] = tuple_truncate(*self.cache_2[i], truncate_amt_2)
    
    def length_at(self, idx):
        return self.cache[idx][0].shape[2]
        
    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0][0].shape[2]

    def n_frames(self):
        return len(self) // self.config.sample_size**2

    @property
    def shape(self):
        return self.cache[0][0].shape

    @property
    def shape_2(self):
        return self.cache_2[0][0].shape

