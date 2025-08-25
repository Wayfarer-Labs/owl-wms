import torch
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
        self.cache = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        if self.noise_caches > 0.0:
            k = k + torch.randn_like(k) * self.noise_caches
            v = v + torch.randn_like(v) * self.noise_caches
        return k,v

    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"

        old_len = self.length_at(layer_ind)
        new_len = new_k.shape[2]
        delta_len = new_len - old_len
        self.offsets[layer_ind] += delta_len

        self.cache[layer_ind] = (new_k,new_v)

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

class InferenceKVCache:
    """
    Optimized for inference with static shapes and in-place updates
    """
    def __init__(self, kv_cache : SingleKVCache):
        self.cache = kv_cache.cache
        self.offsets = kv_cache.offsets

        self.tokens_per_frame = kv_cache.config.tokens_per_frame
        self.n_layers = kv_cache.config.n_layers
        self.should_update = False

    def enable_cache_updates(self):
        self.should_update = True
    
    def disable_cache_updates(self):
        self.should_update = False
    
    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def get(self, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        return k,v

    def update(self, new_k, new_v, layer_ind):
        # Assumed that both are tokens_per_frame
        self.cache[layer_ind][0][:,:,:-self.tokens_per_frame] = self.cache[layer_ind][0][:,:,self.tokens_per_frame:]
        self.cache[layer_ind][1][:,:,:-self.tokens_per_frame] = self.cache[layer_ind][1][:,:,self.tokens_per_frame:]

        self.cache[layer_ind][0][:,:,-self.tokens_per_frame:] = new_k
        self.cache[layer_ind][1][:,:,-self.tokens_per_frame:] = new_v

        self.offsets[layer_ind] += self.tokens_per_frame

    def _update(self, new_k, new_v, layer_ind):
        self.cache[layer_ind][0] = torch.cat([self.cache[layer_ind][0][:,:,self.tokens_per_frame:], new_k], dim=2)
        self.cache[layer_ind][1] = torch.cat([self.cache[layer_ind][1][:,:,self.tokens_per_frame:], new_v], dim=2)
        self.offsets[layer_ind] += self.tokens_per_frame

    def truncate(self, *args, **kwargs):
        pass

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
