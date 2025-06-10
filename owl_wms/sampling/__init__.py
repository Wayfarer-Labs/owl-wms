from .simple import SimpleSampler
from .cfg import CFGSampler
from .window import WindowCFGSampler

def get_sampler_cls(sampler_id):
    if sampler_id == "simple":
        return SimpleSampler
    elif sampler_id == "cfg":
        return CFGSampler
    elif sampler_id == "window":
        return WindowCFGSampler
    elif sampler_id == "shortcut":
        from .shortcut_sampler import CacheShortcutSampler
        return CacheShortcutSampler