import os
import torch
from torch import nn

try:
    from torchao.prototype.mx_formats.inference_workflow import (
        MXFPInferenceConfig,
        _mx_inference_linear_transform,
        MXGemmKernelChoice,
    )
    _MX_AVAILABLE = True
except Exception:
    _MX_AVAILABLE = False


def _get_env_flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


def apply_mx_transforms(model: nn.Module, scope: str = "mlp") -> int:
    """
    Apply MXFP transforms to MLP Linear layers only.

    Controlled via env vars:
      - OWL_MXFP_ENABLE=0/1
      - OWL_MXFP_BITS=8|4 (default 8)
      - OWL_MXFP_KERNEL=cutlass|triton (default cutlass)
      - OWL_MXFP_LIST=0/1 (print transformed module names)

    Returns number of transformed modules.
    """
    if not _MX_AVAILABLE:
        return 0

    enable = bool(int(_get_env_flag("OWL_MXFP_ENABLE", "0")))
    if not enable:
        return 0

    bits = int(_get_env_flag("OWL_MXFP_BITS", "8"))
    kernel = _get_env_flag("OWL_MXFP_KERNEL", "cutlass").lower()
    list_names = bool(int(_get_env_flag("OWL_MXFP_LIST", "0")))
    scope = _get_env_flag("OWL_MXFP_SCOPE", scope).lower()

    if bits not in (4, 8):
        bits = 8

    kernel_choice = MXGemmKernelChoice.CUTLASS if kernel == "cutlass" else MXGemmKernelChoice.TRITON
    cfg = MXFPInferenceConfig(gemm_kernel_choice=kernel_choice, bits=bits)

    def should_transform(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        name_l = name.lower()
        if scope == "mlp":
            if ("mlp" in name_l) or ("fc1" in name_l) or ("fc2" in name_l):
                if ("qkv" in name_l) or ("to_q" in name_l) or ("to_k" in name_l) or ("to_v" in name_l) or ("out" in name_l):
                    return False
                return True
            return False
        if scope == "attn":
            # qkv packed or separate projections, and output proj
            return ("qkv" in name_l) or ("to_q" in name_l) or ("to_k" in name_l) or ("to_v" in name_l) or ("out" in name_l)
        # scope == "all"
        return True

    transformed = 0
    transformed_names = []
    for name, mod in model.named_modules():
        if should_transform(name, mod):
            _mx_inference_linear_transform(mod, cfg)
            transformed += 1
            if list_names:
                transformed_names.append(name)

    if list_names and transformed_names:
        print(f"[MXFP] Transformed {transformed} MLP linears (bits={bits}, kernel={kernel}):")
        for n in transformed_names[:20]:
            print(f"[MXFP]  - {n}")
        if len(transformed_names) > 20:
            print(f"[MXFP]  ... and {len(transformed_names)-20} more")

    return transformed


def apply_mx_transforms_mlp_only(model: nn.Module) -> int:
    return apply_mx_transforms(model, scope="mlp")


