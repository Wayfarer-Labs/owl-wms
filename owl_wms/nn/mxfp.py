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
    late_layers = int(_get_env_flag("OWL_MXFP_LATE_LAYERS", "0"))  # 0 means all

    if bits not in (4, 8):
        bits = 8

    kernel_choice = MXGemmKernelChoice.CUTLASS if kernel == "cutlass" else MXGemmKernelChoice.TRITON
    cfg = MXFPInferenceConfig(gemm_kernel_choice=kernel_choice, bits=bits)

    # Discover total transformer block count from module names (best-effort)
    total_layers = -1
    for name, _ in model.named_modules():
        if ".blocks." in name:
            try:
                idx = int(name.split(".blocks.")[1].split(".")[0])
                if idx > total_layers:
                    total_layers = idx
            except Exception:
                pass
    total_layers += 1 if total_layers >= 0 else 0

    def layer_idx_from_name(name: str) -> int | None:
        if ".blocks." in name:
            try:
                return int(name.split(".blocks.")[1].split(".")[0])
            except Exception:
                return None
        return None

    def passes_layer_gate(name: str) -> bool:
        if late_layers <= 0 or total_layers <= 0:
            return True
        li = layer_idx_from_name(name)
        if li is None:
            # If we cannot determine layer index, allow only when not clearly under blocks
            return False
        return li >= (total_layers - late_layers)

    def should_transform(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        name_l = name.lower()
        # Extra caution for MXFP4: avoid final output projections by default
        if bits == 4:
            if ("final" in name_l and "proj" in name_l) or ("skip_projs" in name_l):
                return False
        if scope == "mlp":
            if ("mlp" in name_l) or ("fc1" in name_l) or ("fc2" in name_l):
                if ("qkv" in name_l) or ("to_q" in name_l) or ("to_k" in name_l) or ("to_v" in name_l) or ("out" in name_l):
                    return False
                return passes_layer_gate(name)
            return False
        if scope == "attn":
            # qkv packed or separate projections, and output proj
            return (("qkv" in name_l) or ("to_q" in name_l) or ("to_k" in name_l) or ("to_v" in name_l) or ("out" in name_l)) and passes_layer_gate(name)
        # scope == "all"
        return passes_layer_gate(name)

    transformed = 0
    transformed_names = []
    category_counts = {"mlp": 0, "attn": 0, "other": 0}
    for name, mod in model.named_modules():
        if should_transform(name, mod):
            _mx_inference_linear_transform(mod, cfg)
            transformed += 1
            if list_names:
                transformed_names.append(name)
            nl = name.lower()
            if ("mlp" in nl) or ("fc1" in nl) or ("fc2" in nl):
                category_counts["mlp"] += 1
            elif ("qkv" in nl) or ("to_q" in nl) or ("to_k" in nl) or ("to_v" in nl) or ("out" in nl):
                category_counts["attn"] += 1
            else:
                category_counts["other"] += 1

    if list_names and transformed_names:
        print(f"[MXFP] Transformed {transformed} linears (bits={bits}, kernel={kernel}, scope={scope}, late_layers={late_layers}):")
        print(f"[MXFP]  counts: mlp={category_counts['mlp']}, attn={category_counts['attn']}, other={category_counts['other']}")
        for n in transformed_names[:20]:
            print(f"[MXFP]  - {n}")
        if len(transformed_names) > 20:
            print(f"[MXFP]  ... and {len(transformed_names)-20} more")

    return transformed


def apply_mx_transforms_mlp_only(model: nn.Module) -> int:
    return apply_mx_transforms(model, scope="mlp")


