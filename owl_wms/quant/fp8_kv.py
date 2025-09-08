import torch


@torch.no_grad()
def ema_update(prev: torch.Tensor, new: torch.Tensor, momentum: float = 0.99) -> torch.Tensor:
    """Exponentially-weighted moving average update for amax tracking."""
    return momentum * prev + (1.0 - momentum) * new


@torch.no_grad()
def quantize_per_head(
    x_bf16: torch.Tensor,
    amax_prev: torch.Tensor,
    fmt: str = "e4m3",
    momentum: float = 0.99,
):
    """
    Quantize per-head using symmetric int8 + scale as an FP8-like proxy.

    Args:
        x_bf16: [B,H,T,D] or [H,T,D] BF16 tensor
        amax_prev: [B,H] or [H] previous amax EMA
        fmt: 'e4m3' or 'e5m2' (currently just influences potential future scaling heuristics)
        momentum: EMA momentum

    Returns:
        q_int8: same layout as x_bf16 but int8
        scale: [B,H] or [H] BF16 scale per head
        amax: updated EMA amax per head (same shape as scale)
    """
    if x_bf16.dim() == 4:
        # [B,H,T,D]
        reduce_dims = (2, 3)
    elif x_bf16.dim() == 3:
        # [H,T,D]
        reduce_dims = (1, 2)
    else:
        raise ValueError("x_bf16 must be [B,H,T,D] or [H,T,D]")

    amax_now = x_bf16.abs().amax(dim=reduce_dims)
    amax = ema_update(amax_prev, amax_now, momentum)

    # Symmetric int8 range approximation for FP8-like storage
    denom = 127.0
    scale = (amax / denom).clamp_min(1e-8)

    if x_bf16.dim() == 4:
        inv = (1.0 / scale).to(x_bf16.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        q = (x_bf16 * inv).round().clamp_(-128, 127).to(torch.int8)
    else:
        inv = (1.0 / scale).to(x_bf16.dtype).unsqueeze(-1).unsqueeze(-1)  # [H,1,1]
        q = (x_bf16 * inv).round().clamp_(-128, 127).to(torch.int8)

    return q, scale.to(torch.bfloat16), amax.to(torch.bfloat16)


@torch.no_grad()
def dequantize_per_head(q_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize per-head int8 + scale back to BF16.

    Args:
        q_i8: [B,H,T,D] or [H,T,D] int8
        scale: [B,H] or [H]

    Returns:
        x_bf16: BF16 tensor with same layout as q_i8
    """
    if q_i8.dim() == 4:
        mul = scale.to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        x = (q_i8.float() * mul).to(torch.bfloat16)
    elif q_i8.dim() == 3:
        mul = scale.to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [H,1,1]
        x = (q_i8.float() * mul).to(torch.bfloat16)
    else:
        raise ValueError("q_i8 must be [B,H,T,D] or [H,T,D]")
    return x


@torch.no_grad()
def quantize_per_head_timewise(
    x_bf16: torch.Tensor,
    fmt: str = "e4m3",
    bits: int = 8,
):
    """
    Per-head, per-token quantization: scale computed per [B,H,T] by amax over D.
    No EMA smoothing (scales are written alongside tokens in the ring buffer).

    Args:
        x_bf16: [B,H,T,D]
        fmt: 'e4m3' or 'e5m2'

    Returns:
        q_int8: [B,H,T,D]
        scale: [B,H,T] BF16
    """
    assert x_bf16.dim() == 4, "Expected [B,H,T,D]"
    amax = x_bf16.abs().amax(dim=3)  # [B,H,T]
    # Determine quantization range from bits
    qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit
    qmin = - (1 << (bits - 1))    # -128 for 8-bit, -8 for 4-bit
    denom = float(qmax)
    scale = (amax / denom).clamp_min(1e-8).to(torch.bfloat16)  # [B,H,T]
    inv = (1.0 / scale).to(x_bf16.dtype).unsqueeze(-1)  # [B,H,T,1]
    q = (x_bf16 * inv).round().clamp_(qmin, qmax).to(torch.int8)
    return q, scale


@torch.no_grad()
def dequantize_per_head_timewise(q_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantization matching quantize_per_head_timewise.

    Args:
        q_i8: [B,H,T,D]
        scale: [B,H,T]

    Returns:
        x_bf16: [B,H,T,D]
    """
    mul = scale.to(torch.float32).unsqueeze(-1)  # [B,H,T,1]
    return (q_i8.float() * mul).to(torch.bfloat16)


