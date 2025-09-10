import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


@triton.jit
def _rotate_dequant_kernel(
    in_ptr,          # *int8   [BH, T, D]
    scale_ptr,       # *bf16   [BH, T]
    out_ptr,         # *bf16   [BH, T, D]
    BH: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    offset_t: tl.int32,        # token offset to rotate by (oldest index)
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_tb = tl.program_id(1)
    pid_db = tl.program_id(2)

    t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offsets = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_t = t_offsets < T
    mask_d = d_offsets < D

    # Compute source t (rotate by token offset)
    src_t = (t_offsets + offset_t) % T

    # Flattened base for BH plane
    base_td = pid_bh * T * D
    base_t = pid_bh * T

    # Broadcast to 2D tile
    src_t_2d = src_t[:, None]
    t_mask_2d = mask_t[:, None] & mask_d[None, :]

    in_idx = base_td + src_t_2d * D + d_offsets[None, :]
    sc_idx = base_t + src_t
    out_idx = base_td + t_offsets[:, None] * D + d_offsets[None, :]

    # Loads
    q_i8 = tl.load(in_ptr + in_idx, mask=t_mask_2d, other=0).to(tl.float32)
    sc = tl.load(scale_ptr + sc_idx, mask=mask_t, other=0.0).to(tl.float32)
    sc = sc[:, None]  # [BLOCK_T,1]

    # Dequant and store
    x = (q_i8 * sc).to(tl.bfloat16)
    tl.store(out_ptr + out_idx, x, mask=t_mask_2d)


def fused_rotate_dequant_i8_to_bf16(inp_i8: torch.Tensor,
                                    scale_bf16: torch.Tensor,
                                    offset_t: int,
                                    out_bf16: torch.Tensor | None = None,
                                    block_t: int = 64,
                                    block_d: int = 64) -> torch.Tensor:
    """
    Fused blockwise-rotate (by phase*block_len) + per-token dequant.
    Args:
      inp_i8: [B,H,T,D] int8
      scale_bf16: [B,H,T] bfloat16
      phase: rotation phase (blocks)
      block_len: tokens_per_frame
      out_bf16: optional preallocated [B,H,T,D] bfloat16
    Returns:
      out_bf16: [B,H,T,D] bfloat16 in oldest->newest order
    """
    assert _HAS_TRITON, "Triton is required for fused_rotate_dequant_i8_to_bf16"
    assert inp_i8.is_cuda and scale_bf16.is_cuda
    B, H, T, D = inp_i8.shape
    BH = B * H
    inp_i8_f = inp_i8.reshape(BH, T, D)
    scale_f = scale_bf16.reshape(BH, T)
    if out_bf16 is None:
        out_bf16 = torch.empty_like(inp_i8, dtype=torch.bfloat16)
    out_f = out_bf16.reshape(BH, T, D)
    grid = (BH, triton.cdiv(T, block_t), triton.cdiv(D, block_d))
    _rotate_dequant_kernel[grid](
        inp_i8_f,
        scale_f,
        out_f,
        BH=BH,
        T=T,
        D=D,
        offset_t=int(offset_t),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    return out_bf16


