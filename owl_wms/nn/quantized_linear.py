from typing import Tuple
import torch
from torch import Tensor, nn


@torch.library.custom_op("owl::mm", mutates_args=())
@torch.compile
def mm_op(
        x: Tensor, w: Tensor, bias: Tensor | None, x_s: Tensor, w_s: Tensor, grad_s: Tensor, out_dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    assert x.is_contiguous() and w.is_contiguous()
    x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
    w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        x_f8,
        w_f8.T,
        bias=bias,
        scale_a=x_s,
        scale_b=w_s,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )
    return out, x_f8, w_f8


@torch.library.custom_op("owl::mm_backward", mutates_args=())
@torch.compile
def mm_backward_op(
        g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: Tensor, w_s: Tensor, grad_s: Tensor
) -> Tuple[Tensor, Tensor]:
    assert g.is_contiguous()

    grad_f8 = g.div(grad_s).to(torch.float8_e5m2)
    grad_x = torch._scaled_mm(
        grad_f8,
        w_f8.T.contiguous().T,
        out_dtype=g.dtype,
        scale_a=grad_s,
        scale_b=w_s,
        use_fast_accum=False,
    )
    grad_w = torch._scaled_mm(
        x_f8.T.contiguous(),
        grad_f8.T.contiguous().T,
        out_dtype=g.dtype,
        scale_a=x_s,
        scale_b=grad_s,
        use_fast_accum=False,
    ).T

    return grad_x, grad_w


@mm_op.register_fake
def _(x: Tensor, w: Tensor, bias: Tensor | None,
      x_s: Tensor, w_s: Tensor, grad_s: Tensor, out_dtype: torch.dtype):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    y = (x @ w.T).to(out_dtype)
    if bias is not None:
        y = y + bias.to(out_dtype)
    return y, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.bfloat16)


def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.owl.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    grad_bias = grad_out.sum(dim=list(range(grad_out.dim() - 1)))
    return grad_x, grad_w, grad_bias, None, None, None, None


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s, out_dtype = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


class QLinear(nn.Linear):
    """
    Based on modded-nanogpt's CastedLinear and Micikevicius et al

    original dtype for weights/bias/grad/activation
    FP8 for matmul

    "FP8 Formats for Deep Learning" (https://arxiv.org/pdf/2209.05433) states
    - E4M3 max norm: 448
    """
    def __init__(self, in_features: int, out_features: int, bias=True, out_dtype: torch.dtype | None = torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        self.x_s = nn.Buffer(torch.tensor(in_features**0.5 / 448), persistent=False)
        self.w_s = nn.Buffer(torch.tensor(in_features**0.5 / 448), persistent=False)
        self.grad_s = nn.Buffer(torch.tensor(1 / 448), persistent=False)
        self.out_dtype = out_dtype

    def forward(self, x: Tensor):
        orig_shape = x.shape
        x = x.to(self.out_dtype)
        bias = self.bias.to(self.out_dtype) if self.bias is not None else None
        x = torch.ops.owl.mm(
            x.flatten(0, -2),
            self.weight,
            bias=bias,
            x_s=self.x_s.float(),
            w_s=self.w_s.float(),
            grad_s=self.grad_s.float(),
            out_dtype=self.out_dtype
        )[0]
        return x.view(*orig_shape[:-1], -1)
