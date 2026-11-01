"""Optional Decode RMSNorm: retain the reference's rounding before weight multiply."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    weight = tl.load(W + col, col < H, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    # The BF16 cast is a numerical operation, not merely the final storage dtype.
    normalized = (x * scale).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + col, normalized * weight, col < H)


def rmsnorm(x, weight, eps):
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('Fused RMSNorm requires contiguous CUDA tensors')
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError('Fused RMSNorm weight must match the last dimension, dtype and device')
    out = torch.empty_like(x)
    width = x.shape[-1]
    if x.numel():
        _rms[(x.numel() // width,)](x, weight, out, width, eps,
            BLOCK=triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False)
    return out
