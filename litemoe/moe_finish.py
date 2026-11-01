"""Weighted route reduction enabled by the explicit optimized EP1 entrypoint."""
import torch
import triton
import triton.language as tl


@triton.jit
def _finish(Y, IDS, INV, SCALE, LOGICAL, OUT, H: tl.constexpr, K: tl.constexpr,
            RANK: tl.constexpr, COUNT: tl.constexpr, REMAP: tl.constexpr,
            BK: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    branch = tl.arange(0, BK)
    physical = tl.load(IDS + token * K + branch, branch < K, other=0)
    logical = physical
    if REMAP:
        logical = tl.load(LOGICAL + physical, branch < K, other=0)
    keys = tl.where(branch < K, logical * K + branch, 2147483647)
    sorted_keys = tl.sort(keys, descending=False)
    total = tl.full((BH,), 0, tl.float32)
    for j in range(K):
        selected = tl.sum(tl.where(branch == j, sorted_keys % K, 0), 0)
        original = token * K + selected
        row = tl.load(INV + original)
        expert = tl.load(IDS + original) - RANK * COUNT
        owned = (expert >= 0) & (expert < COUNT)
        value = tl.load(Y + row * H + col, (col < H) & owned, other=0).to(tl.float32)
        scale = tl.load(SCALE + original).to(Y.dtype.element_ty).to(tl.float32)
        weighted = (value * scale).to(Y.dtype.element_ty).to(tl.float32)
        total += weighted
    tl.store(OUT + token * H + col, total, col < H)


def finish(y, ids, order, scales, out, rank, count, logical=None):
    """FP32 ordered accumulation of rounded BF16 route contributions.

    Nonlocal rows may be uninitialized and must never be read. This candidate
    intentionally has its own numerical screen; it is not a bitwise HF claim.
    """
    tensors = (y, ids, order, scales, out)
    if any(not t.is_cuda or not t.is_contiguous() or t.device != y.device for t in tensors):
        raise ValueError('Fused combine requires contiguous tensors on one CUDA device')
    if ids.ndim != 2 or y.ndim != 2 or scales.shape != ids.shape:
        raise ValueError('Invalid route tensor shapes')
    batch, topk = ids.shape
    if y.shape[0] != ids.numel() or order.shape != (ids.numel(),) or out.shape != (batch, y.shape[1]):
        raise ValueError('Invalid route/output sizes')
    if y.dtype != torch.bfloat16 or out.dtype != y.dtype or not 0 < topk <= 8 or count <= 0:
        raise ValueError('Fused combine supports BF16 and top-k 1 through 8')
    remap = logical is not None and logical.numel() != 0
    if remap and (not logical.is_cuda or logical.device != y.device or not logical.is_contiguous()):
        raise ValueError('Logical map must be contiguous on the route device')
    inverse = order.argsort()
    _finish[(batch, triton.cdiv(y.shape[1], 128))](y, ids, inverse, scales,
        logical if remap else ids, out, y.shape[1], topk, rank, count, remap,
        triton.next_power_of_2(topk), 128, num_warps=4, enable_fp_fusion=False)
    return out
