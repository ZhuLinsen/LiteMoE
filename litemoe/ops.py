"""One HF-style reference and one DeepSpeed Triton grouped-GEMM adapter."""

import torch
import torch.nn.functional as F


def linear_rows(x, weight, singleton=False):
    """Decode retains the per-request GEMM shape used by the HF oracle."""
    if singleton and x.numel() // x.shape[-1] > 1:
        rows = x.reshape(-1, x.shape[-1]).split(1)
        return torch.cat([F.linear(row, weight) for row in rows]).view(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight)


def combine(out, ids, order, weighted, logical=None):
    """Physical storage must not change the logical BF16 reduction order."""
    if logical is not None:
        routes = weighted.new_zeros((ids.numel(), weighted.shape[-1]))
        routes.index_copy_(0, order, weighted)
        order = logical[ids].to(ids.dtype).flatten().argsort()
        weighted = routes[order]
    return out.index_add_(0, order // ids.shape[1], weighted)


def reference_experts(x, ids, scales, w1, w2, rank=0, world=1, out=None, logical=None, routes=False, singleton=False):
    """Return weighted local contributions; optional route output keeps BF16 order."""
    if out is None:
        out = x.new_zeros((*ids.shape, x.shape[-1])) if routes else torch.zeros_like(x)
    else:
        out.zero_()
    start = rank * w1.shape[0]
    # Match HF's grouped route ordering and single combine operation. Repeated
    # BF16 index_add calls round after every expert and are a different oracle.
    order = ids.flatten().argsort()
    token, branch = order // ids.shape[1], order % ids.shape[1]
    # One host transfer of group sizes replaces two CUDA boolean selections
    # (and their implicit synchronizations) per expert. Keep HF's row order.
    counts = torch.zeros(w1.shape[0] * world, dtype=torch.int32, device=x.device)
    counts.scatter_add_(0, ids.flatten().long(), torch.ones_like(order, dtype=torch.int32))
    sizes = counts.tolist()
    grouped = x[token].split(sizes)
    all_outputs = []
    for local, rows in enumerate(grouped[start:start + w1.shape[0]]):
        if rows.shape[0]:
            gate, up = linear_rows(rows, w1[local], singleton).chunk(2, -1)
            y = linear_rows(F.silu(gate) * up, w2[local], singleton)
            all_outputs.append(y)
    if all_outputs:
        begin, length = sum(sizes[:start]), sum(sizes[start:start + w1.shape[0]])
        selected, selected_branch = token[begin:begin + length], branch[begin:begin + length]
        weighted = torch.cat(all_outputs) * scales[selected, selected_branch, None].to(x.dtype)
        if routes:
            out.view(-1, x.shape[1]).index_copy_(0, order[begin:begin + length], weighted)
        else:
            combine(out, ids, order[begin:begin + length], weighted, logical)
    return out


class Reference:
    graph_safe = False
    name = "reference"

    def __call__(self, x, ids, scales, w1, w2, rank, world, out, logical=None):
        return reference_experts(x, ids, scales, w1, w2, rank, world, out, logical)

    def decode(self, x, ids, scales, w1, w2, rank, world, out, logical=None):
        return reference_experts(x, ids, scales, w1, w2, rank, world, out, logical, singleton=True)

    def routes(self, x, ids, scales, w1, w2, rank, world, out, singleton=False):
        # Preserve each source's expert GEMM batch shape as well as combine order.
        for a, i, s, o in zip(x.chunk(world), ids.chunk(world), scales.chunk(world), out.chunk(world)):
            # Prefill inputs are valid prefixes. All-zero scales mark transport padding.
            length = a.shape[0] if singleton else int(s.any(-1).sum().item())
            o.zero_()
            if length:
                reference_experts(a[:length], i[:length], s[:length], w1, w2, rank, world,
                                  o[:length], routes=True, singleton=singleton)
        return out

    def routes_decode(self, *args):
        return self.routes(*args, singleton=True)


class Cublas(Reference):
    """Reference Prefill and graph-safe, single-row cuBLAS Decode; NK pointers only."""
    graph_safe = True
    name = "cublas-single-row-decode"

    def __init__(self):
        import ctypes as C
        from pathlib import Path
        self.C = C
        torch.cuda.current_blas_handle()
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libcublas.so.' in line}
        if len(libraries) != 1:
            raise RuntimeError("Expected one loaded cuBLAS library")
        self.lib = C.CDLL(libraries.pop())
        self.gemm = self.lib.cublasGemmBatchedEx
        v, i = C.c_void_p, C.c_int
        self.gemm.argtypes = [v, i, i, i, i, i, v, v, i, i, v, i, i, v, v, i, i, i, i, i]
        self.gemm.restype = i
        self.alpha, self.beta = C.c_float(1), C.c_float(0)

    def linear(self, x, weights, selected):
        if x.dtype != torch.bfloat16 or weights.dtype != x.dtype or not weights.is_contiguous():
            raise ValueError("cuBLAS adapter requires contiguous BF16 expert weights")
        rows, hidden = x.shape
        outputs = weights.shape[1]
        result = torch.empty(rows, outputs, device=x.device, dtype=x.dtype)
        offsets = torch.arange(rows, device=x.device, dtype=torch.int64)
        ap = weights.data_ptr() + selected * weights.stride(0) * weights.element_size()
        bp = x.data_ptr() + offsets * hidden * x.element_size()
        cp = result.data_ptr() + offsets * outputs * result.element_size()
        for row in range(rows):
            status = self.gemm(torch.cuda.current_blas_handle(), 1, 0, outputs, 1, hidden,
                self.C.byref(self.alpha), ap.data_ptr() + row * 8, 14, hidden,
                bp.data_ptr() + row * 8, 14, hidden, self.C.byref(self.beta),
                cp.data_ptr() + row * 8, 14, outputs, 1, 68, 99)
            if status:
                raise RuntimeError(f"cublasGemmBatchedEx status={status}")
        return result

    def decode(self, x, ids, scales, w1, w2, rank, world, out, logical=None, routes=False):
        local = ids.flatten().long() - rank * w1.shape[0]
        order = local.argsort()
        local = local[order]
        owned = (local >= 0) & (local < w1.shape[0])
        selected = local.clamp(0, w1.shape[0] - 1)
        gate, up = self.linear(x[order // ids.shape[1]], w1, selected).chunk(2, -1)
        y = self.linear(F.silu(gate) * up, w2, selected)
        weighted = torch.where(owned[:, None], y, 0) * scales.flatten()[order, None].to(y.dtype)
        if routes:
            return out.zero_().view(-1, x.shape[-1]).index_copy_(0, order, weighted).view_as(out)
        return combine(out.zero_(), ids, order, weighted, logical)

    def routes_decode(self, *args):
        return self.decode(*args, routes=True)


class Grouped:
    """Fixed NK dispatch, GPU offsets and no host-dependent expert loop.

    Nonlocal routes sort into an ignored tail. Only NK rows are allocated;
    there is no NE padding and no replication of expert weights per token.
    Operator temporaries use PyTorch's CUDA Graph private memory pool.
    """

    graph_safe = True
    name = "deepspeed-triton-grouped-gemm"
    fused_combine = False

    def __init__(self, capacity, hidden, intermediate, experts, topk, dtype, device):
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("The grouped backend requires SM80+; run G0")
        from deepspeed.ops.triton_ops import group_gemm_triton
        self.mm = group_gemm_triton
        if dtype != torch.bfloat16:
            raise ValueError("The admitted grouped backend is BF16 only")
        self.capacity = capacity
        self.workspace = torch.empty(0, dtype=torch.uint8, device=device)

    def __call__(self, x, ids, scales, w1, w2, rank, world, out, logical=None):
        if x.shape[0] > self.capacity:
            raise ValueError("Grouped dispatch capacity exceeded")
        count, topk = w1.shape[0], ids.shape[1]
        local = ids.flatten().long() - rank * count
        local = torch.where((local >= 0) & (local < count), local, count)
        order = local.argsort()
        selected = local[order]
        counts = torch.zeros(count + 1, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, selected, torch.ones_like(selected, dtype=torch.int32))
        offsets = counts[:-1].cumsum(0, dtype=torch.int32)
        token = order // topk
        gate, up = self.mm(x[token], w1, offsets, trans_b=True).chunk(2, -1)
        y = self.mm(F.silu(gate) * up, w2, offsets, trans_b=True)
        if self.fused_combine:
            from .moe_finish import finish
            return finish(y, ids, order, scales, out, rank, count, logical)
        # Skipped rows are uninitialized; multiplying NaN by zero is insufficient.
        y = torch.where((selected < count)[:, None], y, 0)
        weighted = y * scales.flatten()[order, None].to(y.dtype)
        return combine(out.zero_(), ids, order, weighted, logical)
