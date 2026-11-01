"""Fixed-order AG -> sharded experts -> RS; explicit partial or route return."""

import torch
import torch.distributed as dist


class ExpertParallel:
    def __init__(self, world, rank, capacity, hidden, topk, dtype, device, backend, combine="partial"):
        if combine not in ("partial", "routes"):
            raise ValueError("Unknown EP combine policy")
        if combine == "routes" and not hasattr(backend, "routes"):
            raise ValueError("Route return requires a backend with source-sized route outputs")
        self.world, self.rank, self.backend = world, rank, backend
        self.prefill_backend = None
        self.decode_lean = False
        self.combine = combine if world > 1 else "local"
        self.buffers = {}
        self.returned_routes = {}
        self.capacity, self.hidden, self.topk = capacity, hidden, topk
        self.dtype, self.device = dtype, device

    def prepare(self, tokens):
        if tokens > self.capacity:
            raise ValueError("Communication capacity exceeded")
        if tokens not in self.buffers:
            n = tokens * self.world
            shape = (n, self.topk, self.hidden) if self.combine == "routes" else (n, self.hidden)
            self.buffers[tokens] = (
                torch.empty(n, self.hidden, dtype=self.dtype, device=self.device),
                torch.empty(n, self.topk, dtype=torch.int32, device=self.device),
                torch.empty(n, self.topk, dtype=torch.float32, device=self.device),
                torch.empty(*shape, dtype=self.dtype, device=self.device),
                torch.empty(tokens, self.hidden, dtype=self.dtype, device=self.device),
            )
            if self.combine == "routes":
                self.returned_routes[tokens] = torch.empty(tokens, self.topk, self.hidden, dtype=self.dtype, device=self.device)

    def __call__(self, x, ids, weights, valid, physical, w1, w2, logical=None, decode=False, tokens=None):
        if decode and self.decode_lean:  # EP1 identity placement; zero scales are not needed for stopped rows.
            return self.backend.operation(x.contiguous(), w1, w2, weights.float().contiguous(), ids.int().contiguous(),
                                          apply_router_weight_on_input=False, global_num_experts=w1.shape[0])
        local_tokens = x.shape[0]
        t = tokens or local_tokens
        X, PID, A, partial, out = self.buffers[t]
        x = torch.where(valid[:, None], x, 0).contiguous()
        logical_ids = ids
        ids = physical[ids].to(torch.int32).contiguous()
        scales = (weights.float() * valid[:, None]).contiguous()
        options = {"logical": logical} if logical is not None and logical.numel() else {}
        backend = self.backend if decode or self.prefill_backend is None else self.prefill_backend
        local_op = getattr(backend, "decode", backend) if decode else backend
        if self.world == 1:
            local_op(x, ids, scales, w1, w2, 0, 1, out, **options)
        else:
            # Pad transport only: padding must not alter local Prefill GEMM shapes.
            pad = t - local_tokens
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            ids = torch.nn.functional.pad(ids, (0, 0, 0, pad))
            scales = torch.nn.functional.pad(scales, (0, 0, 0, pad))
            dist.all_gather_into_tensor(X, x)
            dist.all_gather_into_tensor(PID, ids)
            dist.all_gather_into_tensor(A, scales)
            if self.combine == "routes":
                route_op = getattr(backend, "routes_decode", backend.routes) if decode else backend.routes
                route_op(X, PID, A, w1, w2, self.rank, self.world, partial)
                returned = self.returned_routes[t]
                dist.reduce_scatter_tensor(returned, partial, op=dist.ReduceOp.SUM)
                order = logical_ids.to(torch.int32).flatten().argsort()
                out.zero_().index_add_(0, order // self.topk, returned[:local_tokens].flatten(0, 1)[order])
            else:
                local_op(X, PID, A, w1, w2, self.rank, self.world, partial, **options)
                dist.reduce_scatter_tensor(out, partial, op=dist.ReduceOp.SUM)
        return out[:local_tokens]


def agree(value, device):
    """Collectively detect inconsistent plans before model collectives."""
    if not dist.is_initialized():
        return
    import hashlib
    import json
    raw = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
    local = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    if any(not torch.equal(local, other) for other in gathered):
        raise ValueError("Ranks disagree on input/config/placement plan")
