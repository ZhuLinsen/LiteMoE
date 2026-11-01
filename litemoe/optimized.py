"""Explicit EP1 performance policy; no runtime imports or patches from tools."""
from contextlib import contextmanager

import torch

from .batch import BatchPrefillEngine
from .packed import optimized_prefill_logits
from .cache import FusedKVCache
from .ops import Reference
from .vllm_moe import VllmFusedExperts


# Screened on A100 for M <= 32: same tokens as vLLM's default, 1-2% faster. L4 lost 4-5%, so it keeps the default.
A100_DECODE_TILES = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4, num_stages=3)


def decode_tiles(run):
    if isinstance(run.model.ep.backend, VllmFusedExperts) and run.batch <= 32 and "A100" in torch.cuda.get_device_name(run.device):
        return A100_DECODE_TILES
    return None


@contextmanager
def optimized_policy(run):
    model = run.model
    if run.world != 1:
        raise ValueError("Optimized execution currently requires EP1")
    limit = max((len(r.prompt) + r.budget - 1 for r in run.requests), default=1)
    if limit > run.capacity:
        raise ValueError("Optimized attention exceeds KV capacity")
    # Stopped rows have zero positions and masked KV writes, and their logits are never published.
    lean = run.lean_decode and run.flash_decode and run.device.type == "cuda" and isinstance(model.ep.backend, VllmFusedExperts) and all(
        w["logical"].numel() == 0 for w in model.w["layers"])
    key = (run.capacity, limit, id(model.ep.backend), model.decode_packed_qkv, run.fused_norm, run.flash_decode,
           getattr(model.ep.backend, 'fused_combine', False), lean)
    if run.graph is not None and getattr(run, "_optimized_graph_policy", None) != key:
        raise ValueError("Recreate Engine before changing captured execution policy")
    previous = (model.native_gqa, model.decode_kv_limit, model.decode_folded_q,
                model.decode_batched_linears, model.ep.prefill_backend, model.decode_fused_norm, model.decode_flash, model.decode_lean, model.prefill_lean)
    model.native_gqa, model.decode_kv_limit = False, limit
    model.decode_folded_q, model.decode_batched_linears = True, True
    model.ep.prefill_backend = run.prefill_reference
    model.decode_fused_norm = run.fused_norm
    model.decode_flash = run.flash_attention if run.flash_decode else None
    model.decode_lean = model.ep.decode_lean = lean
    model.prefill_lean = lean and run.lean_prefill
    try:
        yield
    finally:
        if run.graph is not None:
            run._optimized_graph_policy = key
        (model.native_gqa, model.decode_kv_limit, model.decode_folded_q,
         model.decode_batched_linears, model.ep.prefill_backend, model.decode_fused_norm, model.decode_flash, model.decode_lean, model.prefill_lean) = previous
        model.ep.decode_lean = model.decode_lean


class OptimizedEngine(BatchPrefillEngine):
    fused_norm = True
    flash_decode = True
    lean_decode = True
    lean_prefill = True
    cache_type = FusedKVCache
    prefill_function = staticmethod(optimized_prefill_logits)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_reference = self.model.ep.backend if isinstance(self.model.ep.backend, VllmFusedExperts) else Reference()
        from .flash_decode import FlashDecode
        cfg = self.model.cfg
        maximum = max((len(r.prompt) + r.budget - 1 for r in self.requests), default=1)
        paged = isinstance(self.model.ep.backend, VllmFusedExperts) and self.capacity % 256 == 0
        self.flash_attention = FlashDecode(self.batch, cfg.num_key_value_heads,
            cfg.num_attention_heads // cfg.num_key_value_heads, self.capacity, self.device, maximum if paged else None)

    def prepare_graph(self):
        tiles = decode_tiles(self) if self.device.type == "cuda" else None
        if tiles is None:
            with optimized_policy(self):
                return super().prepare_graph()
        from vllm.model_executor.layers.fused_moe import override_config
        with optimized_policy(self), override_config(tiles):  # The captured Graph keeps these tiles.
            return super().prepare_graph()

    def prefill(self):
        with optimized_policy(self):
            return super().prefill()

    def decode_step(self):
        with optimized_policy(self):
            return super().decode_step()
