"""Default forward dispatch with actual CUDA router and expert computation."""
import importlib.util
import pytest
import torch

from litemoe.engine import Engine, Request
from litemoe.ep import ExpertParallel
from litemoe.model import Granite
from litemoe.ops import Reference
from litemoe.optimized import OptimizedEngine
from litemoe.placement import uniform
from litemoe.vllm_moe import VllmFusedExperts
from litemoe.weights import load_weights


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('vllm') is None,
                    reason='Pinned vLLM CUDA environment required')
@pytest.mark.parametrize('optimized,backend,flash', [
    (True, 'vllm', True), (True, 'vllm', False),
    (False, 'vllm', True), (True, 'reference', True), (False, 'reference', True)])
@torch.inference_mode()
def test_default_router_dispatch_and_weight_dtype(tiny, monkeypatch, optimized, backend, flash):
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router
    _, mini, source = tiny
    cfg, capacity = mini.cfg, 16
    weights = load_weights(cfg, source, uniform(cfg.num_hidden_layers, cfg.num_local_experts, 1),
                           0, 'cuda', torch.bfloat16)
    op = VllmFusedExperts() if backend == 'vllm' else Reference()
    ep = ExpertParallel(1, 0, capacity, cfg.hidden_size, cfg.num_experts_per_tok, torch.bfloat16, 'cuda', op)
    model = Granite(cfg, weights, ep, capacity)
    cls = OptimizedEngine if optimized else Engine
    run = cls(model, [Request('active', [2, 3], 4), Request('one', [7], 1)],
              capacity=capacity, ignore_eos=True)
    if optimized:
        run.flash_decode = flash
        run.lean_prefill = False  # This test pins the masked Prefill contract; see test_lean_decode.py.
    native_calls, expert_weights = [], []
    native, expert = fused_topk_router.fused_topk, ExpertParallel.__call__

    def observe_native(hidden, logits, topk, **kwargs):
        assert hidden.dtype == logits.dtype == torch.bfloat16
        assert kwargs == {'renormalize': True}
        native_calls.append(tuple(logits.shape))
        return native(hidden, logits, topk, **kwargs)

    def observe_expert(self, hidden, ids, scales, *args, **kwargs):
        if self is ep:
            expert_weights.append((ids.dtype, scales.dtype, kwargs.get('decode')))
        return expert(self, hidden, ids, scales, *args, **kwargs)

    monkeypatch.setattr(fused_topk_router, 'fused_topk', observe_native)
    monkeypatch.setattr(ExpertParallel, '__call__', observe_expert)
    run.prefill(); run.synchronize()
    assert not native_calls
    for _ in range(2):
        run.decode_step(); run.synchronize()
    expected = 2*cfg.num_hidden_layers if optimized and backend == 'vllm' else 0
    assert len(native_calls) == expected
    decode_weights = [row for row in expert_weights if row[2]]
    assert len(decode_weights) == 2*cfg.num_hidden_layers
    lean = optimized and backend == 'vllm' and flash  # Lean Decode passes FP32 scales directly.
    assert all(dtype == (torch.float32 if lean and decode else torch.bfloat16) for _, dtype, decode in expert_weights)
    if expected:
        assert all(ids == torch.int32 for ids, _, _ in decode_weights)
    assert not model.decode_batched_linears and model.decode_flash is None
