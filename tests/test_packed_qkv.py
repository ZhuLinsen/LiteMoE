import pytest
import torch

from litemoe.engine import Request
from litemoe.optimized import OptimizedEngine, optimized_policy


@torch.inference_mode()
def test_packed_decode_preserves_mixed_cache_and_mask(tiny):
    _, model, _ = tiny
    requests = [Request('a', [2, 3, 5, 7], 5), Request('b', [11], 2)]
    control = OptimizedEngine(model, requests, capacity=16, ignore_eos=True)
    expected, _ = control.generate()
    model.prepare_packed_qkv()
    pointers = [w['self_attn.qkv_proj'].data_ptr() for w in model.w['layers']]
    model.prepare_packed_qkv()
    assert pointers == [w['self_attn.qkv_proj'].data_ptr() for w in model.w['layers']]
    model.decode_packed_qkv = True
    run = OptimizedEngine(model, requests, capacity=16, ignore_eos=True)
    actual, _ = run.generate()
    assert actual == expected
    torch.testing.assert_close(run.cache.k, control.cache.k, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(run.cache.v, control.cache.v, atol=2e-6, rtol=2e-5)
    # Captured graphs must never silently retain the opposite projection policy.
    run.graph = object()
    run._optimized_graph_policy = (run.capacity, 8, id(model.ep.backend), True)
    model.decode_packed_qkv = False
    with pytest.raises(ValueError, match='Recreate'):
        with optimized_policy(run):
            pass
