from types import SimpleNamespace
import pytest
import torch
from litemoe.cache import FusedKVCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('capacity', [8, 4096])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_fused_write_strides_inactive_and_graph(capacity, dtype):
    cfg = SimpleNamespace(num_key_value_heads=3, hidden_size=21,
                          num_attention_heads=3, num_hidden_layers=2)
    cache = FusedKVCache(cfg, 3, capacity, 'cuda', dtype)
    cache.k.normal_(); cache.v.normal_()
    # Noncontiguous channels and batch strides, as in a packed projection view.
    k = torch.randn(3, 3, 1, 28, device='cuda', dtype=dtype)[..., ::4]
    v = torch.randn(3, 3, 1, 42, device='cuda', dtype=dtype)[..., 1::6]
    pos = torch.tensor([0, capacity+10, capacity-1], device='cuda')
    active = torch.tensor([True, False, True], device='cuda')
    k[1].fill_(float('nan')); v[1].fill_(float('nan'))
    expected_k, expected_v = cache.k.clone(), cache.v.clone()
    for slot, p in ((0, 0), (2, capacity-1)):
        expected_k[1, slot, :, p] = k[slot, :, 0]
        expected_v[1, slot, :, p] = v[slot, :, 0]
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        cache.write(1, k, v, pos, active)
        torch.cuda.synchronize()
        assert torch.equal(cache.k, expected_k) and torch.equal(cache.v, expected_v)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache.write(1, k, v, pos, active)
        pos.copy_(torch.tensor([1, 2, capacity+99], device='cuda'))
        active.copy_(torch.tensor([True, True, False], device='cuda'))
        k.normal_(); v.normal_()
        for slot, p in ((0, 1), (1, 2)):
            expected_k[1, slot, :, p] = k[slot, :, 0]
            expected_v[1, slot, :, p] = v[slot, :, 0]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cache.k, expected_k) and torch.equal(cache.v, expected_v)
        assert torch.are_deterministic_algorithms_enabled()
        graph.reset()
    finally:
        torch.use_deterministic_algorithms(previous)
