from types import SimpleNamespace
import pytest
import torch
from litemoe.cache import KVCache


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('capacity', [8, 4096])
@pytest.mark.parametrize('deterministic', [True, False])
def test_scatter_preserves_inactive_and_untouched_cache(device, capacity, deterministic):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        cfg = SimpleNamespace(num_key_value_heads=2, hidden_size=16,
                              num_attention_heads=4, num_hidden_layers=2)
        cache = KVCache(cfg, 3, capacity, device, torch.bfloat16)
        cache.k.normal_(); cache.v.normal_()
        expected_k, expected_v = cache.k.clone(), cache.v.clone()
        k = torch.randn(3, 2, 1, 4, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k[1].fill_(float('nan')); v[1].fill_(float('nan'))
        lengths = torch.tensor([0, capacity + 10, capacity-1], device=device)
        active = torch.tensor([True, False, True], device=device)
        for slot, position in ((0, 0), (2, capacity-1)):
            expected_k[1, slot, :, position] = k[slot, :, 0]
            expected_v[1, slot, :, position] = v[slot, :, 0]
        _, _, mask = cache.decode(1, k, v, lengths, active)
        assert torch.equal(cache.k, expected_k) and torch.equal(cache.v, expected_v)
        assert mask[1].sum().item() == 1 and mask[1, 0, 0, 0]
    finally:
        torch.use_deterministic_algorithms(previous)
