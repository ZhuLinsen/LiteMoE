import pytest
import torch
import torch.nn.functional as F
from litemoe.packed import PackedPrefill, packed_prefill_logits, optimized_prefill_logits
from litemoe.cache import KVCache
from litemoe.batch import prepare_batch, prefill_logits
from litemoe.model import Granite
from litemoe.ep import ExpertParallel
from litemoe.ops import Reference
from litemoe.placement import uniform
from litemoe.weights import load_weights


def test_packed_cache_scatter_preserves_slots_and_unused_positions(tiny):
    _, model, _ = tiny
    cache = KVCache(model.cfg, 4, 32, 'cpu', torch.float32)
    slots, positions = torch.tensor([1, 1, 2]), torch.tensor([0, 1, 0])
    layout = PackedPrefill(torch.tensor([0, 2, 3], dtype=torch.int32), 2, slots, positions)
    k = torch.arange(48.).reshape(1, 2, 3, 8)
    cache.prefill(0, layout, k, k + 1, torch.ones(1, 3, dtype=torch.bool))
    for i, (s, p) in enumerate(zip(slots, positions)):
        torch.testing.assert_close(cache.k[0, s, :, p], k[0, :, i])
    assert cache.k[:, [0, 3]].count_nonzero() == 0
    assert cache.k[:, :, :, 2:].count_nonzero() == 0
    assert cache.k[1].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_packed_flash_gqa_causal_and_sequence_isolation():
    torch.manual_seed(11)
    q = torch.randn(1, 4, 9, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 2, 9, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    layout = PackedPrefill(torch.tensor([0, 2, 9], device='cuda', dtype=torch.int32), 7, None, None)
    actual = layout.attention(q, k, v, .125)
    expected = torch.cat([F.scaled_dot_product_attention(q[:, :, l:r],
        k[:, :, l:r].repeat_interleave(2, 1), v[:, :, l:r].repeat_interleave(2, 1),
        is_causal=True, scale=.125)[0].transpose(0, 1) for l, r in [(0, 2), (2, 9)]])
    torch.testing.assert_close(actual, expected, atol=.016, rtol=.016)
    k[:, :, 1] += 4; v[:, :, 1] += 4
    changed = layout.attention(q, k, v, .125)
    torch.testing.assert_close(changed[0], actual[0], atol=0, rtol=0)
    torch.testing.assert_close(changed[2:], actual[2:], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('lengths,adaptive,uses_packed', [([20,1,3],False,True),
    ([20,1,3],True,False),([512,128,256],True,True),
    ([512,480,384],True,False),([1024,960,768],True,True),([1024],True,False)])
@torch.inference_mode()
def test_packed_model_unsorted_prefill_and_decode(tiny, monkeypatch, lengths, adaptive, uses_packed):
    hf, mini, source = tiny
    cfg = mini.cfg
    weights = load_weights(cfg, source, uniform(2, 4, 1), 0, 'cuda', torch.bfloat16)
    ep = ExpertParallel(1, 0, 128, 32, 2, torch.bfloat16, 'cuda', Reference())
    capacity = max(lengths) + 16
    model = Granite(cfg, weights, ep, capacity)
    hf = hf.to(device='cuda', dtype=torch.bfloat16)
    prompts = [[2+(i+j)%40 for j in range(n)] for i,n in enumerate(lengths)]
    packed, padded = [KVCache(cfg, len(prompts), capacity, 'cuda', torch.bfloat16) for _ in range(2)]
    prepare_batch(model, len(prompts)*max(lengths))
    expected = prefill_logits(model, prompts, padded)
    calls, raw = [], PackedPrefill.attention
    def observed(self, *args):
        calls.append(True)
        return raw(self, *args)
    monkeypatch.setattr(PackedPrefill, 'attention', observed)
    fn = optimized_prefill_logits if adaptive else packed_prefill_logits
    actual = fn(model, prompts, packed)
    assert bool(calls) == uses_packed
    torch.testing.assert_close(actual, expected, atol=.004, rtol=.03)
    for i, prompt in enumerate(prompts):
        torch.testing.assert_close(actual[i:i+1], hf(torch.tensor([prompt], device='cuda')).logits[:, -1:], atol=.004, rtol=.03)
        assert packed.k[:, i, :, len(prompt):].count_nonzero() == 0
    torch.testing.assert_close(packed.k, padded.k, atol=.004, rtol=.03)
    torch.testing.assert_close(packed.v, padded.v, atol=.004, rtol=.03)
    tokens = expected[:, 0].argmax(-1)[:, None]
    positions = torch.tensor([[len(p)] for p in prompts], device='cuda')
    valid = torch.ones_like(tokens, dtype=torch.bool)
    ep.prepare(len(prompts))
    continuation = model.forward(tokens, positions, valid, packed)
    control = model.forward(tokens, positions, valid, padded)
    torch.testing.assert_close(continuation, control, atol=.004, rtol=.03)
