import pytest
import torch

from litemoe.cache import KVCache
from litemoe.engine import Engine, Request
from litemoe.ops import Reference, reference_experts
from litemoe.placement import Placement, lpt, uniform
from litemoe.weights import load_weights


def test_placement_capacity_and_bijection():
    p = lpt([[100, 80, 60, 1, 1, 1, 1, 1]], 2)
    p.validate(1, 8, 2)
    assert len(p.logical_on_rank(0, 0)) == len(p.logical_on_rank(0, 1)) == 4
    assert p.physical == lpt([[100, 80, 60, 1, 1, 1, 1, 1]], 2).physical
    with pytest.raises(ValueError):
        Placement([[0, 0]], 1).validate(1, 2, 1)
    with pytest.raises(ValueError):
        lpt([[float("nan"), 1]], 2)


def test_expert_weight_once_and_no_local_hits():
    x = torch.tensor([[1., 2.], [2., 1.]])
    w1 = torch.arange(16).float().view(2, 4, 2) / 20
    w2 = torch.arange(8).float().view(2, 2, 2) / 20
    ids = torch.tensor([[0, 1], [1, 0]])
    scales = torch.tensor([[.25, .75], [.1, .9]])
    all_experts = reference_experts(x, ids, scales, w1, w2)
    parts = sum(reference_experts(x, ids, scales, w1[r:r + 1], w2[r:r + 1], r, 2) for r in range(2))
    torch.testing.assert_close(all_experts, parts)
    doubled = reference_experts(x, ids, scales * 2, w1, w2)
    torch.testing.assert_close(doubled, all_experts * 2)
    out = torch.full_like(x, float("nan"))
    reference_experts(x, torch.ones_like(ids), scales, w1[:1], w2[:1], 0, 2, out)
    assert torch.equal(out, torch.zeros_like(out))
    assert not reference_experts(x, ids, scales * 0, w1, w2).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_source_sized_routes_restore_reference_and_clear_unowned(dtype):
    torch.manual_seed(35)
    x = torch.randn(6, 16).to(dtype)
    w1, w2 = torch.randn(8, 24, 16).to(dtype), torch.randn(8, 16, 12).to(dtype)
    ids = torch.randn(6, 8).topk(4, -1).indices.to(torch.int32)
    scales = torch.rand(6, 4).softmax(-1).to(dtype).float()
    scales[-1] = 0  # Padding must not retain poisoned route output.
    physical = torch.tensor([5, 0, 7, 2, 6, 1, 4, 3])
    logical = physical.argsort()
    partials = []
    for rank in range(2):
        owned = logical.chunk(2)[rank]
        out = torch.full((6, 4, 16), float('nan'), dtype=dtype)
        Reference().routes(x, physical[ids].to(torch.int32), scales, w1[owned], w2[owned], rank, 2, out)
        local = (physical[ids] // 4) == rank
        assert torch.isfinite(out).all()
        assert not out[~local].any()
        partials.append(out)
    returned = partials[0] + partials[1]  # Each route has exactly one owner.
    for a, i, s, routes in zip(x.chunk(2), ids.chunk(2), scales.chunk(2), returned.chunk(2)):
        order = i.flatten().argsort()
        actual = torch.zeros_like(a).index_add_(0, order // 4, routes.flatten(0, 1)[order])
        valid_length = int(s.any(-1).sum())
        expected = torch.zeros_like(a)
        expected[:valid_length] = reference_experts(a[:valid_length], i[:valid_length], s[:valid_length], w1, w2)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("world", [1, 2])
def test_reordered_experts_keep_logical_combine(world):
    torch.manual_seed(29)
    x = torch.randn(9, 16)
    w1, w2 = torch.randn(8, 24, 16), torch.randn(8, 16, 12)
    ids = torch.randn(9, 8).topk(4, dim=-1).indices.to(torch.int32)
    scales = torch.randn(9, 4).softmax(-1)
    physical = torch.tensor([5, 0, 7, 2, 6, 1, 4, 3])
    logical = physical.argsort()
    expected = reference_experts(x, ids, scales, w1, w2)
    parts = []
    for rank in range(world):
        owned = logical.chunk(world)[rank]
        parts.append(reference_experts(x, physical[ids].to(ids.dtype), scales,
                     w1[owned], w2[owned], rank, world, logical=logical))
    torch.testing.assert_close(sum(parts), expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_hf_prefill_and_incremental_logits(tiny):
    hf, model, _ = tiny
    model.ep.prepare(4)
    model.ep.prepare(1)
    cache = KVCache(model.cfg, 1, 8, "cpu", torch.float32)
    tokens = torch.tensor([[2, 5, 7, 9]])
    logits = model.forward(tokens, torch.arange(4)[None], torch.ones(1, 4, dtype=torch.bool), cache, slot=0)
    reference = hf(tokens).logits
    torch.testing.assert_close(logits, reference, atol=2e-6, rtol=2e-5)
    token = reference[:, -1].argmax(-1)[:, None]
    actual = model.forward(token, torch.tensor([[4]]), torch.tensor([[True]]), cache)
    expected = hf(torch.cat((tokens, token), -1)).logits[:, -1:]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_variable_prompts_budgets_and_capacity(tiny):
    hf, model, _ = tiny
    requests = [Request("long", [3, 4, 8], 3), Request("short", [2], 1), Request("zero", [], 0)]
    engine = Engine(model, requests, ignore_eos=True)
    outputs, metrics = engine.generate()
    for request in requests[:2]:
        seq, expected = request.prompt[:], []
        for _ in range(request.budget):
            token = hf(torch.tensor([seq])).logits[0, -1].argmax().item()
            expected.append(token)
            seq.append(token)
        assert outputs[request.request_id] == expected
    assert outputs["zero"] == []
    assert metrics["output_tokens"] == 4 and metrics["decode_tokens"] == 2
    assert engine.state["length"].tolist() == [5, 1]
    again, _ = engine.generate()
    assert outputs == again
    with pytest.raises(ValueError):
        Engine(model, requests, capacity=4)


@torch.inference_mode()
def test_eos_first_token_and_zero_decode(tiny):
    _, model, _ = tiny
    model.w["head"] = torch.zeros_like(model.w["head"])
    engine = Engine(model, [Request("a", [2, 3], 5), Request("b", [4], 1)])
    outputs, metrics = engine.generate()
    assert outputs == {"a": [0], "b": [0]}
    assert engine.state["length"].tolist() == [2, 1]
    assert metrics["decode_tps"] is None
    assert not engine.cache.k.isnan().any()


@torch.inference_mode()
def test_inactive_full_slot_safe_and_frozen(tiny):
    _, model, _ = tiny
    engine = Engine(model, [Request("a", [2, 3], 1)])
    engine.generate()
    before = engine.cache.k.clone()
    engine.state["length"].fill_(engine.capacity)
    engine.decode_step()
    torch.testing.assert_close(before, engine.cache.k)
    assert engine.state["generated"].item() == 1
    assert not engine.emit[:, 2].any()


@torch.inference_mode()
def test_placement_reload_changes_storage_preserves_semantics(tiny):
    _, model, source = tiny
    requests = [Request("a", [2, 3, 4], 3)]
    expected, _ = Engine(model, requests, ignore_eos=True).generate()
    placement = Placement([[3, 0, 2, 1], [1, 3, 0, 2]], 1)
    old = model.w["layers"][0]["input_linear"].clone()
    model.w = load_weights(model.cfg, source, placement, 0, "cpu", torch.float32)
    assert not torch.equal(old, model.w["layers"][0]["input_linear"])
    actual, _ = Engine(model, requests, ignore_eos=True).generate()
    assert actual == expected


def test_empty_and_invalid_requests(tiny):
    _, model, _ = tiny
    assert Engine(model, [Request("z", [], 0)]).generate()[0] == {"z": []}
    with pytest.raises(ValueError):
        Engine(model, [Request("x", [], 1)])
    with pytest.raises(ValueError):
        Engine(model, [Request("x", [1], -1)])
    with pytest.raises(ValueError):
        Engine(model, [Request("x", [1], 1)], mode="graph")


@torch.inference_mode()
def test_mixed_eos_budget_and_delayed_polling(tiny):
    hf, model, _ = tiny
    first_eos = hf(torch.tensor([[3, 4, 8]])).logits[0, -1].argmax().item()
    model.cfg.eos_token_id = first_eos
    requests = [Request("eos", [3, 4, 8], 6), Request("budget", [2], 2)]
    engine = Engine(model, requests, poll=4)
    outputs, metrics = engine.generate()
    assert outputs["eos"] == [first_eos]
    assert 1 <= len(outputs["budget"]) <= 2
    assert engine.state["length"][0].item() == 3
    assert metrics["output_tokens"] == sum(map(len, outputs.values()))
    assert not engine.state["active"].any()
