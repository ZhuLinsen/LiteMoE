"""Small, static expert oracle isolates Graph/state/ring correctness on CUDA."""

import pytest
import torch
import torch.nn.functional as F

from litemoe.engine import Engine, Request
from litemoe.cache import KVCache
from litemoe.ops import Cublas, Reference, linear_rows, reference_experts


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize('theta', [10000.0, 1000000.0])
@torch.inference_mode()
def test_long_rope_matches_hf_cpu_initialization(theta):
    from transformers import GraniteMoeConfig
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeRotaryEmbedding
    from litemoe.model import Granite
    cfg = GraniteMoeConfig(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
                           rope_parameters=dict(rope_type='default', rope_theta=theta))
    cfg.rope_theta = theta
    dummy = torch.zeros(1, 1, device='cuda', dtype=torch.bfloat16)
    model = Granite(cfg, {'embedding': dummy}, None, 4096)
    reference = GraniteMoeRotaryEmbedding(cfg).cuda()
    cosine, sine = reference(dummy, torch.arange(4096, device='cuda')[None])
    torch.testing.assert_close(model.cos, cosine[0], rtol=0, atol=0)
    torch.testing.assert_close(model.sin, sine[0], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_single_row_decode_and_dynamic_pointer_graph():
    torch.manual_seed(41)
    x = torch.randn(3, 64, device='cuda', dtype=torch.bfloat16) / 8
    w1 = torch.randn(8, 64, 64, device='cuda', dtype=x.dtype) / 8
    w2 = torch.randn(8, 64, 32, device='cuda', dtype=x.dtype) / 8
    ids = torch.randn(3, 8, device='cuda').topk(4, -1).indices.to(torch.int32)
    scales = torch.randn(3, 4, device='cuda').softmax(-1)
    assert torch.equal(linear_rows(x, w1[0], True), torch.cat([F.linear(a, w1[0]) for a in x.split(1)]))
    op, ref = Cublas(), Reference()
    out = torch.empty_like(x)
    for _ in range(3):
        op.decode(x, ids, scales, w1, w2, 0, 1, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.decode(x, ids, scales, w1, w2, 0, 1, out)
    for _ in range(2):
        ids.copy_(torch.randn(3, 8, device='cuda').topk(4, -1).indices)
        graph.replay()
        expected = ref.decode(x, ids, scales, w1, w2, 0, 1, torch.empty_like(x))
        assert torch.equal(out, expected)
        parts = []
        for rank in range(2):
            routes = torch.full((3, 4, 64), float('nan'), device='cuda', dtype=x.dtype)
            op.routes_decode(x, ids, scales, w1.chunk(2)[rank], w2.chunk(2)[rank], rank, 2, routes)
            assert torch.isfinite(routes).all()
            parts.append(routes)
        order = ids.flatten().argsort()
        actual = torch.zeros_like(x).index_add_(0, order // 4, (parts[0] + parts[1]).flatten(0, 1)[order])
        assert torch.equal(actual, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_bf16_reordered_experts_bitwise():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(29)
        x = torch.randn(31, 64, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(16, 64, 64, device="cuda", dtype=torch.bfloat16) / 8
        w2 = torch.randn(16, 64, 32, device="cuda", dtype=torch.bfloat16) / 8
        ids = torch.randn(31, 16, device="cuda").topk(8, -1).indices.to(torch.int32)
        scales = torch.randn(31, 8, device="cuda").softmax(-1)
        physical = torch.randperm(16, device="cuda")
        logical = physical.argsort()
        expected = reference_experts(x, ids, scales, w1, w2)
        actual = reference_experts(x, physical[ids].to(ids.dtype), scales,
                                   w1[logical], w2[logical], logical=logical)
        assert torch.equal(actual, expected)
    finally:
        torch.use_deterministic_algorithms(previous)


class StaticOracle:
    # Only for four-expert toy tests; not a production NE-padded backend.
    graph_safe = True

    def __call__(self, x, ids, scales, w1, w2, rank, world, out):
        out.zero_()
        for expert in range(w1.shape[0]):
            gate, up = F.linear(x, w1[expert]).chunk(2, -1)
            y = F.linear(F.silu(gate) * up, w2[expert])
            factor = ((ids == expert) * scales).sum(-1)
            out.add_(y * factor[:, None])
        return out


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_graph_restore_and_async_ring(tiny):
    _, model, _ = tiny
    model.w = {**model.w,
               "embedding": model.w["embedding"].cuda(), "head": model.w["head"].cuda(),
               "norm": model.w["norm"].cuda(),
               "layers": [{k: v.cuda() for k, v in layer.items()} for layer in model.w["layers"]]}
    model.cos, model.sin = model.cos.cuda(), model.sin.cuda()
    model.ep.device = "cuda"
    model.ep.backend = StaticOracle()
    model.ep.buffers.clear()
    requests = [Request("a", [3, 8, 2], 7), Request("b", [4], 2), Request("c", [9, 2], 1)]
    eager, _ = Engine(model, requests, ignore_eos=True, capacity=16).generate()
    for mode in ("graph", "graph-async"):
        engine = Engine(model, requests, mode=mode, ignore_eos=True, capacity=16)
        for _ in range(3):
            actual, metrics = engine.generate()
            assert actual == eager
            assert metrics["output_tokens"] == 10 and metrics["decode_tokens"] == 7
            assert engine.state["length"].tolist() == [9, 2, 2]
    # Exercise device EOS suppression and delayed global polling inside Graph.
    model.cfg.eos_token_id = eager["a"][0]
    expected, _ = Engine(model, requests, poll=4, capacity=16).generate()
    assert expected["a"] == [eager["a"][0]]
    for mode in ("graph", "graph-async"):
        engine = Engine(model, requests, mode=mode, poll=4, capacity=16)
        actual, _ = engine.generate()
        assert actual == expected
        assert engine.state["length"][0].item() == len(requests[0].prompt)
        assert not engine.state["active"].any()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_bf16_hf_sdpa_incremental_deterministic(tiny):
    hf, model, _ = tiny
    hf = hf.cuda().to(torch.bfloat16)
    hf.config._attn_implementation = "sdpa"
    for key in ("embedding", "norm", "head"):
        model.w[key] = model.w[key].cuda().to(torch.bfloat16)
    model.w["layers"] = [{k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()
                          for k, v in layer.items()} for layer in model.w["layers"]]
    model.cos, model.sin = model.cos.cuda().to(torch.bfloat16), model.sin.cuda().to(torch.bfloat16)
    model.ep.device, model.ep.dtype = "cuda", torch.bfloat16
    model.ep.buffers.clear()
    model.ep.prepare(3)
    model.ep.prepare(1)
    cache = KVCache(model.cfg, 1, 8, "cuda", torch.bfloat16)
    ids = torch.tensor([[3, 4, 8]], device="cuda")
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        actual = model.forward(ids, torch.arange(3, device="cuda")[None], torch.ones_like(ids, dtype=torch.bool), cache, slot=0)
        expected = hf(ids, use_cache=True)
        for step in range(3):
            torch.testing.assert_close(actual[:, -1], expected.logits[:, -1], atol=.002, rtol=.02)
            token = expected.logits[:, -1].argmax(-1)[:, None]
            actual = model.forward(token, torch.tensor([[3 + step]], device="cuda"), torch.tensor([[True]], device="cuda"), cache)
            expected = hf(token, past_key_values=expected.past_key_values, use_cache=True)
    finally:
        torch.use_deterministic_algorithms(before)
