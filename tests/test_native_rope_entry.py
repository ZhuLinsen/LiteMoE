"""Production RoPE entry coverage; operator Graph/state tests live in existing suites."""

import importlib.util
import sys
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from litemoe.engine import Request
from litemoe.ep import ExpertParallel
from litemoe.model import Granite
from litemoe.ops import Grouped, Reference
from litemoe.optimized import OptimizedEngine, optimized_policy
from litemoe.placement import uniform
from litemoe.vllm_moe import VllmFusedExperts
from litemoe.weights import load_weights


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VLLM_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="Pinned vLLM CUDA environment required",
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, CUDA])])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_native_cache_preserves_legacy_cos_sin(tiny, device, dtype):
    _, mini, _ = tiny
    # Reuse the real, loader-normalized Granite config, with production D64 and
    # every position through 4095; the fixture's D8/64-position cache is too small.
    cfg = copy(mini.cfg)
    cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads = 1024, 16, 8
    cfg.max_position_embeddings = capacity = 4096
    model = Granite(cfg, {"embedding": torch.empty(1, cfg.hidden_size, device=device, dtype=dtype)}, None, capacity)
    dim = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / cfg.rope_theta ** (torch.arange(0, dim, 2, device="cpu").float() / dim)
    freq = torch.outer(torch.arange(capacity, device=device).float(), inv.to(device))
    angle = torch.cat((freq, freq), -1)
    cosine, sine = angle.cos().to(dtype), angle.sin().to(dtype)
    torch.testing.assert_close(model.cos, cosine, rtol=0, atol=0)
    torch.testing.assert_close(model.sin, sine, rtol=0, atol=0)
    torch.testing.assert_close(model.cos_sin, torch.cat((cosine[:, :dim // 2], sine[:, :dim // 2]), -1),
                               rtol=0, atol=0)
    assert model.cos_sin.is_contiguous() and model.cos_sin.dtype == dtype


@torch.inference_mode()
def test_generate_scopes_policy_to_execution_and_keeps_cpu_rope_fallback(tiny, monkeypatch):
    _, model, _ = tiny

    class CPUVllm(VllmFusedExperts):
        # The type selects production policy; CPU expert math stays in Reference.
        def __init__(self):
            self.reference = Reference()

        def __call__(self, *args, **kwargs):
            return self.reference(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU attention reached the native CUDA RoPE operator")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=SimpleNamespace(rotary_embedding=forbidden)))
    model.ep.backend = CPUVllm()
    requests = [Request("long", [2, 3, 5], 4), Request("one", [7], 1)]
    fields = ("native_gqa", "decode_kv_limit", "decode_folded_q", "decode_batched_linears",
              "decode_fused_norm", "decode_flash")

    def policy_state():
        return tuple(getattr(model, name) for name in fields) + (model.ep.prefill_backend,)

    original = policy_state()
    callbacks, phases = [], []

    def consume(tokens):
        # The old generate-wide wrapper leaves the optimized policy active here.
        assert policy_state() == original
        callbacks.append(tuple(tokens))

    run = OptimizedEngine(model, requests, capacity=16, ignore_eos=True, host_process=consume)
    raw_forward = model.forward

    def observe_forward(*args, **kwargs):
        assert policy_state() == (False, 6, True, True, run.fused_norm, run.flash_attention,
                                  run.prefill_reference)
        phases.append("prefill" if kwargs.get("slot") is not None else "decode")
        return raw_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", observe_forward)
    first, _ = run.generate()
    second, _ = run.generate()
    assert first == second and {key: len(value) for key, value in first.items()} == {"long": 4, "one": 1}
    assert set(phases) == {"prefill", "decode"} and len(callbacks) == 10
    assert policy_state() == original


@pytest.mark.cuda
@VLLM_CUDA
@pytest.mark.parametrize("backend,flash,prefill", [
    ("vllm", True, False), ("reference", True, False), ("grouped", True, False),
    ("vllm", False, False), ("vllm", True, True),
])
@torch.inference_mode()
def test_production_attention_native_rope_dispatch(tiny, monkeypatch, backend, flash, prefill):
    from vllm import _custom_ops as ops

    _, mini, source = tiny
    cfg, dtype, capacity = mini.cfg, torch.bfloat16, 16
    weights = load_weights(cfg, source, uniform(cfg.num_hidden_layers, cfg.num_local_experts, 1),
                           0, "cuda", dtype)
    vllm = VllmFusedExperts()
    # Attention dispatch reads backend identity only. Grouped expert kernels have
    # their own tests; do not require DeepSpeed to exercise this bypass branch.
    op = {"vllm": vllm, "reference": Reference(), "grouped": object.__new__(Grouped)}[backend]
    ep = ExpertParallel(1, 0, capacity, cfg.hidden_size, cfg.num_experts_per_tok, dtype, "cuda", op)
    model = Granite(cfg, weights, ep, capacity)
    run = OptimizedEngine(model, [Request("active", [2, 3, 5], 3), Request("inactive", [7], 1)],
                          capacity=capacity, ignore_eos=True)
    assert run.flash_decode
    run.flash_decode = flash
    run.lean_decode = False  # This test pins the masked control contract.
    z = weights["embedding"][torch.tensor([[2], [7]], device="cuda")]
    positions = torch.tensor([[0 if prefill else 3], [9999]], device="cuda")
    valid = torch.tensor([[True], [False]], device="cuda")
    slot = slice(0, 2) if prefill else None
    calls = []
    raw_rope = ops.rotary_embedding

    def observe_rope(*args, **kwargs):
        calls.append(args[0].clone())
        return raw_rope(*args, **kwargs)

    monkeypatch.setattr(ops, "rotary_embedding", observe_rope)
    with optimized_policy(run):
        actual = model.attention(z, weights["layers"][0], run.cache, 0, positions, valid, slot)
    native = backend == "vllm" and flash and not prefill
    assert len(calls) == int(native)
    assert torch.isfinite(actual).all() and not actual[1].count_nonzero()
    assert model.decode_flash is None and model.ep.prefill_backend is None
    if native:
        assert calls[0].tolist() == [3, 0]
        native_k, native_v = run.cache.k.clone(), run.cache.v.clone()
        run.cache.reset()
        model.ep.backend = Reference()
        with optimized_policy(run):
            expected = model.attention(z, weights["layers"][0], run.cache, 0, positions, valid, None)
        assert len(calls) == 1
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.03)
        torch.testing.assert_close(native_k, run.cache.k, atol=.004, rtol=.03)
        torch.testing.assert_close(native_v, run.cache.v, atol=0, rtol=0)
