"""Lean execution must keep KV placement and never touch stopped rows' KV."""
import pytest
import torch

from litemoe.engine import Request
from litemoe.ep import ExpertParallel
from litemoe.model import Granite
from litemoe.optimized import OptimizedEngine, optimized_policy
from litemoe.placement import uniform
from litemoe.vllm_moe import VllmFusedExperts
from litemoe.weights import load_weights

REQUESTS = [("stops", [2, 3, 5], 2), ("long", [7, 11], 8), ("mid", [13], 4), ("one", [17], 1)]


def engine(tiny, lean, physical=None):
    _, mini, source = tiny
    cfg = mini.cfg
    placement = uniform(cfg.num_hidden_layers, cfg.num_local_experts, 1)
    if physical is not None:
        placement.physical = [physical(cfg.num_local_experts) for _ in placement.physical]
    weights = load_weights(cfg, source, placement, 0, "cuda", torch.bfloat16)
    ep = ExpertParallel(1, 0, 16, cfg.hidden_size, cfg.num_experts_per_tok, torch.bfloat16, "cuda", VllmFusedExperts())
    model = Granite(cfg, weights, ep, 16)
    model.prepare_packed_qkv()
    model.decode_packed_qkv = True
    run = OptimizedEngine(model, [Request(*r) for r in REQUESTS], capacity=16, ignore_eos=True, mode="eager")
    run.lean_decode = lean
    return run


@pytest.mark.cuda
@torch.inference_mode()
def test_lean_prefill_kv_matches_control_and_keeps_stopped_kv(tiny):
    control, run = engine(tiny, False), engine(tiny, True)
    for item in (control, run):
        with optimized_policy(item):
            item.prefill()
    # Different slots and lengths must land in the same KV positions (BF16 rounding differs).
    torch.testing.assert_close(run.cache.k, control.cache.k, atol=.03, rtol=.03)
    torch.testing.assert_close(run.cache.v, control.cache.v, atol=.03, rtol=.03)
    outputs, _ = engine(tiny, True).generate()
    assert {k: len(v) for k, v in outputs.items()} == {r[0]: r[2] for r in REQUESTS}
    run = engine(tiny, True)
    run.generate()
    # "one" never decodes and "stops" ends after one step: no writes past their final KV position.
    assert not run.cache.k[:, 3, :, 1:].count_nonzero() and not run.cache.v[:, 3, :, 1:].count_nonzero()
    assert not run.cache.k[:, 0, :, 4:].count_nonzero() and not run.cache.v[:, 0, :, 4:].count_nonzero()


@pytest.mark.cuda
@torch.inference_mode()
def test_lean_is_scoped_and_rejects_reordered_placement(tiny):
    run = engine(tiny, True)
    with optimized_policy(run):
        assert run.model.decode_lean and run.model.ep.decode_lean
    assert not run.model.decode_lean and not run.model.ep.decode_lean
    reordered = engine(tiny, True, physical=lambda n: list(range(n))[::-1])
    with optimized_policy(reordered):
        assert not reordered.model.decode_lean and not reordered.model.ep.decode_lean


@pytest.mark.cuda
def test_decode_tiles_only_for_screened_a100_batches(tiny, monkeypatch):
    from litemoe.optimized import A100_DECODE_TILES, decode_tiles
    run = engine(tiny, True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA L4")
    assert decode_tiles(run) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-SXM4-40GB")
    assert decode_tiles(run) is A100_DECODE_TILES
    run.batch = 33
    assert decode_tiles(run) is None
