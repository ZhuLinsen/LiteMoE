import pytest
import torch

from litemoe.batch import prefill_logits, prepare_batch
from litemoe.cache import KVCache
from litemoe.engine import Request
from litemoe.ep import ExpertParallel
from litemoe.model import Granite, rmsnorm
from litemoe.ops import Reference
from litemoe.optimized import OptimizedEngine, optimized_policy
from litemoe.placement import uniform
from litemoe.weights import load_weights


def test_norm_policy_change_rejects_captured_graph(tiny):
    _, model, _ = tiny
    run = OptimizedEngine(model, [Request('a',[2,3],3)], capacity=32)
    with optimized_policy(run):
        run.graph = object()
    run.fused_norm = not run.fused_norm
    with pytest.raises(ValueError, match='Recreate'):
        with optimized_policy(run):
            pytest.fail('Must not replay a graph captured with another norm policy')


@torch.inference_mode()
def test_norm_candidate_keeps_cpu_fallback(tiny):
    _, model, _ = tiny
    requests = [Request('a',[2,3],3),Request('b',[5],2)]
    control = OptimizedEngine(model,requests,capacity=32)
    control.fused_norm = False
    expected, _ = control.generate()
    actual, _ = OptimizedEngine(model,requests,capacity=32).generate()
    assert actual == expected
    assert not model.decode_fused_norm


def test_norm_policy_restores_model_on_nested_exit_and_failure(tiny):
    _, model, _ = tiny
    run = OptimizedEngine(model,[Request('a',[2],2)],capacity=32)
    assert run.fused_norm and not model.decode_fused_norm
    with pytest.raises(RuntimeError, match='injected'):
        with optimized_policy(run):
            with optimized_policy(run):
                assert model.decode_fused_norm
            assert model.decode_fused_norm
            raise RuntimeError('injected')
    assert not model.decode_fused_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_default_optimized_engine_uses_fused_norm_only_in_decode(tiny, monkeypatch):
    import litemoe.norm as fused_module
    hf, mini, source = tiny
    cfg = mini.cfg
    weights = load_weights(cfg,source,uniform(2,4,1),0,'cuda',torch.bfloat16)
    ep = ExpertParallel(1,0,64,32,2,torch.bfloat16,'cuda',Reference())
    model = Granite(cfg,weights,ep,32)
    run = OptimizedEngine(model,[Request('a',[2,3,5],3),Request('b',[4],1)],
                          capacity=32,ignore_eos=True)
    calls, raw = [], fused_module.rmsnorm
    def observed(*args):
        calls.append(True)
        return raw(*args)
    monkeypatch.setattr(fused_module,'rmsnorm',observed)
    run.prefill()
    assert not calls and not model.decode_fused_norm
    outputs,_ = run.generate()
    assert {k:len(v) for k,v in outputs.items()} == {'a':3,'b':1}
    assert len(calls) == 2*(2*cfg.num_hidden_layers+1)
    assert not model.decode_fused_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape',[(1,1,1536),(8,1,2048),(32,1,3072),(2,1,33)])
@torch.inference_mode()
def test_fused_norm_rounding_and_graph_replay(shape):
    from litemoe.norm import rmsnorm as fused
    torch.manual_seed(17)
    x = torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    weight = (1 + .2 * torch.randn(shape[-1],device='cuda')).bfloat16()
    eps = 1e-6
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused(x,weight,eps)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused(x,weight,eps)
    for scale in (0.,.001,1.,1000.):
        x.copy_(torch.randn_like(x)*scale)
        graph.replay()
        expected = rmsnorm(x,weight,eps)
        torch.testing.assert_close(actual,expected,atol=.016,rtol=.008)
        assert float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)) < .005


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_fused_norm_model_decode_leaves_prefill_unchanged(tiny, monkeypatch):
    import litemoe.norm as fused_module
    hf, mini, source = tiny
    cfg = mini.cfg
    weights = load_weights(cfg,source,uniform(2,4,1),0,'cuda',torch.bfloat16)
    ep = ExpertParallel(1,0,64,32,2,torch.bfloat16,'cuda',Reference())
    model = Granite(cfg,weights,ep,32)
    caches = [KVCache(cfg,2,32,'cuda',torch.bfloat16) for _ in range(2)]
    prompts = [[2,3,5],[4]]
    prepare_batch(model,len(prompts)*max(map(len,prompts)))
    control = prefill_logits(model,prompts,caches[0])
    calls, raw = [], fused_module.rmsnorm
    def observed(*args):
        calls.append(True)
        return raw(*args)
    monkeypatch.setattr(fused_module,'rmsnorm',observed)
    model.decode_fused_norm = True
    torch.testing.assert_close(prefill_logits(model,prompts,caches[1]),control,atol=0,rtol=0)
    assert not calls
    ep.prepare(2)
    valid = torch.ones(2,1,device='cuda',dtype=torch.bool)
    tokens = control[:,0].argmax(-1)[:,None]
    for step in range(4):
        positions = torch.tensor([[3+step],[1+step]],device='cuda')
        model.decode_fused_norm = False
        expected = model.forward(tokens,positions,valid,caches[0])
        model.decode_fused_norm = True
        actual = model.forward(tokens,positions,valid,caches[1])
        torch.testing.assert_close(actual,expected,atol=.004,rtol=.03)
        tokens = expected[:,0].argmax(-1)[:,None]
    assert len(calls) == 4*(2*cfg.num_hidden_layers+1)
    torch.testing.assert_close(caches[1].k,caches[0].k,atol=.004,rtol=.03)
    torch.testing.assert_close(caches[1].v,caches[0].v,atol=.004,rtol=.03)
