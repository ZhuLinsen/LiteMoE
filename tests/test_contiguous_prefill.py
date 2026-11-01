import pytest
import torch
from litemoe.batch import prepare_batch, prefill_logits
from litemoe.cache import KVCache
from litemoe.batch import bucket_prefill_logits as bucket_prefill
from litemoe.engine import Engine, Request
from litemoe.optimized import OptimizedEngine
from litemoe.packed import optimized_prefill_logits


@torch.inference_mode()
@pytest.mark.parametrize('lengths', [[1,2,3,10,20], [20,1,10,2,3]])
def test_bucket_prefill_keeps_nonzero_slot_offset_and_logits(tiny, lengths):
    hf, model, _ = tiny
    prompts = [[2+i]*n for i,n in enumerate(lengths)]
    prepare_batch(model, len(prompts)*max(lengths))
    model.ep.prepare(len(prompts)+2)
    actual_cache = KVCache(model.cfg, len(prompts)+2, 32, 'cpu', torch.float32)
    control_cache = KVCache(model.cfg, len(prompts)+2, 32, 'cpu', torch.float32)
    actual = bucket_prefill(model,prompts,actual_cache,1)
    expected = prefill_logits(model,prompts,control_cache,1)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual_cache.k,control_cache.k,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual_cache.v,control_cache.v,atol=2e-6,rtol=2e-5)
    for i,prompt in enumerate(prompts):
        oracle = hf(torch.tensor([prompt])).logits[:,-1:]
        torch.testing.assert_close(actual[i:i+1],oracle,atol=2e-6,rtol=2e-5)


@torch.inference_mode()
@pytest.mark.parametrize('ignore_eos', [True, False])
def test_optimized_grouped_prefill_preserves_request_ids_budgets_and_reset(tiny, ignore_eos):
    hf,model,_=tiny
    model.cfg.eos_token_id=hf(torch.tensor([[3,4,8]])).logits[0,-1].argmax().item()
    requests=[Request('long',list(range(2,22)),3), Request('short',[2],7),
              Request('eos',[3,4,8],6), Request('one',[6,7],1), Request('empty',[],0)]
    expected,_=Engine(model,requests,ignore_eos=ignore_eos).generate()
    run=OptimizedEngine(model,requests,ignore_eos=ignore_eos,poll=4)
    assert run.prefill_function is optimized_prefill_logits
    for _ in range(2):
        actual,metrics=run.generate()
        assert actual==expected
        assert metrics['output_tokens']==sum(map(len,actual.values()))
