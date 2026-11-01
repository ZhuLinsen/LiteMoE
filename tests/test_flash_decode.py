import pytest
import torch
from litemoe.engine import Request
from litemoe.optimized import OptimizedEngine,optimized_policy


@torch.inference_mode()
def test_flash_policy_cpu_fallback_and_capture_rejection(tiny):
 _,model,_=tiny
 run=OptimizedEngine(model,[Request('a',[2,3],3)],capacity=16)
 assert run.flash_decode and model.decode_flash is None
 run.flash_decode=False
 expected=run.generate()[0]
 run.flash_decode=True
 assert run.generate()[0]==expected and model.decode_flash is None
 with pytest.raises(RuntimeError):
  with optimized_policy(run):
   assert model.decode_flash is run.flash_attention
   raise RuntimeError('scope failure')
 assert model.decode_flash is None
 run._optimized_graph_policy=(run.capacity,4,id(model.ep.backend),model.decode_packed_qkv,run.fused_norm,False)
 run.graph=object()
 with pytest.raises(ValueError,match='Recreate Engine'):
  with optimized_policy(run):pass
 assert model.decode_flash is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_flash_decode_real_forward_masked_slot(tiny):
 from litemoe.batch import prepare_batch,prefill_logits
 from litemoe.cache import KVCache
 from litemoe.ep import ExpertParallel
 from litemoe.flash_decode import FlashDecode
 from litemoe.model import Granite
 from litemoe.ops import Reference
 from litemoe.placement import uniform
 from litemoe.weights import load_weights
 hf,mini,source=tiny;cfg=mini.cfg
 weights=load_weights(cfg,source,uniform(2,4,1),0,'cuda',torch.bfloat16)
 ep=ExpertParallel(1,0,64,32,2,torch.bfloat16,'cuda',Reference())
 model=Granite(cfg,weights,ep,32)
 caches=[KVCache(cfg,2,32,'cuda',torch.bfloat16) for _ in range(2)]
 prepare_batch(model,6)
 model.ep.prepare(2)
 for cache in caches:prefill_logits(model,[[2,3,5],[4]],cache)
 flash=FlashDecode(1,cfg.num_key_value_heads,cfg.num_attention_heads//cfg.num_key_value_heads,64,'cuda')
 tokens=torch.tensor([[6],[7]],device='cuda')
 for step in range(4):
  pos=torch.tensor([[3+step],[1+step]],device='cuda');valid=torch.tensor([[True],[step<2]],device='cuda')
  model.decode_flash=None;expected=model.forward(tokens,pos,valid,caches[0])
  model.decode_flash=flash;actual=model.forward(tokens,pos,valid,caches[1])
  torch.testing.assert_close(actual,expected,atol=.004,rtol=.03)
  tokens=expected[:,0].argmax(-1)[:,None]
 torch.testing.assert_close(caches[1].k,caches[0].k,atol=.004,rtol=.03)
 torch.testing.assert_close(caches[1].v,caches[0].v,atol=.004,rtol=.03)


@pytest.mark.parametrize("capacity,maximum", [(255,10),(256,0),(256,257)])
def test_production_paged_rejects_invalid_bounds(capacity,maximum):
 from litemoe.flash_decode import FlashDecode
 with pytest.raises(ValueError,match="aligned capacity"):
  FlashDecode(2,2,2,capacity,"cpu",maximum=maximum)


def test_production_paged_passes_view_and_masks_inactive(monkeypatch):
 import sys
 import types
 from litemoe.flash_decode import FlashDecode
 seen={}
 def attention(q,k,v,**kwargs):
  seen.update(q=q,k=k,v=v,**kwargs)
  return q.clone()
 monkeypatch.setitem(sys.modules,"vllm.vllm_flash_attn.flash_attn_interface",
                     types.SimpleNamespace(flash_attn_varlen_func=attention))
 run=FlashDecode(2,2,2,256,"cpu",maximum=63)
 q=torch.randn(2,4,1,8);k=torch.randn(2,2,256,8);v=torch.randn_like(k)
 result=run(q,k,v,torch.tensor([62,999]),torch.tensor([True,False]),.125)
 assert torch.equal(result,q)
 assert seen["seqused_k"].tolist()==[63,1]
 assert seen["max_seqlen_k"]==63 and seen["fa_version"]==2
 assert seen["block_table"].tolist()==[[0],[1]]
 assert seen["k"].untyped_storage().data_ptr()==k.untyped_storage().data_ptr()
 assert seen["v"].untyped_storage().data_ptr()==v.untyped_storage().data_ptr()


@pytest.mark.parametrize("capacity,paged",[(64,False),(256,True)])
def test_vllm_engine_selects_attention_by_capacity(tiny,monkeypatch,capacity,paged):
 from litemoe import flash_decode
 from litemoe.vllm_moe import VllmFusedExperts
 _,model,_=tiny
 model.ep.backend=object.__new__(VllmFusedExperts)
 model.cfg.max_position_embeddings=max(model.cfg.max_position_embeddings,capacity)
 captured=[]
 def factory(*args):
  captured.append(args)
  return object()
 monkeypatch.setattr(flash_decode,"FlashDecode",factory)
 OptimizedEngine(model,[Request("a",[2,3],3)],capacity=capacity)
 assert captured[0][-1]==(4 if paged else None)
