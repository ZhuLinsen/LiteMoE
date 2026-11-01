import pytest
import torch

from litemoe.engine import Request
from litemoe.optimized import OptimizedEngine, optimized_policy
from litemoe.ops import combine


def test_combine_policy_change_requires_new_graph(tiny):
    _, model, _ = tiny
    run = OptimizedEngine(model, [Request('a', [2], 3)], capacity=8)
    with optimized_policy(run):
        run.graph = object()
    setattr(model.ep.backend, 'fused_combine', True)
    with pytest.raises(ValueError, match='Recreate'):
        with optimized_policy(run):
            pytest.fail('Captured combine policy changed silently')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch,topk,hidden,rank,world,remap', [
    (1,8,1536,0,1,False), (8,8,2048,0,1,True),
    (32,8,65,1,2,False), (8,3,129,0,2,True), (1,1,65,0,1,False)])
@torch.inference_mode()
def test_fused_combine_mask_order_and_dynamic_graph(batch,topk,hidden,rank,world,remap):
    from litemoe.moe_finish import finish
    torch.manual_seed(29)
    count = 8
    ids = torch.stack([torch.randperm(count*world,device='cuda')[:topk] for _ in range(batch)]).int()
    order = ids.flatten().argsort()
    y = torch.randn(batch*topk,hidden,device='cuda',dtype=torch.bfloat16)
    scales = torch.softmax(torch.randn(batch,topk,device='cuda'),-1)
    if batch > 1:
        scales[-1] = 0
    logical = torch.randperm(count*world,device='cuda') if remap else None
    out = torch.empty(batch,hidden,device='cuda',dtype=torch.bfloat16)

    def expected():
        selected = ids.flatten()[order].long() - rank*count
        weighted = torch.where(((selected>=0)&(selected<count))[:,None],y,0)
        weighted = weighted * scales.flatten()[order,None].to(y.dtype)
        return combine(torch.zeros_like(out),ids,order,weighted,logical)

    def check():
        ref = expected().float()
        assert torch.isfinite(out).all()
        relative = (out.float()-ref).norm()/ref.norm().clamp_min(1e-12)
        assert relative <= .01
        if batch > 1:
            assert torch.count_nonzero(out[-1]) == 0

    selected = ids.flatten()[order].long()-rank*count
    y[(selected<0)|(selected>=count)] = float('nan')
    finish(y,ids,order,scales,out,rank,count,logical)
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            finish(y,ids,order,scales,out,rank,count,logical)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        finish(y,ids,order,scales,out,rank,count,logical)
    # Replay must read new order, IDs, activations and scales from the same storage.
    ids.copy_(ids.roll(1,1));order.copy_(ids.flatten().argsort())
    y.normal_();scales.mul_(.5)
    selected = ids.flatten()[order].long()-rank*count
    y[(selected<0)|(selected>=count)] = float('nan')
    graph.replay();torch.cuda.synchronize()
    check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_fused_combine_never_reads_all_nonlocal_nan_rows():
    from litemoe.moe_finish import finish
    ids = torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32)
    y = torch.full((3,65),float('nan'),device='cuda',dtype=torch.bfloat16)
    out = torch.full((1,65),float('nan'),device='cuda',dtype=torch.bfloat16)
    finish(y,ids,ids.flatten().long(),torch.ones(1,3,device='cuda'),out,1,4)
    assert torch.equal(out,torch.zeros_like(out))
