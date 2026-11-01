"""Optional pinned vLLM BF16 expert operator; Mini owns the execution engine."""
from importlib.metadata import version
import torch


class VllmFusedExperts:
    graph_safe = True
    name = 'vllm-0.32.0-bf16-fused-experts'
    fused_combine = True

    def __init__(self):
        if version('vllm') != '0.32.0':
            raise ValueError('This backend requires vLLM 0.32.0')
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        self.operation = fused_experts

    def __call__(self, x, ids, scales, w1, w2, rank, world, out, logical=None):
        if rank != 0 or world != 1:
            raise ValueError('The fused-expert backend supports EP1 only')
        if any(t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous()
               for t in (x,w1,w2,out)):
            raise ValueError('Contiguous CUDA BF16 activations/weights/output required')
        # IDs already address physical weights. Reduction order is screened,
        # rather than promised bitwise equal under a nonidentity logical map.
        value = self.operation(x, w1, w2, scales.float().contiguous(), ids.int().contiguous(),
                               apply_router_weight_on_input=False, global_num_experts=w1.shape[0])
        return out.copy_(value)
