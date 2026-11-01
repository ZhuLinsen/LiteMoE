"""EP1 variable-length Prefill; preserves request-local KV slots."""
from typing import NamedTuple
import torch


class PackedPrefill(NamedTuple):
    cu: torch.Tensor
    maximum: int
    slots: torch.Tensor
    positions: torch.Tensor

    def attention(self, q, k, v, scale):
        values = [x[0].transpose(0, 1) for x in (q, k, v)]
        return torch.ops.aten._flash_attention_forward(
            *values, self.cu, self.cu, self.maximum, self.maximum,
            0.0, True, False, scale=scale)[0]


def packed_prefill_logits(model, prompts, cache, slot_start=0):
    from .batch import prepare_batch
    if not cache.k.is_cuda:
        raise ValueError("Packed Prefill requires CUDA FlashAttention")
    lengths = [len(p) for p in prompts]
    if not lengths or min(lengths) < 1:
        raise ValueError("Packed Prefill requires nonempty prompts")
    prepare_batch(model, sum(lengths))
    device = cache.k.device
    ends = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu = torch.cat((ends.new_zeros(1), ends))
    positions = torch.tensor([i for n in lengths for i in range(n)], device=device)
    slots = torch.tensor([slot_start + i for i, n in enumerate(lengths) for _ in range(n)], device=device)
    packed = PackedPrefill(cu, max(lengths), slots, positions)
    tokens = torch.tensor([t for p in prompts for t in p], device=device)[None]
    # One packed row; broadcast row index 0 against every request's final position.
    return model.forward(tokens, positions[None], torch.ones_like(tokens, dtype=torch.bool),
                         cache, slot=packed, last=ends.long() - 1)


def optimized_prefill_logits(model, prompts, cache, slot_start=0):
    from .batch import bucket_prefill_logits, prefill_logits, prepare_batch
    if not cache.k.is_cuda:
        return bucket_prefill_logits(model, prompts, cache, slot_start)
    lengths = [len(p) for p in prompts]
    width, batch = max(lengths), len(lengths)
    if model.prefill_lean and min(lengths) > 0:
        return packed_prefill_logits(model, prompts, cache, slot_start)
    # Short Prefill is launch-bound; packing wins with long attention or substantial padding.
    if batch > 1 and (width >= 1024 or (width >= 512 and 3 * sum(lengths) <= 2 * batch * width)):
        return packed_prefill_logits(model, prompts, cache, slot_start)
    prepare_batch(model, batch * width)
    return prefill_logits(model, prompts, cache, slot_start)
