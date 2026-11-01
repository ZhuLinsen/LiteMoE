"""EP1 batched Prefill with request-local KV and right-padded attention."""
import torch

from .engine import Engine


def prepare_batch(model, tokens):
    if model.ep.world != 1:
        raise ValueError("Batched Prefill requires EP1")
    model.ep.capacity = max(model.ep.capacity, tokens)
    model.ep.prepare(tokens)


def prefill_logits(model, prompts, cache, slot_start=0):
    device = cache.k.device
    lengths = torch.tensor([len(x) for x in prompts], device=device)
    width = max(map(len, prompts))
    tokens = torch.tensor([x + [0] * (width - len(x)) for x in prompts], device=device)
    positions = torch.arange(width, device=device)[None].expand(len(prompts), -1)
    valid = positions < lengths[:, None]
    return model.forward(tokens, positions, valid, cache, slot=slice(slot_start, slot_start + len(prompts)), last=lengths - 1)


def bucket_prefill_logits(model, prompts, cache, slot_start=0):
    """Keep contiguous request slots while limiting within-group length ratio."""
    outputs, start = [], 0
    while start < len(prompts):
        end = start + 1
        low = high = len(prompts[start])
        while end < len(prompts):
            length = len(prompts[end])
            if max(high, length) > 8 * min(low, length):
                break
            low, high = min(low, length), max(high, length)
            end += 1
        group = prompts[start:end]
        prepare_batch(model, len(group) * max(map(len, group)))
        outputs.append(prefill_logits(model, group, cache, slot_start + start))
        start = end
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class BatchPrefillEngine(Engine):
    prefill_function = staticmethod(prefill_logits)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.world != 1:
            raise ValueError("Batched Prefill requires EP1")
        if self.requests:
            prepare_batch(self.model, self.batch * max(len(r.prompt) for r in self.local))

    def prefill(self):
        if self.model.trace:
            raise ValueError("Batched Prefill does not support per-request traces")
        first = self.prefill_function(self.model, [r.prompt for r in self.local], self.cache)[:, 0].argmax(-1)
        s = self.state
        s["next"].copy_(first)
        s["length"].copy_(torch.tensor([len(r.prompt) for r in self.local], device=self.device))
        s["generated"].copy_(s["present"])
        active = s["present"] & (s["budget"] > 1)
        if not self.ignore_eos:
            active &= ~self.is_eos(first)
        s["active"].copy_(active)
