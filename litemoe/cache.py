"""Contiguous rank-local KV, safe masked writes, no paging or KV collectives."""

import torch
from .packed import PackedPrefill


class KVCache:
    def __init__(self, cfg, batch, capacity, device, dtype):
        self.capacity = capacity
        h, d = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        shape = (cfg.num_hidden_layers, batch, h, capacity, d)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.keys = torch.arange(capacity, device=device)
        self.slots = torch.arange(batch, device=device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()

    def prefill(self, layer, slot, k, v, valid):
        if isinstance(slot, PackedPrefill):
            self.k[layer, slot.slots, :, slot.positions] = k[0].transpose(0, 1)
            self.v[layer, slot.slots, :, slot.positions] = v[0].transpose(0, 1)
            return k, v, None
        s = k.shape[2]
        slots = slot if isinstance(slot, slice) else slice(slot, slot + 1)
        self.k[layer, slots, :, :s].copy_(k * valid[:, None, :, None])
        self.v[layer, slots, :, :s].copy_(v * valid[:, None, :, None])
        pos = torch.arange(s, device=k.device)
        mask = (pos[:, None] >= pos[None, :])[None, None] & valid[:, None, None, :]
        mask = torch.where(valid[:, None, :, None], mask, (pos == 0)[None, None, None, :])
        return k, v, mask

    def write(self, layer, k, v, pos, active):
        kc, vc = self.k[layer], self.v[layer]
        old_k, old_v = kc[self.slots, :, pos], vc[self.slots, :, pos]
        index = pos[:, None, None, None].expand_as(k)
        kc.scatter_(2, index, torch.where(active[:, None, None, None], k, old_k[:, :, None]))
        vc.scatter_(2, index, torch.where(active[:, None, None, None], v, old_v[:, :, None]))

    def decode(self, layer, k, v, lengths, active):
        pos = torch.where(active, lengths, 0)
        self.write(layer, k, v, pos, active)
        mask = self.keys[None, :] < (lengths + 1)[:, None]
        mask = torch.where(active[:, None], mask, self.keys[None, :] == 0)
        return self.k[layer], self.v[layer], mask[:, None, None, :]


class FusedKVCache(KVCache):
    def write(self, layer, k, v, pos, active, slots=None):
        if not k.is_cuda:
            return super().write(layer, k, v, pos, active)
        from .kv_write import write
        return write(self, layer, k, v, pos, active, self.slots if slots is None else slots)
