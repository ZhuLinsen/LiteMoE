"""Graph-safe attention over each KV head's contiguous cache sequence."""
import torch


class FlashDecode:
    def __init__(self, batch, kv_heads, groups, capacity, device, maximum=None):
        if maximum is not None:
            if not 0 < maximum <= capacity or capacity % 256:
                raise ValueError("Paged attention requires aligned capacity and a valid length bound")
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
            self.paged = (flash_attn_varlen_func, torch.arange(batch + 1, device=device, dtype=torch.int32),
                          torch.arange(batch, device=device, dtype=torch.int32)[:, None], maximum)
        offsets = torch.arange(batch * kv_heads + 1, device=device, dtype=torch.int32)
        self.layouts = {(batch, kv_heads, groups, capacity): (offsets * groups, offsets * capacity)}

    def __call__(self, q, k, v, lengths, active, scale, used=None):
        if hasattr(self, "paged"):
            attend, cuq, pages, maximum = self.paged
            return attend(q.squeeze(2), k.transpose(1, 2), v.transpose(1, 2),
                max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=maximum,
                seqused_k=torch.where(active, lengths + 1, 1).to(torch.int32) if used is None else used, block_table=pages,
                softmax_scale=scale, causal=False, fa_version=2, num_splits=0).unsqueeze(2)
        b, h, c, d = k.shape
        g = q.shape[1] // h
        key = (b, h, g, c)
        if key not in self.layouts:
            offsets = torch.arange(b * h + 1, device=k.device, dtype=torch.int32)
            self.layouts[key] = (offsets * g, offsets * c)
        cuq, cuk = self.layouts[key]
        used = torch.where(active, lengths + 1, 1).to(torch.int32).repeat_interleave(h)
        return torch.ops.aten._flash_attention_forward(
            q.reshape(b*h*g, 1, d), k.reshape(b*h*c, 1, d), v.reshape(b*h*c, 1, d),
            cuq, cuk, g, c, 0.0, False, False, scale=scale,
            seqused_k=used)[0].reshape(b, h*g, 1, d)
