"""Granite forward wiring, independent of HF's forward/generate implementations."""

import torch
import torch.nn.functional as F
from .ops import linear_rows
from .packed import PackedPrefill
from .vllm_moe import VllmFusedExperts

def rmsnorm(x, weight, eps):
    z = x.float()
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)

class Granite:
    def __init__(self, cfg, weights, ep, capacity):
        self.cfg, self.w, self.ep = cfg, weights, ep
        self.trace = None
        self.native_gqa, self.decode_kv_limit = False, None
        self.decode_folded_q = False
        self.decode_batched_linears = False
        self.decode_packed_qkv = False
        self.decode_fused_norm = False
        self.decode_flash = None
        self.decode_lean = False
        self.prefill_lean = False
        self.used = None
        device, dtype = weights["embedding"].device, weights["embedding"].dtype
        dim = cfg.hidden_size // cfg.num_attention_heads
        # Match HF's CPU initialization: GPU pow can shift long-position BF16 RoPE.
        inv = 1.0 / cfg.rope_theta ** (torch.arange(0, dim, 2, device="cpu").float() / dim)
        freq = torch.outer(torch.arange(capacity, device=device).float(), inv.to(device))
        self.cos_sin = torch.cat((freq.cos(), freq.sin()), -1).to(dtype)
        self.cos, self.sin = [half.repeat(1, 2) for half in self.cos_sin.chunk(2, -1)]
    def prepare_packed_qkv(self):
        for w in self.w["layers"]:
            if "self_attn.qkv_proj" not in w:
                w["self_attn.qkv_proj"] = torch.cat([w[f"self_attn.{p}_proj"] for p in ("q", "k", "v")])

    def attention(self, z, w, cache, layer, positions, valid, slot):
        cfg = self.cfg
        b, s, _ = z.shape
        d = cfg.hidden_size // cfg.num_attention_heads
        single = slot is None and not self.decode_batched_linears
        if slot is None and self.decode_packed_qkv:
            sizes = [cfg.num_attention_heads * d, cfg.num_key_value_heads * d, cfg.num_key_value_heads * d]
            q, k, v = linear_rows(z, w["self_attn.qkv_proj"]).split(sizes, dim=-1)
        else:
            q, k, v = [linear_rows(z, w[f"self_attn.{p}_proj"], single) for p in ("q", "k", "v")]
        q = q.view(b, s, cfg.num_attention_heads, d).transpose(1, 2)
        k = k.view(b, s, cfg.num_key_value_heads, d).transpose(1, 2)
        v = v.view(b, s, cfg.num_key_value_heads, d).transpose(1, 2)
        lean = slot is None and self.decode_lean  # decode_step already zeroes inactive positions.
        if isinstance(slot, PackedPrefill) and self.prefill_lean:  # Packed rows have no padding.
            from vllm import _custom_ops as ops
            ops.rotary_embedding(positions[0], q[0].transpose(0, 1), k[0].transpose(0, 1), d, self.cos_sin, True)
            cache.write(layer, k.permute(2, 1, 0, 3), v.permute(2, 1, 0, 3), slot.positions, valid[0], slot.slots)
            return linear_rows(slot.attention(q, k, v, cfg.attention_multiplier).reshape(b, s, cfg.hidden_size), w["self_attn.o_proj"])
        safe = positions if lean else torch.where(valid, positions, 0)
        if slot is None and q.is_cuda and self.decode_flash is not None and isinstance(self.ep.backend, VllmFusedExperts):
            from vllm import _custom_ops as ops
            ops.rotary_embedding(safe[:, 0], q.squeeze(2), k.squeeze(2), d, self.cos_sin, True)
        else:
            cos, sin = self.cos[safe][:, None], self.sin[safe][:, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if slot is None:
            if self.decode_flash is not None and q.is_cuda:
                cache.write(layer, k, v, safe[:, 0], valid[:, 0])
                a = self.decode_flash(q, cache.k[layer], cache.v[layer], positions[:, 0], valid[:, 0], cfg.attention_multiplier, self.used)
                a = linear_rows(a.transpose(1, 2).reshape(b, s, cfg.hidden_size), w["self_attn.o_proj"], single)
                return a if lean else a * valid[..., None]
            k, v, mask = cache.decode(layer, k, v, positions[:, 0], valid[:, 0])
            if self.decode_kv_limit is not None:
                k, v, mask = k[:, :, :self.decode_kv_limit], v[:, :, :self.decode_kv_limit], mask[..., :self.decode_kv_limit]
        else:
            k, v, mask = cache.prefill(layer, slot, k, v, valid)
        if isinstance(slot, PackedPrefill):
            a = slot.attention(q, k, v, cfg.attention_multiplier).reshape(b, s, cfg.hidden_size)
            return linear_rows(a, w["self_attn.o_proj"]) * valid[..., None]
        # Explicit GQA expansion also works on SDPA's math fallback and older GPUs.
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        folded = slot is None and self.decode_folded_q
        if folded:
            # Decode has one query position: each shared-KV head's queries can be
            # separate query rows, preserving head order and the same key mask.
            q = q.reshape(b, cfg.num_key_value_heads, groups, d)
        elif not self.native_gqa:
            k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                           dropout_p=0.0, scale=cfg.attention_multiplier, enable_gqa=self.native_gqa and not folded)
        a = a.reshape(b, cfg.num_attention_heads, s, d).transpose(1, 2).reshape(b, s, cfg.hidden_size)
        return linear_rows(a, w["self_attn.o_proj"], single) * valid[..., None]
    def forward(self, tokens, positions, valid, cache, slot=None, last=None, communication_tokens=None):
        cfg = self.cfg
        norm = rmsnorm
        fast_prefill = isinstance(slot, PackedPrefill) and self.prefill_lean
        if (slot is None and self.decode_fused_norm and tokens.is_cuda) or fast_prefill:
            from .norm import rmsnorm as norm
        single = slot is None and not self.decode_batched_linears
        fused = fast_prefill or (slot is None and self.decode_lean)
        self.used = (positions[:, 0] + 1).to(torch.int32) if slot is None and self.decode_lean else None
        x = F.embedding(tokens, self.w["embedding"]) * cfg.embedding_multiplier
        for layer, w in enumerate(self.w["layers"]):
            z = norm(x, w["input_layernorm"], cfg.rms_norm_eps)
            a = self.attention(z, w, cache, layer, positions, valid, slot)
            x = x.add(a, alpha=cfg.residual_multiplier) if fused else x + cfg.residual_multiplier * a
            z = norm(x, w["post_attention_layernorm"], cfg.rms_norm_eps).reshape(-1, cfg.hidden_size)
            scores = linear_rows(z, w["block_sparse_moe.router.layer"], single)
            if fast_prefill or (slot is None and self.decode_batched_linears and z.is_cuda and isinstance(self.ep.backend, VllmFusedExperts)):
                from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
                scales, ids, _ = fused_topk(z, scores, cfg.num_experts_per_tok, renormalize=True)
            else:
                top, ids = scores.float().topk(cfg.num_experts_per_tok, dim=-1)
                scales = top.softmax(-1)
            if not fused:  # vLLM experts consume FP32 scales directly.
                scales = scales.to(z.dtype)
            if self.trace is not None:
                self.trace.record(layer, ids, valid.reshape(-1))
            y = self.ep(z, ids, scales, valid.reshape(-1), w["physical"], w["input_linear"], w["output_linear"], w["logical"], decode=slot is None or fast_prefill, tokens=communication_tokens)
            x = x.add(y.view_as(x), alpha=cfg.residual_multiplier) if fused else x + cfg.residual_multiplier * y.view_as(x)
        if last is not None:
            x = (x[torch.arange(x.shape[0], device=x.device), last][:, None]
                 if isinstance(last, torch.Tensor) else x[:, last:last + 1])
        x = norm(x, self.w["norm"], cfg.rms_norm_eps)
        return linear_rows(x, self.w["head"], single) / cfg.logits_scaling
