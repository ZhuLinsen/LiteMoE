"""CPU safetensors slicing; only owned expert slices reach this rank's GPU."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


class ModelConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def load_config(path):
    """Keep checkpoint RoPE semantics across Transformers config encodings."""
    from transformers import AutoConfig
    raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    values = AutoConfig.from_pretrained(path, local_files_only=True).to_dict()
    theta = raw.get("rope_theta")
    expected = {"rope_type": "default", "rope_theta": theta}
    if (raw.get("model_type") != "granitemoe" or raw.get("rope_scaling") is not None
            or isinstance(theta, bool) or not isinstance(theta, (int, float))
            or not math.isfinite(theta) or theta <= 0
            or raw.get("rope_parameters", expected) != expected):
        raise ValueError("Checkpoint must declare unscaled Granite RoPE with a finite positive theta")
    if "rope_parameters" in values:
        if (values.pop("rope_parameters") != expected or values.get("rope_scaling") not in (None, expected)
                or values.get("rope_theta", theta) != theta):
            raise ValueError("Parsed RoPE differs from the original checkpoint")
    elif values.get("rope_theta") != theta or values.get("rope_scaling") is not None:
        raise ValueError("Parsed RoPE differs from the original checkpoint")
    values.update(rope_theta=theta, rope_scaling=None)
    cfg = ModelConfig(**values)
    validate_config(cfg)
    return cfg


class TensorSource:
    def __init__(self, path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(str(self.path / "model.safetensors"), framework="pt", device="cpu") as f:
                self.index = {key: "model.safetensors" for key in f.keys()}

    def get(self, key, expert=None):
        from safetensors import safe_open
        with safe_open(str(self.path / self.index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key) if expert is None else f.get_slice(key)[expert]


def checkpoint(model, revision):
    from huggingface_hub import snapshot_download
    if Path(model).is_dir():
        return Path(model)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Remote models require an immutable 40-character revision")
    return Path(snapshot_download(model, revision=revision, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*", "vocab*", "merges*",
    ]))


def validate_config(cfg):
    if cfg.model_type != "granitemoe" or cfg.hidden_act != "silu":
        raise ValueError("This adapter implements Granite MoE / SwiGLU only")
    if cfg.attention_bias or cfg.rope_scaling is not None:
        raise ValueError("Attention bias and scaled RoPE require a separate validated adapter")
    if cfg.hidden_size % cfg.num_attention_heads or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("Invalid GQA geometry")
    for field in ("embedding_multiplier", "attention_multiplier", "residual_multiplier", "logits_scaling"):
        if not getattr(cfg, field, 0) > 0:
            raise ValueError(f"Invalid checkpoint field {field}")


def load_weights(cfg, source, placement, rank, device, dtype, backend="reference"):
    validate_config(cfg)
    placement.validate(cfg.num_hidden_layers, cfg.num_local_experts, placement.world)
    load = lambda key: source.get(key).to(device=device, dtype=dtype)
    embedding = load("model.embed_tokens.weight")
    layers = []
    for layer in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        weights = {name: load(prefix + name + ".weight") for name in (
            "input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj", "block_sparse_moe.router.layer",
        )}
        ids = placement.logical_on_rank(layer, rank)
        for name, shape in (("input_linear", (2 * cfg.intermediate_size, cfg.hidden_size)),
                            ("output_linear", (cfg.hidden_size, cfg.intermediate_size))):
            target = torch.empty((len(ids), *shape), dtype=dtype, device=device)
            for local, logical in enumerate(ids):
                value = source.get(prefix + "block_sparse_moe." + name + ".weight", expert=logical)
                if tuple(value.shape) != shape:
                    raise ValueError(f"Checkpoint expert geometry mismatch: {name}")
                target[local].copy_(value)
            weights[name] = target
        weights["physical"] = torch.tensor(placement.physical[layer], dtype=torch.int64, device=device)
        # Empty means identity: keep the common uniform path free of extra work.
        reordered = placement.physical[layer] != list(range(cfg.num_local_experts))
        weights["logical"] = weights["physical"].argsort() if reordered else weights["physical"][:0]
        layers.append(weights)
    head = embedding if cfg.tie_word_embeddings else load("lm_head.weight")
    return dict(embedding=embedding, layers=layers, norm=load("model.norm.weight"), head=head)
