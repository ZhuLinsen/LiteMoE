import os
import json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from litemoe.ep import ExpertParallel
from litemoe.model import Granite
from litemoe.ops import Reference
from litemoe.placement import uniform
from litemoe.weights import load_weights, load_config


def pytest_sessionstart(session):
    torch.set_num_threads(1)


def checkpoint_state(hf):
    """Export tiny HF4/HF5 weights to the original Granite checkpoint schema."""
    return {key.replace(".router.weight", ".router.layer.weight")
               .replace(".experts.gate_up_proj", ".input_linear.weight")
               .replace(".experts.down_proj", ".output_linear.weight"): value
            for key, value in hf.state_dict().items()}


def save_tiny_checkpoint(hf, mini, path):
    from safetensors.torch import save_file
    hf.config.save_pretrained(path)
    (path / "config.json").write_text(json.dumps(mini.cfg.to_dict()))
    save_file(checkpoint_state(hf), str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path):
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
    torch.manual_seed(7)
    cfg = GraniteMoeConfig(vocab_size=47, hidden_size=32, intermediate_size=24, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                          num_experts_per_tok=2, max_position_embeddings=64, eos_token_id=0,
                          embedding_multiplier=3.0, attention_multiplier=0.125, residual_multiplier=0.3,
                          logits_scaling=2.0, rope_theta=10000.0)
    cfg._attn_implementation = "eager"
    hf = GraniteMoeForCausalLM(cfg).eval()
    state = checkpoint_state(hf)

    class Source:
        def get(self, key, expert=None):
            value = state[key]
            return value if expert is None else value[expert]

    # Preserve the explicitly declared fixture semantics through the production loader.
    raw = cfg.to_dict()
    raw.update(rope_theta=10000.0, rope_scaling=None)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = load_config(tmp_path)
    placement = uniform(2, 4, 1)
    weights = load_weights(cfg, Source(), placement, 0, "cpu", torch.float32)
    ep = ExpertParallel(1, 0, 64, 32, 2, torch.float32, "cpu", Reference())
    return hf, Granite(cfg, weights, ep, 64), Source()
