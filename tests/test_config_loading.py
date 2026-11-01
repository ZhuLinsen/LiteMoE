from conftest import save_tiny_checkpoint
"""Checkpoint boundary checks plus an actual saved-HF-checkpoint CPU round trip."""
import copy
import json

import pytest
import torch
from transformers import AutoConfig

from litemoe.cli import build
from litemoe.engine import Request
from litemoe.weights import load_config


def test_saved_checkpoint_build_preserves_generation(tiny, tmp_path):
    hf, mini, _ = tiny
    save_tiny_checkpoint(hf, mini, tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.rope_theta == mini.cfg.rope_theta
    assert cfg.rope_scaling is None
    requests = [Request('a', [2, 4, 6], 5), Request('b', [3, 7], 3)]
    engine, _, _ = build(str(tmp_path), '', requests, device='cpu', dtype=torch.float32, ignore_eos=True)
    actual, _ = engine.generate()
    for r in requests:
        tokens = torch.tensor([r.prompt])
        expected = []
        with torch.inference_mode():
            for _ in range(r.budget):
                token = hf(tokens).logits[:, -1].argmax(-1, keepdim=True)
                expected.append(token.item())
                tokens = torch.cat((tokens, token), dim=1)
        assert actual[r.request_id] == expected
    exported = cfg.to_dict()
    exported['rope_theta'] = 1
    assert cfg.rope_theta == mini.cfg.rope_theta


@pytest.mark.parametrize('encoding', ['legacy', 'default-parameters'])
def test_config_encoding_keeps_original_semantics(tiny, tmp_path, monkeypatch, encoding):
    hf, mini, _ = tiny
    (tmp_path/"config.json").write_text(json.dumps(mini.cfg.to_dict()))
    parsed = mini.cfg.to_dict()
    if encoding == 'default-parameters':
        theta = parsed.pop('rope_theta')
        parsed['rope_parameters'] = {'rope_type': 'default', 'rope_theta': theta}
        parsed['rope_scaling'] = parsed['rope_parameters'].copy()
    before = copy.deepcopy(parsed)
    class Parsed:
        def to_dict(self):
            return copy.deepcopy(parsed)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: Parsed())
    result = load_config(tmp_path)
    assert result.rope_theta == mini.cfg.rope_theta and result.rope_scaling is None
    assert parsed == before


@pytest.mark.parametrize('case', ['scaled', 'theta', 'extra', 'nan', 'missing', 'bool', 'bias', 'conflict', 'legacy-conflict'])
def test_loading_rejects_unsupported_or_changed_config(tiny, tmp_path, monkeypatch, case):
    hf, mini, _ = tiny
    (tmp_path/"config.json").write_text(json.dumps(mini.cfg.to_dict()))
    raw = json.loads((tmp_path/'config.json').read_text())
    parsed = mini.cfg.to_dict()
    parsed['rope_parameters'] = {'rope_type': 'default', 'rope_theta': raw['rope_theta']}
    if case == 'scaled': raw['rope_scaling'] = {'rope_type': 'linear', 'factor': 2}
    if case == 'theta': parsed['rope_parameters']['rope_theta'] *= 2
    if case == 'extra': parsed['rope_parameters']['factor'] = 2
    if case == 'nan': raw['rope_theta'] = float('nan')
    if case == 'missing': del raw['rope_theta']
    if case == 'bool': raw['rope_theta'] = True
    if case == 'bias': parsed['attention_bias'] = True
    if case == 'conflict': parsed['rope_scaling'] = {'rope_type': 'linear', 'factor': 2}
    if case == 'legacy-conflict': parsed['rope_theta'] *= 2
    (tmp_path/'config.json').write_text(json.dumps(raw))
    class Parsed:
        def to_dict(self):
            return copy.deepcopy(parsed)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: Parsed())
    with pytest.raises(ValueError):
        load_config(tmp_path)
