"""Record one greedy request on the eager reference path, for tools/build_visualizer.py.

  python tools/record_inference.py "Explain what a GPU does in one sentence." --output inference.json

The default fast path can't be traced, and its rounding differs, so its text may differ.
"""
import argparse
import json
from pathlib import Path

import torch

from litemoe.cli import build
from litemoe.engine import Request
from litemoe.generate import MODELS


class Recorder:
    def __init__(self):
        self.routes = []

    def context(self, phase, step, slot):
        pass

    def record(self, layer, ids, valid):
        rows = ids.reshape(-1, ids.shape[-1])[valid.reshape(-1)].tolist()
        if layer == 0:
            self.routes.extend([] for _ in rows)
        for route, row in zip(self.routes[-len(rows):], rows):
            route.append(row)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt")
    p.add_argument("--model", choices=MODELS, default="granite-1b")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    name, revision = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    text = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)], add_generation_prompt=True, tokenize=False)
    prompt = tokenizer.encode(text, add_special_tokens=False)
    engine, _, _ = build(name, revision, [Request("0", prompt, args.max_tokens)], device=args.device)
    model, recorder, candidates = engine.model, Recorder(), []
    model.trace = recorder
    forward = model.forward

    def recorded(*a, **k):
        logits = forward(*a, **k)
        top, ids = logits[0, -1].float().softmax(-1).topk(5)
        candidates.append([dict(text=tokenizer.decode([i]), p=round(v, 4)) for i, v in zip(ids.tolist(), top.tolist())])
        return logits

    model.forward = recorded
    outputs, _ = engine.generate()
    generated = outputs["0"]
    cfg = model.cfg
    Path(args.output).write_text(json.dumps(dict(
        model=name, revision=revision, device=args.device, user_prompt=args.prompt,
        prompt=[tokenizer.decode([t]) for t in prompt], output=[tokenizer.decode([t]) for t in generated],
        text=tokenizer.decode(generated, skip_special_tokens=True), candidates=candidates, routes=recorder.routes,
        config=dict(layers=cfg.num_hidden_layers, experts=cfg.num_local_experts, experts_per_token=cfg.num_experts_per_tok,
                    hidden=cfg.hidden_size, q_heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads),
    ), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
