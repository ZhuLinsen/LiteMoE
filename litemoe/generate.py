"""litemoe "your prompt": greedy generation with the optimized one-GPU engine."""
import argparse

import torch

from .cli import build
from .engine import Request

MODELS = {
    "granite-1b": ("ibm-granite/granite-3.1-1b-a400m-instruct", "0da7a48b0276d500ce5922fd2b33944091fc6c09"),
    "granite-3b": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89"),
}
CONTEXT_LIMIT = 4096


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt")
    p.add_argument("--model", choices=MODELS, default="granite-1b")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    if args.max_tokens < 1:
        p.error("--max-tokens must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("LiteMoE requires an NVIDIA GPU with BF16 support")
    from transformers import AutoTokenizer
    name, revision = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    prompt = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)], add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) + args.max_tokens - 1 > CONTEXT_LIMIT:
        p.error(f"Prompt ({len(tokens)} tokens) plus --max-tokens exceeds the {CONTEXT_LIMIT}-token context limit")
    engine, _, _ = build(name, revision, [Request("0", tokens, args.max_tokens)], backend="vllm", mode="graph-async", capacity=4096)
    outputs, metrics = engine.generate()
    print(tokenizer.decode(outputs["0"], skip_special_tokens=True))
    if metrics["decode_tps"]:
        print(f"\n{metrics['output_tokens']} tokens, decode {metrics['decode_tps']:.1f} tok/s")


if __name__ == "__main__":
    main()
