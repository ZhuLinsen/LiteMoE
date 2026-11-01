"""Task-level check on GSM8K: LiteMoE against vLLM and HF on the same prompts, greedy.

  python tools/quality.py run --backend litemoe --model granite-1b --output 1b-litemoe.json
  python tools/quality.py compare 1b-litemoe.json 1b-vllm.json 1b-hf.json
"""
import argparse
import json
import math
import re
from pathlib import Path

BATCH, BUDGET = 16, 256
MIN_CI_LOWER = -0.03  # Paired accuracy difference, fixed before the first run.
SUFFIX = " Think step by step, then finish with 'Answer: <number>'."


def problems(tokenizer, count):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()[:count]
    prompts = [tokenizer.encode(tokenizer.apply_chat_template([dict(role="user", content=r["question"] + SUFFIX)],
               add_generation_prompt=True, tokenize=False), add_special_tokens=False) for r in rows]
    return prompts, [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]


def extract(text):
    tail = text.rsplit("Answer", 1)[-1] if "Answer" in text else text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def same_number(a, b):
    try:
        return a is not None and float(a) == float(b)
    except ValueError:
        return False


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def litemoe_runner(name, revision):
    import torch
    from litemoe.cli import build
    from litemoe.engine import Request

    @torch.inference_mode()
    def run(prompts):
        requests = [Request(str(i), p, BUDGET) for i, p in enumerate(prompts)]
        engine, _, _ = build(name, revision, requests, backend="vllm", mode="graph-async")
        outputs, _ = engine.generate()
        result = [outputs[str(i)] for i in range(len(prompts))]
        del engine
        torch.cuda.empty_cache()
        return result
    return run


def vllm_runner(name, revision):
    from vllm import LLM, SamplingParams
    llm = LLM(model=name, revision=revision, dtype="bfloat16", max_model_len=2048, max_num_seqs=BATCH,
              gpu_memory_utilization=.6, enable_prefix_caching=False, seed=0, disable_log_stats=True)
    sampling = SamplingParams(temperature=0, max_tokens=BUDGET, seed=0, detokenize=False)

    def run(prompts):
        result = llm.generate([dict(prompt_token_ids=p) for p in prompts], sampling, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in result]
    return run


def hf_runner(name, revision, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def run(prompts):
        width = max(map(len, prompts))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts]).cuda()
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts]).cuda()
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, temperature=None, top_p=None,
                             top_k=None, max_new_tokens=BUDGET, pad_token_id=pad)[:, width:].tolist()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        return [row[:next((i + 1 for i, t in enumerate(row) if t in eos), len(row))] for row in out]
    return run


def run(args):
    from transformers import AutoTokenizer
    from litemoe.generate import MODELS
    name, revision = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    prompts, gold = problems(tokenizer, args.count)
    runner = dict(litemoe=lambda: litemoe_runner(name, revision), vllm=lambda: vllm_runner(name, revision),
                  hf=lambda: hf_runner(name, revision, tokenizer))[args.backend]()
    tokens = [row for batch in chunks(prompts, BATCH) for row in runner(batch)]
    probe = prompts[:BATCH]
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    Path(args.output).write_text(json.dumps(dict(
        backend=args.backend, model=args.model, revision=revision, gold=gold, tokens=tokens, texts=texts,
        correct=[same_number(extract(t), g) for t, g in zip(texts, gold)],
        repeat_identical=runner(probe) == tokens[:BATCH],
        solo_identical=[runner([p])[0] == t for p, t in zip(probe, tokens)],
        eos_ids=sorted(set(eos if isinstance((eos := tokenizer.eos_token_id), list) else [eos])),
    )))


def compare(args):
    runs = [json.loads(Path(p).read_text()) for p in args.files]
    base = runs[0]
    bad = [i for i, t in enumerate(base["tokens"]) if not t or len(t) > BUDGET
           or any(x in base["eos_ids"] for x in t[:-1])]
    for r in runs:  # Backends differ on whether the final EOS is reported.
        r["tokens"] = [t[:-1] if t and t[-1] in base["eos_ids"] else t for t in r["tokens"]]
    failed = False
    print(f"{base['model']} {base['backend']}: accuracy {sum(base['correct']) / len(base['correct']):.1%} on "
          f"{len(base['correct'])}; repeat identical {base['repeat_identical']}; "
          f"solo identical {sum(base['solo_identical'])}/{len(base['solo_identical'])}; structural failures {len(bad)}")
    failed |= bool(bad) or not base["repeat_identical"]
    for other in runs:
        n = len(other["correct"])
        print(f"  {other['backend']:8} accuracy {sum(other['correct']) / n:.1%} "
              f"mean tokens {sum(map(len, other['tokens'])) / n:.1f} repeat identical {other['repeat_identical']} "
              f"solo identical {sum(other['solo_identical'])}/{len(other['solo_identical'])}")
    for other in runs[1:]:
        n = len(base["correct"])
        diff = [int(a) - int(b) for a, b in zip(base["correct"], other["correct"])]
        mean = sum(diff) / n
        ci = 1.96 * math.sqrt(max(sum((d - mean) ** 2 for d in diff) / (n - 1), 0) / n)
        same = sum(a == b for a, b in zip(base["tokens"], other["tokens"]))
        ok = mean - ci >= MIN_CI_LOWER
        failed |= not ok
        print(f"  vs {other['backend']:8} diff {mean:+.1%} (95% CI {mean - ci:+.1%} to {mean + ci:+.1%}) "
              f"identical outputs {same}/{n} {'PASS' if ok else 'FAIL'}")
    print("RESULT", "FAIL" if failed else "PASS")
    raise SystemExit(failed)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(required=True)
    r = commands.add_parser("run")
    r.add_argument("--backend", choices=["litemoe", "vllm", "hf"], required=True)
    r.add_argument("--model", choices=["granite-1b", "granite-3b"], required=True)
    r.add_argument("--count", type=int, default=500)
    r.add_argument("--output", required=True)
    r.set_defaults(func=run)
    c = commands.add_parser("compare")
    c.add_argument("files", nargs="+", help="LiteMoE result first, then references")
    c.set_defaults(func=compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
