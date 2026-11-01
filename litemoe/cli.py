"""torchrun-compatible entrypoint; no server, scheduler, or hidden generation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from .engine import Engine, Request
from .ep import ExpertParallel, agree
from .model import Granite
from .ops import Cublas, Grouped, Reference
from .placement import Placement, digest, uniform
from .weights import TensorSource, checkpoint, load_config, load_weights
from .vllm_moe import VllmFusedExperts
def build(model, revision, requests, world=1, rank=0, device="cuda", dtype=torch.bfloat16,
          backend="reference", placement=None, mode="eager", ep_combine="auto", **engine_options):
    if ep_combine not in ("auto", "partial", "routes"):
        raise ValueError("Unknown EP combine policy")
    if backend in ("optimized", "vllm") and world != 1:
        raise ValueError("Optimized execution currently requires EP1")
    if ep_combine == "auto":
        ep_combine = "routes" if world > 1 and backend in ("reference", "cublas") else "partial"
    path = checkpoint(model, revision)
    cfg = load_config(path)
    placement = placement or uniform(cfg.num_hidden_layers, cfg.num_local_experts, world, model=model, revision=revision)
    placement.validate(cfg.num_hidden_layers, cfg.num_local_experts, world)
    if placement.model and placement.model != model or placement.revision and placement.revision != revision:
        raise ValueError("Placement belongs to a different checkpoint/revision")
    if world > 1:
        agree(dict(model=model, revision=revision, config=cfg.to_dict(), placement=placement.__dict__,
                   backend=backend, mode=mode, dtype=str(dtype), world=world, ep_combine=ep_combine,
                   engine={k: v for k, v in engine_options.items() if k != "host_process"}), device)
    capacity = max([len(r.prompt) for r in requests if r.budget] + [(len(requests) + world - 1) // world, 1024])
    if backend not in ("reference", "fast", "cublas", "optimized", "vllm"):
        raise ValueError("Unknown expert backend")
    if backend == "vllm":
        op = VllmFusedExperts()
    else:
        op = Reference() if backend == "reference" else Cublas() if backend == "cublas" else Grouped(
            world * capacity, cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts,
            cfg.num_experts_per_tok, dtype, torch.device(device),
        )
    ep = ExpertParallel(world, rank, capacity, cfg.hidden_size, cfg.num_experts_per_tok, dtype, device, op, ep_combine)
    weights = load_weights(cfg, TensorSource(path), placement, rank, device, dtype, backend)
    model_obj = Granite(cfg, weights, ep, 4096)
    if backend == "vllm":
        model_obj.prepare_packed_qkv()
        model_obj.decode_packed_qkv = True
    engine_type = Engine
    if backend in ("optimized", "vllm"):
        from .optimized import OptimizedEngine
        op.fused_combine = True
        engine_type = OptimizedEngine
    return engine_type(model_obj, requests, world, rank, mode=mode, **engine_options), placement, path


def environment():
    versions = {}
    for package in ("torch", "transformers", "safetensors", "huggingface-hub", "deepspeed", "triton", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return dict(python=platform.python_version(), platform=platform.platform(), dependencies=versions,
                cuda=torch.version.cuda, nccl=torch.cuda.nccl.version() if torch.cuda.is_available() else None,
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], mini_commit=commit,
                mini_source_sha256=source_digest())


def source_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def append_result(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--input", required=True, help="JSON list: request_id, prompt token ids (or text), budget")
    p.add_argument("--placement")
    p.add_argument("--backend", choices=["reference", "fast", "cublas", "optimized", "vllm"], default="reference")
    p.add_argument("--mode", choices=["eager", "graph", "graph-async"], default="eager")
    p.add_argument("--layout", choices=["ep", "dp"], default="ep")
    p.add_argument("--ep-combine", choices=["auto", "partial", "routes"], default="auto",
                   help="auto: source-sized route return for reference EP2; routes uses K times partial return traffic")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--poll", type=int, default=1)
    p.add_argument("--capacity", type=int, help="Explicit KV capacity for matched ablations")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--results", default="results/results.jsonl")
    p.add_argument("--trace")
    p.add_argument("--raw-trace", action="store_true")
    p.add_argument("--skip-uninitialized-fill", action="store_true",
                   help="Keep deterministic algorithms; omit redundant empty-tensor debug fills")
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        p.error("warmup must be nonnegative and repeats positive")
    if args.trace and (args.mode != "eager" or args.repeats != 1):
        p.error("Trace collection requires eager mode and --repeats 1 to preserve step boundaries")
    if args.backend == "fast":
        p.error("The experimental fast adapter failed real-checkpoint numerical admission; use --backend vllm")
    if args.backend in ("optimized", "vllm") and args.trace:
        p.error("Optimized batched Prefill does not support per-request traces; use reference")
    if args.mode != "eager" and args.backend not in ("cublas", "optimized", "vllm"):
        p.error("Graph modes require --backend cublas, optimized or vllm")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = not args.skip_uninitialized_fill
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.backend in ("optimized", "vllm") and world != 1:
        p.error("Optimized execution currently requires a single GPU / EP1")
    if world > 1 and args.layout == "ep" and args.ep_combine == "partial":
        p.error("BF16 partial EP2 failed checkpoint admission; use --ep-combine routes")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CLI requires a CUDA GPU with native BF16 support")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    row = dict(vars(args), **environment(), world=world, rank=rank, dtype="bfloat16", status="started",
               instrumented=bool(args.trace), deterministic=True)
    if args.backend in ("optimized", "vllm"):
        row["numerical_contract"] = "EP1 quality-screened candidate; not bitwise HF admission"
    engine = None
    try:
        from transformers import AutoTokenizer
        data = json.loads(Path(args.input).read_text(encoding="utf-8")) if rank == 0 else None
        if world > 1:
            objects = [data]
            dist.broadcast_object_list(objects, src=0)
            data = objects[0]
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        requests = [Request(str(r.get("request_id", i)), r["prompt"] if "prompt" in r else tokenizer.encode(r["text"]), r["budget"])
                    for i, r in enumerate(data)]
        placement = Placement.load(args.placement) if args.placement else None
        # DP baseline: separate EP1 cores; global wall time is aggregated below.
        core_world, core_rank = (1, 0) if args.layout == "dp" else (world, rank)
        local_requests = requests[rank::world] if args.layout == "dp" else requests
        engine, placement, _ = build(args.model, args.revision, local_requests, core_world, core_rank, device,
                                     backend=args.backend, placement=placement, mode=args.mode, ep_combine=args.ep_combine,
                                     ignore_eos=args.ignore_eos, poll=args.poll, capacity=args.capacity, host_process=tokenizer.decode)
        row.update(ep_combine_effective=engine.model.ep.combine,
                   ep_return_elements_per_token=0 if core_world == 1 else engine.model.cfg.hidden_size * (
                       engine.model.cfg.num_experts_per_tok if engine.model.ep.combine == "routes" else 1),
                   placement_digest=placement.sha256, input_digest=digest(data), batch=len(requests),
                   lengths=[len(r.prompt) for r in requests], budgets=[r.budget for r in requests])
        for _ in range(args.warmup):
            engine.generate()
        if args.trace:
            if args.mode != "eager":
                raise ValueError("Trace must run separately in eager mode")
            from .trace import RouteTrace
            engine.model.trace = RouteTrace(engine.model.cfg.num_hidden_layers, engine.model.cfg.num_local_experts, rank, args.raw_trace)
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            if args.layout == "dp" and world > 1:
                dist.barrier()
            outputs, metrics = engine.generate()
            if args.layout == "dp" and world > 1:
                times = torch.tensor([metrics[k] for k in ("prefill_s", "decode_s", "e2e_s")], device=device)
                counts = torch.tensor([metrics[k] for k in ("input_tokens", "output_tokens", "decode_tokens")], device=device)
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                metrics.update(zip(("prefill_s", "decode_s", "e2e_s"), times.tolist()))
                metrics.update(zip(("input_tokens", "output_tokens", "decode_tokens"), counts.tolist()))
                metrics["decode_tps"] = metrics["decode_tokens"] / metrics["decode_s"] if metrics["decode_tokens"] else None
                metrics["e2e_tps"] = metrics["output_tokens"] / metrics["e2e_s"] if metrics["e2e_s"] else None
            result = dict(row, **metrics, status="ok", repeat=repeat, outputs=outputs,
                          allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved())
            append_result(args.results if world == 1 else f"{args.results}.rank{rank}", result)
            if rank == 0:
                print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.trace:
            engine.model.trace.save(f"{args.trace}.rank{rank}.json", model=args.model, revision=args.revision,
                                    input_digest=digest(data), world=world)
    except Exception as error:
        append_result(args.results + f".rank{rank}", dict(row, status="error", error=repr(error)))
        raise
    finally:
        # NCCL graph resources must be released before communicator destruction.
        if engine is not None and engine.graph is not None:
            engine.synchronize()
            engine.graph.reset()
            engine.graph = None
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
