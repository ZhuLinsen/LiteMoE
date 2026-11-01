"""Fixed request ownership, coordinated prefill and GPU-owned greedy state."""

import math
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .cache import KVCache
from .ep import agree
from .execution import OutputRing, capture_decode


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    budget: int


class Engine:
    cache_type = KVCache

    def __init__(self, model, requests, world=1, rank=0, mode="eager", ignore_eos=False, poll=1,
                 capacity=None, host_process=None):
        if world not in (1, 2) or not 0 <= rank < world:
            raise ValueError("M0 requires W=1/2 and a valid source rank")
        self.model, self.world, self.rank = model, world, rank
        self.device, self.dtype = model.w["embedding"].device, model.w["embedding"].dtype
        if world > 1:
            agree([r.__dict__ for r in requests], self.device)
        if poll < 1 or mode not in ("eager", "graph", "graph-async"):
            raise ValueError("Invalid execution mode or polling period")
        if any(type(r.budget) is not int or r.budget < 0 for r in requests):
            raise ValueError("Output budgets must be nonnegative integers")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Request ids must be unique")
        self.empty = {r.request_id: [] for r in requests if r.budget == 0}
        self.requests = [r for r in requests if r.budget]
        cfg = model.cfg
        if any(not r.prompt or any(type(t) is not int or not 0 <= t < cfg.vocab_size for t in r.prompt)
               for r in self.requests):
            raise ValueError("Each prompt must contain valid vocabulary ids")
        required = max((len(r.prompt) + r.budget - 1 for r in self.requests), default=1)
        if required > min(4096, cfg.max_position_embeddings):
            raise ValueError("Request exceeds the M0/model context limit")
        cap = capacity or (2 ** math.ceil(math.log2(required)) if mode != "eager" else required)
        if cap < required or cap > min(4096, cfg.max_position_embeddings):
            raise ValueError("Insufficient/unsupported KV capacity")
        self.capacity = cap
        self.batch = math.ceil(len(self.requests) / world)
        self.local = self.requests[rank::world]
        self.mode, self.ignore_eos, self.poll = mode, ignore_eos, poll
        self.host_process = host_process
        self.graph = None
        if not self.requests:
            return
        if mode != "eager" and (self.device.type != "cuda" or not model.ep.backend.graph_safe):
            raise ValueError("Full Decode Graph requires CUDA and a graph-safe expert backend")
        self.cache = self.cache_type(cfg, self.batch, cap, self.device, self.dtype)
        self.state = {name: torch.zeros(self.batch, dtype=dtype, device=self.device) for name, dtype in (
            ("next", torch.int64), ("length", torch.int64), ("generated", torch.int64),
            ("budget", torch.int64), ("present", torch.bool), ("active", torch.bool),
        )}
        self.emit = torch.zeros(self.batch, 3, dtype=torch.int64, device=self.device)
        eos = cfg.eos_token_id
        self.eos = torch.tensor(eos if isinstance(eos, list) else ([] if eos is None else [eos]), device=self.device)
        self.rounds = [max([len(r.prompt) for r in self.requests[j * world:(j + 1) * world]] or [1])
                       for j in range(self.batch)]
        for t in set(self.rounds + [self.batch]):
            model.ep.prepare(t)
        self.reset()

    def reset(self):
        self.cache.reset()
        for value in self.state.values():
            value.zero_()
        for j, r in enumerate(self.local):
            self.state["budget"][j] = r.budget
            self.state["present"][j] = True
        self.emit.zero_()

    def is_eos(self, token):
        return (token[:, None] == self.eos[None, :]).any(-1)

    def decode_step(self):
        s = self.state
        valid = s["active"].clone()
        positions = torch.where(valid, s["length"], 0)[:, None]
        logits = self.model.forward(s["next"][:, None], positions, valid[:, None], self.cache)
        candidate = logits[:, 0].argmax(-1)
        self.emit[:, 0].copy_(candidate)
        self.emit[:, 1].copy_(s["generated"])
        self.emit[:, 2].copy_(valid)
        s["length"].add_(valid)
        s["generated"].add_(valid)
        s["next"].copy_(torch.where(valid, candidate, s["next"]))
        done = s["generated"] >= s["budget"]
        if not self.ignore_eos:
            done = done | self.is_eos(candidate)
        s["active"].copy_(valid & ~done)

    def prepare_graph(self):
        if self.mode == "eager" or self.graph is not None or not self.requests:
            return
        if self.model.trace is not None:
            raise ValueError("Trace/count collection must be disabled during capture/timing")

        def warm_state():
            self.reset()
            self.state["active"].copy_(self.state["present"])
            self.state["budget"].fill_(2)

        self.graph = capture_decode(self.decode_step, warm_state)
        self.reset()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def stopped(self):
        active = self.state["active"].sum(dtype=torch.int64)
        if self.world > 1:
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
        return active.item() == 0

    def prefill(self):
        s = self.state
        for j, length in enumerate(self.rounds):
            r = self.local[j] if j < len(self.local) else None
            prompt = r.prompt if r else []
            local_length = max(len(prompt), 1)
            valid = torch.arange(local_length, device=self.device)[None, :] < len(prompt)
            tokens = torch.tensor([prompt or [0]], device=self.device)
            pos = torch.arange(local_length, device=self.device)[None, :]
            if self.model.trace:
                self.model.trace.context("prefill", j, j)
            logits = self.model.forward(tokens, pos, valid, self.cache, slot=j,
                                        last=max(len(prompt) - 1, 0), communication_tokens=length)
            first = logits[0, 0].argmax(-1)
            s["next"][j].copy_(first)
            s["length"][j] = len(prompt)
            s["generated"][j] = int(r is not None)
            active = s["present"][j] & (s["budget"][j] > 1)
            if not self.ignore_eos:
                active = active & ~self.is_eos(first.reshape(1))[0]
            s["active"][j].copy_(active)

    @torch.inference_mode()
    def generate(self, step_events=None):
        if not self.requests:
            return self.empty, dict(prefill_s=0.0, decode_s=0.0, e2e_s=0.0, input_tokens=0, output_tokens=0,
                                    decode_tokens=0, decode_tps=None, e2e_tps=None)
        self.prepare_graph()
        self.reset()
        outputs = [[] for _ in self.local]

        def consume(rows):
            for slot, (token, index, valid) in enumerate(rows):
                if valid:
                    if slot >= len(outputs) or index != len(outputs[slot]):
                        raise RuntimeError("Lost, duplicated or reordered output snapshot")
                    outputs[slot].append(token)
                    if self.host_process:
                        self.host_process(outputs[slot])

        ring = OutputRing(self.emit, consume, self.mode == "graph-async")
        if self.world > 1:
            dist.barrier()
        self.synchronize()
        start = time.perf_counter()
        self.prefill()
        s = self.state
        self.emit[:, 0].copy_(s["next"])
        self.emit[:, 1].zero_()
        self.emit[:, 2].copy_(s["present"])
        # First tokens are fully published before the Decode phase.
        consume(self.emit.cpu().tolist())
        self.synchronize()
        prefill_end = time.perf_counter()
        hard_limit = max(r.budget for r in self.requests) - 1
        done = not self.ignore_eos and self.stopped()
        for n in range(hard_limit):
            if done:
                break
            if self.model.trace:
                self.model.trace.context("decode", n, None)
            if step_events is not None:
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            if self.graph:
                self.graph.replay()
            else:
                self.decode_step()
            if step_events is not None:
                finish.record()
                step_events.append((begin, finish))
            ring.publish(n)
            if not self.ignore_eos and (n + 1) % self.poll == 0:
                done = self.stopped()
        ring.drain()
        self.synchronize()
        end = time.perf_counter()
        durations = torch.tensor([prefill_end - start, end - prefill_end, end - start], device=self.device, dtype=torch.float64)
        totals = torch.tensor([sum(len(r.prompt) for r in self.local), sum(map(len, outputs)),
                               sum(max(len(o) - 1, 0) for o in outputs)], device=self.device, dtype=torch.int64)
        if self.world > 1:
            dist.all_reduce(durations, op=dist.ReduceOp.MAX)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        p, d, e = durations.tolist()
        inp, out, dec = totals.tolist()
        result = {r.request_id: o for r, o in zip(self.local, outputs)}
        if self.world > 1:
            parts = [None] * self.world
            dist.all_gather_object(parts, result)
            result = {key: value for part in parts for key, value in part.items()}
        result.update(self.empty)
        return result, dict(prefill_s=p, decode_s=d, e2e_s=e, input_tokens=inp, output_tokens=out,
                            decode_tokens=dec, decode_tps=dec / d if dec else None, e2e_tps=out / e,
                            capacity=self.capacity)
