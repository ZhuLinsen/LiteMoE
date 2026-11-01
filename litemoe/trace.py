"""Explicit instrumentation mode; never enabled for official throughput runs."""

import json
from pathlib import Path

import torch


class RouteTrace:
    def __init__(self, layers, experts, rank=0, raw=False):
        self.counts = {p: torch.zeros(layers, experts, dtype=torch.int64) for p in ("prefill", "decode")}
        self.rank, self.raw = rank, raw
        self.rows = []
        self.context("prefill", 0, 0)

    def context(self, phase, step, slot):
        self.phase, self.step, self.slot = phase, step, slot

    def record(self, layer, ids, valid):
        ids, valid = ids.cpu(), valid.cpu()
        self.counts[self.phase][layer].add_(torch.bincount(ids[valid].flatten(), minlength=self.counts[self.phase].shape[1]))
        if self.raw:
            for token, (row, keep) in enumerate(zip(ids.tolist(), valid.tolist())):
                self.rows.append(dict(layer=layer, phase=self.phase, step=self.step, source_rank=self.rank,
                                      request_slot=self.slot if self.slot is not None else token,
                                      token_in_round=token, valid=keep, topk_ids=row))

    def save(self, path, **metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = dict(metadata, source_rank=self.rank, counts={p: c.tolist() for p, c in self.counts.items()}, trace=self.rows)
        Path(path).write_text(json.dumps(data), encoding="utf-8")
