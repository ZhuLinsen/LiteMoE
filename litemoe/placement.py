"""Per-layer bijections from checkpoint expert ids to rank-contiguous slots."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass
class Placement:
    physical: list[list[int]]
    world: int
    model: str = ""
    revision: str = ""
    calibration_digest: str = ""

    def validate(self, layers, experts, world):
        if world not in (1, 2) or experts % world or self.world != world:
            raise ValueError("M0 requires W=1/2 and E divisible by W")
        if len(self.physical) != layers:
            raise ValueError("Placement layer count mismatch")
        for row in self.physical:
            if any(type(x) is not int for x in row) or sorted(row) != list(range(experts)):
                raise ValueError("Each placement layer must be an integer bijection")
        return self

    def logical_on_rank(self, layer, rank):
        row = self.physical[layer]
        local = len(row) // self.world
        # Physical slots are a permutation; invert in O(E) without sorting.
        inverse = [0] * len(row)
        for expert, slot in enumerate(row):
            inverse[slot] = expert
        return inverse[rank * local:(rank + 1) * local]

    @property
    def sha256(self):
        return digest(self.__dict__)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def uniform(layers, experts, world, **metadata):
    return Placement([list(range(experts)) for _ in range(layers)], world, **metadata).validate(layers, experts, world)


def lpt(counts, world, **metadata):
    """Capacity-constrained LPT, deterministic ties by rank and logical id."""
    if not counts or not counts[0] or any(len(row) != len(counts[0]) for row in counts):
        raise ValueError("Counts must be a nonempty rectangular matrix")
    experts = len(counts[0])
    if world not in (1, 2) or experts % world:
        raise ValueError("Invalid EP size")
    physical = []
    for row in counts:
        if any(not isinstance(x, (int, float)) or not 0 <= x < float("inf") for x in row):
            raise ValueError("Counts must be finite nonnegative numbers")
        load, slots, mapping = [0] * world, [0] * world, [0] * experts
        for expert in sorted(range(experts), key=lambda e: (-row[e], e)):
            owner = min((r for r in range(world) if slots[r] < experts // world), key=lambda r: (load[r], r))
            mapping[expert] = owner * (experts // world) + slots[owner]
            slots[owner] += 1
            load[owner] += row[expert]
        physical.append(mapping)
    return Placement(physical, world, **metadata).validate(len(counts), experts, world)
