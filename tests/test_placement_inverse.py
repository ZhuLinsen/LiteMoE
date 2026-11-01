"""Regression tests for the logical/physical expert permutation."""

import random

import pytest

from litemoe.placement import Placement, lpt


@pytest.mark.parametrize('experts,world', [(1, 1), (32, 1), (32, 2), (40, 2)])
def test_inverse_matches_physical_slot_order(experts, world):
    rng = random.Random(8102)
    rows = []
    for _ in range(12):
        row = list(range(experts))
        rng.shuffle(row)
        rows.append(row)
    placement = Placement(rows, world).validate(len(rows), experts, world)
    for layer, row in enumerate(rows):
        inverse = sorted(range(experts), key=row.__getitem__)
        for rank in range(world):
            size = experts // world
            expected = inverse[rank * size:(rank + 1) * size]
            assert placement.logical_on_rank(layer, rank) == expected
            assert [row[e] for e in expected] == list(range(rank * size, (rank + 1) * size))


def test_lpt_inversion_preserves_capacity_and_all_experts():
    counts = [[20, 0, 4, 10, 3, 5, 1, 7]]
    placement = lpt(counts, 2)
    members = [placement.logical_on_rank(0, r) for r in range(2)]
    assert all(len(x) == 4 for x in members)
    assert sorted(sum(members, [])) == list(range(8))
