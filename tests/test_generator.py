"""Tests for the procedural level generator.

The generator turns a ``(seed, difficulty)`` into a :class:`Level`: a connected
grid path from a start cell to a finish cell, descending within a slope budget.
The properties that matter are determinism (so a seed is a shareable level),
connectivity/solvability (there is always a way from start to finish), and
difficulty scaling.
"""
from collections import deque

from openglcontext_marble_demo import generator
from openglcontext_marble_demo.level import Level


def _reachable(level):
    """Cells reachable from the start by 4-adjacency over occupied cells."""
    seen = {level.start_cell}
    queue = deque([level.start_cell])
    while queue:
        col, row = queue.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (col + dc, row + dr)
            if nxt in level.cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def test_generate_returns_a_level():
    level = generator.generate(seed=1, difficulty=1)
    assert isinstance(level, Level)
    assert level.start_cell in level.cells
    assert level.finish_cell in level.cells


def test_generation_is_deterministic_for_a_seed():
    a = generator.generate(seed=7, difficulty=2)
    b = generator.generate(seed=7, difficulty=2)
    assert a.cells == b.cells
    assert a.finish_cell == b.finish_cell
    assert a.time_limit == b.time_limit


def test_different_seeds_give_different_layouts():
    a = generator.generate(seed=1, difficulty=2)
    b = generator.generate(seed=2, difficulty=2)
    assert a.cells != b.cells


def test_finish_is_reachable_from_start():
    for seed in range(20):
        level = generator.generate(seed=seed, difficulty=2)
        assert level.finish_cell in _reachable(level), f"seed {seed} unsolvable"


def test_start_and_finish_are_distinct():
    level = generator.generate(seed=3, difficulty=2)
    assert level.start_cell != level.finish_cell


def test_higher_difficulty_makes_a_longer_track_and_more_time():
    easy = generator.generate(seed=5, difficulty=1)
    hard = generator.generate(seed=5, difficulty=4)
    assert len(hard.cells) > len(easy.cells)
    assert hard.time_limit > easy.time_limit


def test_track_descends_within_a_slope_budget():
    """No single step drops more than the slope budget (keeps it rollable)."""
    level = generator.generate(seed=9, difficulty=3)
    for (col, row), height in level.cells.items():
        for dc, dr in ((1, 0), (0, 1)):
            nb = (col + dc, row + dr)
            if nb in level.cells:
                assert abs(level.cells[nb] - height) <= generator.MAX_STEP + 1e-9
