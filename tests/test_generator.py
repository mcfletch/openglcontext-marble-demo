"""Tests for the procedural level generator.

The generator turns a ``(seed, difficulty)`` into a :class:`Level`: a connected
grid path from a start cell to a finish cell, descending within a slope budget.
The properties that matter are determinism (so a seed is a shareable level),
connectivity/solvability (there is always a way from start to finish), and
difficulty scaling.
"""
from collections import deque

from openglcontext_marble_demo import boards, generator
from openglcontext_marble_demo.level import Bumper, Level, RotatingArm, SpringTrap


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


def test_higher_difficulty_makes_a_tighter_board_rather_than_a_longer_one():
    """What makes a board hard is the decisions in it.

    A long easy board is only a long one, so difficulty *shortens* the route and
    takes away room -- the plazas thin out -- rather than adding tiles.  The
    clock follows the route, so a shorter board gets less of it.
    """
    def route(level):
        return boards.distances(set(level.cells), level.start_cell)[level.finish_cell]

    easy = [generator.generate(seed=s, difficulty=1) for s in range(12)]
    hard = [generator.generate(seed=s, difficulty=5) for s in range(12)]
    assert sum(map(route, hard)) < sum(map(route, easy))
    assert sum(l.time_limit for l in hard) < sum(l.time_limit for l in easy)


def test_harder_boards_carry_more_hazards():

    def hazards(difficulty):
        return sum(sum(isinstance(f, (Bumper, RotatingArm, SpringTrap))
                       for f in generator.generate(s, difficulty).features)
                   for s in range(16))
    assert hazards(5) > hazards(1)


def test_a_run_is_short_enough_that_another_go_is_cheap():
    """A fast game is one you can retry without thinking about it."""
    for seed in range(12):
        for difficulty in (1, 3, 5):
            level = generator.generate(seed=seed, difficulty=difficulty)
            assert 15.0 <= level.time_limit <= 45.0


def test_track_descends_within_a_slope_budget():
    """No single step drops more than the slope budget (keeps it rollable)."""
    level = generator.generate(seed=9, difficulty=3)
    for (col, row), height in level.cells.items():
        for dc, dr in ((1, 0), (0, 1)):
            nb = (col + dc, row + dr)
            if nb in level.cells:
                assert abs(level.cells[nb] - height) <= generator.MAX_STEP + 1e-9
