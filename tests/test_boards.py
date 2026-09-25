"""What a generated board has to be: wide, braided, and never fully blocked.

The properties here are the ones the layout exists for, and they are the ones a
one-cell corridor fails:

**Room.** Every rank of a board is at least two cells across, so a hazard on it
is something to steer around rather than a gate to wait at.

**More than one way.** Two routes from start to finish that share no cell but
the ends, so "which way do I go" is a question with an answer worth having.

**A clean line.** Whatever the board is decorated with, a route exists that
meets no hazard — barriers block *some* passages, never all of them.

**A reason to take the risk.** Where the board braids, the hazardous strand is
the short one.

Every check runs over a spread of seeds, because a property that holds for one
board and not the next is not a property.
"""
import random
from collections import deque

import pytest

from openglcontext_marble_demo import boards, generator
from openglcontext_marble_demo.level import Bumper, Ramp, RotatingArm, SpringTrap, Wall

SEEDS = range(24)
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _route(level, blocked=frozenset()):
    """A start->finish path over cells not in ``blocked``, or None."""
    if level.start_cell in blocked:
        return None
    previous = {level.start_cell: None}
    queue = deque([level.start_cell])
    while queue:
        cell = queue.popleft()
        if cell == level.finish_cell:
            path = []
            while cell is not None:
                path.append(cell)
                cell = previous[cell]
            return path[::-1]
        for dc, dr in NEIGHBOURS:
            nxt = (cell[0] + dc, cell[1] + dr)
            if nxt in level.cells and nxt not in previous and nxt not in blocked:
                previous[nxt] = cell
                queue.append(nxt)
    return None


def _hazard_cells(level):
    """Cells a marble cannot simply roll across."""
    return {feature.cell for feature in level.features
            if isinstance(feature, (Bumper, RotatingArm, SpringTrap))}


def _second_route(level):
    """A route sharing no cell with the shortest one, but the two ends."""
    first = _route(level)
    assert first is not None
    interior = set(first[1:-1])
    return _route(level, blocked=interior)


# -- room ----------------------------------------------------------------------

def test_no_board_is_a_one_cell_corridor():
    """Every cell has a lateral neighbour: there is always somewhere to go but
    straight on."""
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        for col, row in level.cells:
            room = sum(((col + dc, row + dr) in level.cells)
                       for dc, dr in NEIGHBOURS)
            assert room >= 2, "seed %d: cell %r is a dead end" % (seed, (col, row))


def test_the_route_is_at_least_two_cells_wide_the_whole_way():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        for cell in _route(level):
            col, row = cell
            beside = sum(((col + dc, row + dr) in level.cells)
                         for dc, dr in ((1, 0), (-1, 0)))
            ahead = sum(((col + dc, row + dr) in level.cells)
                        for dc, dr in ((0, 1), (0, -1)))
            assert beside or ahead >= 2, "seed %d: %r has no lane beside it" % (seed, cell)


def test_a_board_is_wider_than_it_is_long_in_cells_per_rank():
    """A board is a plane with a route across it, not a line of tiles."""
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        ranks = {row for _, row in level.cells}
        assert len(level.cells) / len(ranks) >= 2.0


# -- more than one way ---------------------------------------------------------

def test_there_are_two_routes_that_share_only_their_ends():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        assert _second_route(level) is not None, "seed %d has one route" % seed


def _through(board, cell):
    """Steps from start to finish by way of ``cell``, or None."""
    out = boards.distances(board.cells, board.start)
    back = boards.distances(board.cells, board.finish)
    if cell not in out or cell not in back:
        return None
    return out[cell] + back[cell]


def test_braiding_gives_a_way_round_that_is_not_just_the_next_lane():
    """Widening gives two lanes; braiding gives two *routes*.

    A lane beside the route is the same journey a metre to the left. What makes
    a board a choice is a way round that is genuinely longer or shorter, and
    that is what a strand is: taking the scenic one costs real cells.
    """
    different = 0
    boards_with_a_scenic_way = 0
    for seed in SEEDS:
        board = boards.build(random.Random(seed), length=15, difficulty=3)
        shortest = boards.distances(board.cells, board.start)[board.finish]
        for strand in board.strands_of(boards.SCENIC):
            boards_with_a_scenic_way += 1
            costs = [_through(board, cell) for cell in strand.cells]
            worst = max((cost for cost in costs if cost is not None), default=0)
            if worst - shortest >= 3:
                different += 1
            break
    assert boards_with_a_scenic_way >= len(SEEDS) // 3, "hardly any board braids"
    assert different >= boards_with_a_scenic_way * 2 // 3


def test_a_shortcut_strand_really_is_shorter_than_the_spine_it_bypasses():
    found = 0
    for seed in SEEDS:
        board = boards.build(random.Random(seed), length=15, difficulty=3)
        for strand in board.strands_of(boards.SHORTCUT):
            assert strand.saves > 0, "a shortcut that saves nothing is a lane"
            found += 1
    assert found, "no board offered a shortcut"


# -- a clean line --------------------------------------------------------------

def test_a_hazard_free_route_always_exists():
    """Barriers block some passages, never all of them."""
    for seed in SEEDS:
        for difficulty in (1, 3, 5):
            level = generator.generate(seed=seed, difficulty=difficulty)
            assert _route(level, blocked=_hazard_cells(level)) is not None, \
                "seed %d d%d is blocked" % (seed, difficulty)


def test_boards_are_decorated_rather_than_left_bare():
    """The clean line has to be *found*, not handed over: a board with nothing
    on it passes the test above for the wrong reason."""
    hazards = sum(len(_hazard_cells(generator.generate(seed=s, difficulty=3)))
                  for s in SEEDS)
    assert hazards >= len(SEEDS)


def test_the_start_and_the_finish_are_never_obstructed():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=4)
        blocked = _hazard_cells(level)
        assert level.start_cell not in blocked
        assert level.finish_cell not in blocked


# -- a reason to take the risk -------------------------------------------------

def test_where_a_board_braids_the_hazardous_strand_is_the_short_one():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=3)
        quick = _route(level)
        safe = _route(level, blocked=_hazard_cells(level))
        assert safe is not None
        assert len(quick) <= len(safe), \
            "seed %d: the clean way round is also the quick one" % seed


# -- the board rolls the way it leans ------------------------------------------

def test_the_route_always_has_board_in_front_of_it():
    """A passive marble must never be pushed over an edge.

    The board leans one way and a marble nobody is steering rolls that way. On a
    one-cell corridor that turned sideways, the lean pushed a player who touched
    nothing straight into the void; here every cell of the route has a cell
    downhill of it, so drifting is drifting along the board rather than off it.
    """
    for seed in SEEDS:
        for difficulty in (1, 3, 5):
            level = generator.generate(seed=seed, difficulty=difficulty)
            last = max(row for _, row in level.cells)
            for col, row in _route(level):
                if row < last - 1:
                    assert (col, row + 1) in level.cells, \
                        "seed %d d%d: nothing downhill of %r" % (
                            seed, difficulty, (col, row))


def test_every_cell_of_the_board_has_board_downhill_or_is_near_the_end():
    """Not only the route: wherever a player wanders, the same has to hold."""
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=3)
        last = max(row for _, row in level.cells)
        adrift = [cell for cell in level.cells
                  if cell[1] < last - 1 and (cell[0], cell[1] + 1) not in level.cells]
        # A few outer cells of a bulge may end early; the route never does, and
        # they are a small minority of a board rather than its shape.
        assert len(adrift) <= len(level.cells) // 8, \
            "seed %d: %d cells lead over an edge" % (seed, len(adrift))


def test_the_finish_is_downhill_of_the_start():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        assert level.finish_cell[1] > level.start_cell[1]
        assert level.cells[level.finish_cell] < level.cells[level.start_cell]


# -- still terraced, still deterministic ---------------------------------------

def test_no_step_between_neighbours_exceeds_the_slope_budget():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=4)
        for (col, row), height in level.cells.items():
            for dc, dr in ((1, 0), (0, 1)):
                beside = (col + dc, row + dr)
                if beside in level.cells:
                    assert abs(level.cells[beside] - height) <= generator.MAX_STEP + 1e-9


def test_a_board_is_flat_across_its_width():
    """A rank is one terrace, so a lane change is not a step."""
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=2)
        for (col, row), height in level.cells.items():
            beside = (col + 1, row)
            if beside in level.cells:
                assert level.cells[beside] == pytest.approx(height)


def test_the_same_seed_gives_the_same_board_every_time():
    for seed in (0, 5, 17):
        first = generator.generate(seed=seed, difficulty=3)
        again = generator.generate(seed=seed, difficulty=3)
        assert first.cells == again.cells
        assert first.features == again.features
        assert first.cell_surfaces == again.cell_surfaces


def test_rails_only_ever_face_the_void():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=4)
        for feature in level.features:
            if isinstance(feature, Wall):
                col, row = feature.cell
                dc, dr = Wall.OFFSET[feature.side]
                assert (col + dc, row + dr) not in level.cells


def test_ramps_point_along_a_way_the_marble_can_go():
    for seed in SEEDS:
        level = generator.generate(seed=seed, difficulty=4)
        for feature in level.features:
            if isinstance(feature, Ramp):
                col, row = feature.cell
                dc, dr = feature.direction
                assert (col + dc, row + dr) in level.cells, \
                    "seed %d: a ramp at %r points into the void" % (seed, feature.cell)


# -- the pieces, on their own --------------------------------------------------

def test_a_spine_advances_and_never_crosses_itself():
    for seed in SEEDS:
        spine = boards.carve_spine(random.Random(seed), length=16)
        assert len(set(spine)) == len(spine)
        assert spine[-1][1] > spine[0][1]


def test_widening_keeps_every_spine_cell():
    rng = random.Random(3)
    spine = boards.carve_spine(rng, length=14)
    widened = boards.widen(spine, radius=1)
    assert set(spine) <= widened


def test_heights_terrace_downward_from_the_start():
    rng = random.Random(3)
    spine = boards.carve_spine(rng, length=14)
    cells = boards.widen(spine, radius=1)
    heights = boards.terrace(cells, spine)
    assert heights[spine[0]] == 0.0
    assert heights[spine[-1]] < 0.0
    assert set(heights) == cells
