"""The peg board, which is a question about odds and so is tested as one.

Most of a mechanism can be held to a single run: a spring trap either launched
the marble or it did not.  A peg board cannot be, because a board that always
answers the same is a corridor and one whose answer owes nothing to the entry is
a coin.  What it promises is a *distribution* -- every slot happens, none of
them happens nearly always, and where the marble goes in moves the odds -- so
the tests here put many marbles down the board and count where they arrived.

They are honest about being statistics.  The entries come from a seeded
generator and the physics is deterministic, so a run of the suite repeats the
last one rather than drawing a fresh sample; and the assertions are on
directions and on wide margins rather than on the tallies, because a board tuned
until a particular count comes out is a board tuned to its test.

Each sweep costs a few tens of seconds, so they are ``slow`` and are gathered by
module-scoped fixtures: one pass down the board answers every question asked of
it.
"""
import random

import numpy as np
import pytest
from omi_physics import raycast
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import fragments, levelfile, materials, mechanisms
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.pegs import PegBoard
from openglcontext_marble_demo.pieces import MAX_STEP

DT = 1 / 120.0

#: How long one marble is given to get down the board, in seconds of simulation,
#: before the run counts as not having got through.  Well past the slowest run
#: seen, so a marble that fails to arrive has stopped rather than dawdled.
BUDGET = 16.0

#: Marbles per entry window.  Enough that a slot taking a fifth of the entries
#: arrives several times, which is what the reachability assertion rests on.
RUNS = 20

#: Where a marble goes in, in metres to the left of the board's centre line.
#: The board is three cells -- twelve metres -- across, so these are the middle
#: and the two outer thirds: an aim rather than a nudge.
CENTRE = (-1.5, 1.5)
LEFT = (2.0, 4.0)
RIGHT = (-4.0, -2.0)

#: Entry speeds, in m/s: a roll and a run at it.
ENTRY_SPEEDS = (2.0, 6.0)


def _board(**named):
    settings = dict(cell=(0, 0), seed=11)
    settings.update(named)
    return PegBoard(**settings)


def _level(board, runout=3):
    """``board`` as a playable level, with a flat run-out past the slots.

    The run-out is where a marble that got through is measured.  Without it the
    slots end in air, and a marble leaving one falls and is respawned before
    anything can read the speed it left with.
    """
    cells = dict(board.cells())
    bottom = min(cells.values())
    for extra in range(runout):
        for offset in board.offsets():
            cells[board.cell_at(board.rows() + extra, offset)] = bottom
    finish = board.cell_at(board.rows() + runout - 1, board.offsets()[0])
    return Level(name='pegs', cells=cells, start_cell=board.cell,
                 finish_cell=finish, time_limit=600.0,
                 features=[board, Finish(finish)])


def _built(board, level=None):
    """``board`` in a physics scene, as ``(scene, level, BuildResult)``."""
    level = level if level is not None else _level(board)
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, level, level.build_into(scene, index)


def _run(board, across, speed):
    """One marble down the board, entering ``across`` metres left of centre.

    Answers ``(slot, speed)`` as it crosses the far end of the slots, or
    ``(None, 0.0)`` if it never got there.
    """
    level = _level(board)
    game = MarbleGame(level, base_tilt=fragments.DESIGN_TILT)
    world, marble = game.scene.world, game.marble.index
    # place_body rather than a write to ``position``: the world AABB is what the
    # broad phase asks about, and a bare write leaves it where the marble was.
    world.place_body(marble, board.entry_position(across=across))
    world.linear_velocity[marble] = (board.facing[0] * speed, 0.0,
                                     board.facing[1] * speed)
    world.angular_velocity[marble] = (0.0, 0.0, 0.0)
    world.wake(marble)
    end = (board.rows() - 0.5) * level.cell_size
    for _ in range(int(BUDGET / DT)):
        game.advance(DT)
        at = world.position[marble]
        along, _left = board.board_coordinates(at, level.cell_size)
        if along >= end:
            return (board.slot_of(at, level.cell_size),
                    float(np.linalg.norm(world.linear_velocity[marble][[0, 2]])))
    return None, 0.0


def _sweep(window, runs=RUNS, seed=7):
    """``runs`` marbles entered somewhere in ``window``.

    Answers ``(counts, speeds, lost)``: how many arrived in each slot, the speeds
    they left each slot with, and how many never arrived.
    """
    board = _board()
    rng = random.Random(seed)
    counts: dict = {}
    speeds: dict = {}
    lost = 0
    for _ in range(runs):
        slot, leaving = _run(board, rng.uniform(*window), rng.uniform(*ENTRY_SPEEDS))
        if slot is None:
            lost += 1
            continue
        counts[slot] = counts.get(slot, 0) + 1
        speeds.setdefault(slot, []).append(leaving)
    return counts, speeds, lost


def _share(counts, slot):
    total = sum(counts.values())
    return counts.get(slot, 0) / total if total else 0.0


@pytest.fixture(scope='module')
def centre_entries():
    return _sweep(CENTRE)


@pytest.fixture(scope='module')
def left_entries():
    return _sweep(LEFT)


@pytest.fixture(scope='module')
def right_entries():
    return _sweep(RIGHT)


# -- the board as data ---------------------------------------------------

def test_a_board_lays_a_cell_per_slot_per_row():
    board = _board(width=3, length=4, slot_length=2)
    assert len(board.cells()) == 3 * 6
    assert board.owned_cells() == set(board.cells())


def test_a_board_descends_from_its_entry_to_its_slots():
    board = _board(drop=3.5)
    heights = board.cells()
    top = heights[board.cell_at(0, 0)]
    bottom = heights[board.cell_at(board.rows() - 1, 0)]
    assert board.step() < 0
    assert top - bottom == pytest.approx(3.5 * (board.rows() - 1) / board.rows())


def test_no_row_steps_further_than_a_marble_can_roll():
    assert abs(_board().step()) <= MAX_STEP


def test_a_slot_is_a_column_of_the_board_counted_from_the_left():
    board = _board(width=3)
    assert board.offsets() == [1, 0, -1]
    # Facing +Z, a marble's left is +X, so the leftmost slot is the +X column.
    assert board.slot_of((4.0, 0.0, 0.0)) == 0
    assert board.slot_of((0.0, 0.0, 0.0)) == 1
    assert board.slot_of((-4.0, 0.0, 0.0)) == 2
    assert board.slot_of((40.0, 0.0, 0.0)) is None


def test_each_slot_owns_the_chute_cells_under_its_column():
    board = _board(width=3, length=4, slot_length=2)
    every = set()
    for slot in range(board.width):
        cells = board.slot_cells(slot)
        assert len(cells) == board.slot_length
        assert set(cells) <= board.owned_cells()
        every |= set(cells)
    assert len(every) == board.width * board.slot_length


def test_a_board_knows_when_a_marble_is_past_it():
    board = _board()
    assert board.past_the_end((0.0, 0.0, board.rows() * 4.0))
    assert not board.past_the_end((0.0, 0.0, 0.0))


def test_the_pegs_are_a_field_rather_than_a_handful():
    places = _board().peg_places()
    assert len(places) >= 12
    rows = {round(along, 3) for along, _across in places}
    assert len(rows) >= 4, 'one row of posts is a comb'


def test_neighbouring_pegs_leave_a_marble_room_to_pass():
    """A gap narrower than a marble is a pocket rather than a way through."""
    board = _board()
    reach = board.peg_radius * 2 ** 0.5          # a post stands corner-on
    by_row: dict = {}
    for along, across in board.peg_places():
        by_row.setdefault(round(along, 3), []).append(across)
    for row in by_row.values():
        row.sort()
        for near, far in zip(row, row[1:], strict=False):
            assert far - near - 2 * reach > 1.0, (near, far)


def test_a_peg_is_less_bouncy_than_a_bumper():
    """So the marble works down the board rather than being fired back up it."""
    assert _board().peg_restitution < materials.SURFACES['rubber_pad'].restitution


def test_the_same_board_is_the_same_board():
    """Two boards with one seed are one board, so a run can be repeated."""
    first, again = _board(seed=4), _board(seed=4)
    assert first == again
    assert first.peg_places() == again.peg_places()


# -- building ------------------------------------------------------------

def test_a_board_built_without_its_cells_says_what_is_missing():
    board = _board()
    bare = Level(name='bare', cells={board.cell: 0.0}, start_cell=board.cell,
                 finish_cell=board.cell, time_limit=60.0, features=[board])
    with pytest.raises(ValueError, match='cells laid'):
        _built(board, bare)


def test_a_board_builds_a_tile_a_peg_and_a_wall_for_everything_it_declares():
    board = _board()
    _scene, _level_, result = _built(board)
    wanted = len(board.owned_cells()) + len(board.peg_places()) + board.slot_length
    assert len(result.feature_bodies) > wanted


def _surface_height(world, x, z, from_above=8.0):
    """Where a ray straight down at ``(x, z)`` meets the board, or ``None``."""
    hit = raycast.raycast(world, (x, from_above, z), (0.0, -1.0, 0.0),
                          max_distance=40.0)
    return None if hit is None else float(hit.point[1])


def test_the_board_is_one_slope_and_not_a_staircase():
    """Every tile's far edge meets the next tile's near edge.

    A tile turned the other way puts a riser in front of the marble every cell,
    and a marble rolls down a riser perfectly well and cannot roll up one at
    all.  The board has to be a surface, so the height under it never climbs.
    """
    board = _board()
    scene, _level_, _result = _built(board)
    # Down the middle of the leftmost slot, clear of the peg lattice and of both
    # the outer rail and the divider.
    line = 4.62
    heights = [(step * 0.5, _surface_height(scene.world, line, step * 0.5))
               for step in range(1, board.rows() * 8)]
    found = [(along, height) for along, height in heights if height is not None]
    assert len(found) > board.rows() * 4, 'the ray found no board to measure'
    for (_before, high), (after, low) in zip(found, found[1:], strict=False):
        assert low <= high + 0.05, 'the surface climbs at %.1f m along' % after


def test_only_the_fast_slot_gives_speed_back():
    board = _board(fast_slot=1, slot_length=2)
    _scene, _level_, result = _built(board)
    assert len(result.effects) == board.slot_length     # one per chute cell


def test_the_chute_brings_a_slow_marble_up_to_speed_and_never_slows_a_fast_one():
    board = _board()
    scene, _level_, result = _built(board)
    world = scene.world
    effect = next(iter(result.effects.values()))
    marble = scene.add_sphere(radius=0.5, position=(0.0, 20.0, 0.0), mass=3.0)
    for entering, expected in ((2.0, board.boost_speed), (25.0, 25.0)):
        world.linear_velocity[marble.index] = (0.0, 0.0, entering)
        effect(world, marble.index)
        assert world.linear_velocity[marble.index][2] == pytest.approx(expected,
                                                                      abs=1e-6)


def test_the_slots_are_walled_apart_from_each_other():
    """Two slots a marble can roll between are one slot."""
    board = _board()
    scene, level, _result = _built(board)
    world = scene.world
    standing = {(round(float(x), 3), round(float(z), 3))
                for x, _y, z in world.position[:world.body_count]}
    for row in range(board.length, board.rows()):
        for offset in board.offsets()[:-1]:
            cell = board.cell_at(row, offset)
            x, z = level.cell_center(cell)
            assert (round(x - level.cell_size / 2.0, 3), round(z, 3)) in standing, cell


def test_a_board_round_trips_through_the_file_format():
    board = _board(width=5, length=3, slot_length=2, fast_slot=3, drop=4.2,
                   boost_speed=13.5, peg_radius=0.28, facing=(1, 0))
    level = _level(board)
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.cells == level.cells
    read = [feature for feature in again.features if isinstance(feature, PegBoard)]
    assert read == [board]
    assert read[0].cells() == board.cells()


def test_the_file_format_knows_the_board_by_name():
    assert mechanisms.registry()['pegs'] is PegBoard
    assert levelfile.FEATURES['pegs'] is PegBoard


# -- what a marble actually does -----------------------------------------

@pytest.mark.slow
def test_every_marble_gets_down_the_board(centre_entries, left_entries,
                                          right_entries):
    """A board a marble can be stopped on is a trap rather than a slope."""
    for counts, _speeds, lost in (centre_entries, left_entries, right_entries):
        assert lost <= RUNS // 10, counts


@pytest.mark.slow
def test_every_slot_is_reached(centre_entries, left_entries, right_entries):
    pooled: dict = {}
    for counts, _speeds, _lost in (centre_entries, left_entries, right_entries):
        for slot, arrived in counts.items():
            pooled[slot] = pooled.get(slot, 0) + arrived
    assert sorted(pooled) == [0, 1, 2], pooled
    assert min(pooled.values()) >= 3, pooled


@pytest.mark.slow
def test_no_slot_takes_almost_everything(centre_entries):
    """Entered down the middle, the board answers three ways and not one."""
    counts, _speeds, _lost = centre_entries
    assert len(counts) == 3, counts
    assert max(counts.values()) / sum(counts.values()) < 0.7, counts


@pytest.mark.slow
def test_the_fast_slot_is_faster_than_an_ordinary_one(centre_entries):
    """Measured as the speed marbles left with, rather than read off a setting."""
    _counts, speeds, _lost = centre_entries
    fast_slot = _board().fast_slot
    fast = speeds[fast_slot]
    ordinary = [speed for slot, leaving in speeds.items() if slot != fast_slot
                for speed in leaving]
    assert fast and ordinary
    # Every fast-slot exit beats every ordinary one by half again, so the margin
    # is a property of the slots rather than of the average marble.
    assert min(fast) > 1.5 * max(ordinary), (min(fast), max(ordinary))


@pytest.mark.slow
def test_entering_left_lands_left_more_often_than_entering_right_does(
        left_entries, right_entries):
    """The direction of the shift, which is what makes the board worth aiming.

    Not a number: how far the odds move is a tuning decision, and a test that
    fixed it would fail every time the board was tuned rather than when it
    stopped rewarding aim.
    """
    left_counts, _speeds, _lost = left_entries
    right_counts, _speeds, _lost = right_entries
    leftmost, rightmost = 0, _board().width - 1
    assert _share(left_counts, leftmost) > _share(right_counts, leftmost)
    assert _share(right_counts, rightmost) > _share(left_counts, rightmost)


@pytest.mark.slow
def test_where_a_marble_ends_up_is_not_settled_at_the_entry(left_entries):
    """Aim shifts the odds; it does not decide the answer."""
    counts, _speeds, _lost = left_entries
    assert len(counts) > 1, counts
    assert max(counts.values()) / sum(counts.values()) < 0.95, counts
