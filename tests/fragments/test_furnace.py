"""The fire, the islands in it, and why hopping beats crawling.

The burner counts *time*, so what the piece is worth is decided by how much of
the crossing is spent over fire.  That is what these measure: run a marble
across the chamber on the straight line and again by way of the islands, and
compare the heat each took.

Every measurement is taken with the piece entered facing +Z, so a row of the
grid is a step down the board.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.controller import BURNED
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish
from openglcontext_marble_demo.mechanisms.burner import Burner, BurnerHeat

STEP = 1 / 120.0

VARIANTS = sorted(fragments.library()['furnace'].variants)


def _built(variant=None, seed=3):
    return fragments.build('furnace', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _fire(piece):
    return next(f for f in piece.features if isinstance(f, Burner))


def _burning_column(piece):
    """A column of the chamber that is fire in every row of it, or ``None``.

    In grid columns, which are metres across the lane divided by the cell size,
    so it doubles as the offset a crossing is started from.
    """
    fire = set(_fire(piece).cells)
    rows = {row for _, row in fire}
    for col in sorted({col for col, _ in fire}, key=abs):
        if all((col, row) in fire for row in rows):
            return col
    return None


def _cross(piece, speed, sideways=0.0, offset=0.0, seconds=12.0):
    """``(cause, dwell)`` — what the marble was lost to, and its time in the fire.

    ``offset`` is where across the mouth it starts and ``sideways`` a constant
    push across the lane, which is how a line other than the straight one is
    driven without involving the autopilot: what is being measured is the fire,
    not the steering.
    """
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    heat = next(a for a in game.build.animators if isinstance(a, BurnerHeat))
    # ``controller.last_loss`` rather than the animator's own answer: the game
    # takes the marble away when it burns and puts a new one down, and the fresh
    # marble is a body the animator has never heated.
    world.place_body(index, (offset, 0.5, 0.0))
    world.linear_velocity[index] = (sideways, 0.0, speed)
    world.wake(index)
    past = (max(row for _, row in piece.cells) + 0.5) * CELL_SIZE
    dwell, cause = 0.0, None
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        where = world.position[index]
        if heat.holds(where):
            dwell += STEP
        cause = cause or game.controller.last_loss
        if float(where[2]) > past:
            break
    return cause, dwell


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_a_furnace_is_fire_with_cool_tiles_left_in_it(variant):
    piece = _built(variant)
    fire = set(_fire(piece).cells)
    assert fire, variant
    cool = set(piece.cells) - fire
    assert cool, '%s: the chamber is fire edge to edge' % variant


@pytest.mark.parametrize('variant', VARIANTS)
def test_no_column_of_the_chamber_is_a_clear_run_of_islands(variant):
    """A column of cool tiles all the way down would be a corridor with a fire
    painted either side of it, and the piece would ask nothing."""
    piece = _built(variant)
    fire = set(_fire(piece).cells)
    rows = sorted({row for _, row in fire})
    for col in {col for col, _ in fire}:
        cool_rows = [row for row in rows if (col, row) in piece.cells
                     and (col, row) not in fire]
        assert len(cool_rows) < len(rows), \
            '%s: column %d is cool the whole way down' % (variant, col)


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_chamber_is_flat_and_walled_and_leads_through(variant):
    """Flat because the piece is about time and not about height, and walled
    because being on fire *and* off the board is a different piece."""
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    assert abs(piece.exit.height - piece.entry.height) < 1e-6, variant
    heights = set(piece.cells.values())
    assert len(heights) == 1, '%s: the chamber is not flat: %r' % (variant, heights)


# -- and the rule it imposes ---------------------------------------------------

def test_crossing_the_fire_at_speed_gets_through_it():
    piece = _built()
    cause, dwell = _cross(piece, 14.0)
    assert cause is None, \
        'burned crossing at 14 m/s, having stood %.2f s in a %.2f s fire' \
        % (dwell, _fire(piece).burn_time)


@pytest.mark.parametrize('variant', VARIANTS)
def test_a_furnace_has_a_line_across_it_that_is_fire_the_whole_way(variant):
    """Which is what makes the islands worth finding.  A chamber where every
    line touches an island somewhere is one a player can cross without looking:
    the cooling would break the heat into hops on its own."""
    piece = _built(variant)
    assert _burning_column(piece) is not None, \
        '%s: no column of the chamber is fire the whole way down' % variant


@pytest.mark.parametrize('variant', VARIANTS)
def test_crawling_down_the_burning_line_loses_the_marble(variant):
    """The other half of it: a fire nothing dies in is decoration."""
    piece = _built(variant)
    column = _burning_column(piece)
    cause, dwell = _cross(piece, 1.0, offset=column * CELL_SIZE, seconds=16.0)
    assert cause == BURNED, \
        '%s: crawling down the burning line at 1 m/s the marble stood %.2f s ' \
        'in a %.2f s fire and was lost to %r' \
        % (variant, dwell, _fire(piece).burn_time, cause)


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_same_line_taken_at_speed_gets_through(variant):
    """So the line is not a wall: what costs the marble is the crawl, not the
    choice of line, and a player who keeps moving is never punished for it."""
    piece = _built(variant)
    column = _burning_column(piece)
    cause, dwell = _cross(piece, 14.0, offset=column * CELL_SIZE)
    assert cause is None, \
        '%s: down the burning line at 14 m/s the marble stood %.2f s in a ' \
        '%.2f s fire and was lost to %r' \
        % (variant, dwell, _fire(piece).burn_time, cause)


def test_the_fire_is_what_costs_the_time_rather_than_the_distance():
    """More speed is less heat, monotonically: the burner is a clock and the
    only thing that shortens it is being over the fire for less of it."""
    piece = _built('wide')
    dwells = [(speed, _cross(piece, speed, seconds=14.0)[1])
              for speed in (4.0, 8.0, 12.0, 16.0)]
    report = ', '.join('%.0f m/s: %.2f s' % pair for pair in dwells)
    for (_, slower), (_, faster) in zip(dwells, dwells[1:], strict=False):
        assert faster <= slower + 0.05, 'more speed took more heat — %s' % report
    assert dwells[0][1] - dwells[-1][1] > 0.2, \
        'speed made almost no difference to the heat — %s' % report
