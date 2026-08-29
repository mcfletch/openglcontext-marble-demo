"""The lip: a step down the board that a marble cannot come back up.

The rule is the whole of the piece, and it is a rule about what *cannot* happen,
so the measurements go the wrong way down the lane: put the marble on the lower
floor, fire it at the lip, and record the highest it ever gets.  The upper floor
is the number it would have to reach.

Every measurement is taken with the piece entered facing +Z, so a row of the
grid is a step down the board.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish

STEP = 1 / 120.0

#: Where the marble's centre sits when it is resting on a floor.
RESTING = 0.5

VARIANTS = sorted(fragments.library()['one_way'].variants)


def _built(variant=None, seed=3):
    return fragments.build('one_way', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _floors(piece):
    """``(upper_rows, lower_rows, drop)`` of the centre column."""
    col = piece.entry.cell[0]
    upper = sorted(row for (at, row), height in piece.cells.items()
                   if at == col and abs(height - piece.entry.height) < 1e-6)
    lower = sorted(row for (at, row), height in piece.cells.items()
                   if at == col and height < piece.entry.height - 1e-6)
    return upper, lower, piece.entry.height - piece.cells[(col, lower[0])]


def _highest(piece, speed, seconds=4.0):
    """The highest the marble's centre gets, fired at the lip from below."""
    _, lower, drop = _floors(piece)
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    world.place_body(index, (0.0, piece.entry.height - drop + RESTING,
                             max(lower) * CELL_SIZE))
    world.linear_velocity[index] = (0.0, 0.0, -speed)
    world.wake(index)
    highest = -1e9
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        highest = max(highest, float(world.position[index][1]))
        if game.controller.fall_count:
            break
    return highest


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_a_lip_is_one_step_and_no_more(variant):
    """A step is a step whatever its height, and the smallest one the board's
    slope budget forbids climbing is the one to build: anything deeper is a
    cliff for the look of it."""
    piece = _built(variant)
    upper, lower, drop = _floors(piece)
    assert upper and lower, variant
    assert max(upper) < min(lower), '%s: the lip does not face down the board' % variant
    assert abs(drop - pieces.MAX_STEP) < 1e-6, \
        '%s drops %.2f m, and the budget is %.2f' % (variant, drop, pieces.MAX_STEP)


@pytest.mark.parametrize('variant', VARIANTS)
def test_both_floors_are_reachable_the_way_the_piece_is_entered(variant):
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    assert piece.exit.height < piece.entry.height - 1e-6, variant


# -- and the rule it imposes ---------------------------------------------------

@pytest.mark.parametrize('speed', [6.0, 12.0, 20.0, 28.0])
def test_no_speed_carries_a_marble_back_up_the_lip(speed):
    """Twenty-eight metres a second is faster than anything on a board reaches;
    the lip is not a matter of trying harder."""
    piece = _built()
    upper, _, _ = _floors(piece)
    needed = piece.entry.height + RESTING
    highest = _highest(piece, speed)
    assert highest < needed - 0.4, \
        'at %.0f m/s the marble reached %.2f m, and the upper floor holds it ' \
        'at %.2f m' % (speed, highest, needed)


@pytest.mark.parametrize('variant', VARIANTS)
def test_every_variant_holds_against_the_fastest_thing_on_a_board(variant):
    piece = _built(variant)
    needed = piece.entry.height + RESTING
    highest = _highest(piece, 20.0)
    assert highest < needed - 0.4, \
        '%s: at 20 m/s the marble reached %.2f m of the %.2f m it would need' \
        % (variant, highest, needed)
