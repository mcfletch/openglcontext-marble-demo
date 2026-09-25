"""The shaft, and the speed it takes to climb it.

A chimney is the fragment that makes a run-up worth having, so the measurement
is the one that matters to a story: arrive at the mouth at a speed, and record
the highest the marble gets.  The shelf at the top is the number it has to beat.

Every measurement is taken with the piece entered facing +Z.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Wall

STEP = 1 / 120.0

#: Where the marble's centre sits when it is resting on a floor.
RESTING = 0.5

VARIANTS = sorted(fragments.library()['chimney'].variants)


def _built(variant=None, seed=3):
    return fragments.build('chimney', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _highest(piece, speed, seconds=8.0):
    """The highest the marble's centre gets, entering the mouth at ``speed``."""
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    highest = -1e9
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        highest = max(highest, float(world.position[index][1]))
        if game.controller.fall_count:
            break
    return highest


def _shelf(piece):
    """The height a marble's centre is at when it has made the top."""
    return piece.exit.height + RESTING


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_a_chimney_climbs_and_every_step_of_it_is_one_a_marble_can_roll(variant):
    piece = _built(variant)
    assert piece.exit.height > piece.entry.height + 1.0, \
        '%s only climbs %.2f m' % (variant, piece.exit.height - piece.entry.height)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    for (col, row), height in piece.cells.items():
        for dcol, drow in ((1, 0), (0, 1)):
            beside = (col + dcol, row + drow)
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                    '%s steps %.2f at %r' % (variant,
                                             piece.cells[beside] - height, (col, row))


def test_the_shaft_is_walled_down_both_sides():
    """The point of a chimney is that there is nowhere to go but up it."""
    piece = _built()
    sides = {wall.side for wall in piece.features if isinstance(wall, Wall)}
    assert {'E', 'W'} <= sides, 'the shaft is open at the side: %r' % (sorted(sides),)


# -- and the rule it imposes ---------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_arriving_slowly_does_not_get_up_the_chimney(variant):
    piece = _built(variant)
    highest = _highest(piece, 4.0)
    assert highest < _shelf(piece) - 0.5, \
        '%s: at 4 m/s the marble reached %.2f m of the %.2f m the shelf is at' \
        % (variant, highest, _shelf(piece))


@pytest.mark.parametrize('variant', VARIANTS)
def test_arriving_fast_gets_up_the_chimney(variant):
    piece = _built(variant)
    highest = _highest(piece, 16.0)
    assert highest >= _shelf(piece) - 0.05, \
        '%s: at 16 m/s the marble reached %.2f m of the %.2f m the shelf is at' \
        % (variant, highest, _shelf(piece))


def test_the_climb_is_graded_by_speed_rather_than_open_or_shut():
    """Between the speed that fails and the speed that clears there is a range
    where more speed gets further up, which is what makes a chimney worth
    approaching well rather than a door with a key."""
    piece = _built('tall')
    reached = [(speed, _highest(piece, speed)) for speed in (4.0, 8.0, 12.0, 16.0)]
    report = ', '.join('%.0f m/s: %.2f m' % pair for pair in reached)
    for (_, lower), (_, higher) in zip(reached, reached[1:], strict=False):
        assert higher > lower - 0.05, 'more speed got less far up — %s' % report
    assert reached[-1][1] - reached[0][1] > 1.0, \
        'speed made almost no difference to the climb — %s' % report
