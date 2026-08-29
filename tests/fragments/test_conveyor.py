"""The belt: what a marble leaves it at, whatever it arrived at.

The belt exists so that a story has somewhere for speed to come from, so the
measurement is the one a story cares about — the speed at the far end — taken
across the whole range of speeds a marble might arrive with.

Every measurement is taken with the piece entered facing +Z, so a row of the
grid is a step down the board.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.fragments.conveyor import VARIANTS as SETTINGS
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish, Ramp

STEP = 1 / 120.0

VARIANTS = sorted(fragments.library()['conveyor'].variants)


def _built(variant=None, seed=3):
    return fragments.build('conveyor', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _left_at(piece, speed, seconds=8.0):
    """How fast the marble is going as it passes the end of the belt."""
    belt = [feature.cell for feature in piece.features if isinstance(feature, Ramp)]
    plane = (max(row for _, row in belt) + 0.5) * CELL_SIZE
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        if float(world.position[index][2]) >= plane:
            return float(np.linalg.norm(world.linear_velocity[index]))
        if game.controller.fall_count:
            break
    return None


def _say(speed):
    return '%.1f m/s' % speed if speed is not None else 'never reached the end'


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_the_belt_is_flat_and_points_the_way_the_piece_is_entered(variant):
    """A sloped belt tile would be a hill as well as a belt, and the piece would
    be doing two things at once with one number to tune them."""
    piece = _built(variant)
    belts = [f for f in piece.features if isinstance(f, Ramp)]
    assert belts, variant
    for ramp in belts:
        assert ramp.rise == 0.0, '%s: a belt tile rises %.2f' % (variant, ramp.rise)
        assert ramp.direction == piece.entry.facing, \
            '%s: a belt tile pushes %r and the lane runs %r' \
            % (variant, ramp.direction, piece.entry.facing)
        assert not ramp.launch, '%s: a belt tile throws the marble' % variant


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_belt_lies_along_the_lane_and_the_lane_runs_through(variant):
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    assert abs(piece.exit.height - piece.entry.height) < 1e-6, \
        '%s does not leave at the height it was entered' % variant
    for ramp in (f for f in piece.features if isinstance(f, Ramp)):
        assert ramp.cell in piece.cells, '%s: a belt tile is off the board' % variant


# -- and the rule it imposes ---------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_a_slow_marble_leaves_the_belt_at_the_belt_s_speed(variant):
    """Which is the whole reason a story puts one in front of a chimney."""
    piece = _built(variant)
    belt_speed = SETTINGS[variant]['speed']
    left = _left_at(piece, 2.0)
    assert left is not None, '%s: %s' % (variant, _say(left))
    assert left >= belt_speed - 1.0, \
        '%s: arriving at 2 m/s the marble left at %s, and the belt runs at ' \
        '%.1f m/s' % (variant, _say(left), belt_speed)


def test_the_belt_never_brakes_a_marble_that_is_already_faster():
    """A cap, not a governor: a chapter that spent effort on speed keeps it."""
    piece = _built('plain')
    belt_speed = SETTINGS['plain']['speed']
    fast = _left_at(piece, belt_speed + 10.0)
    assert fast is not None, _say(fast)
    assert fast > belt_speed + 5.0, \
        'arriving at %.1f m/s the marble left the %.1f m/s belt at %s' \
        % (belt_speed + 10.0, belt_speed, _say(fast))


def test_what_a_marble_leaves_at_barely_depends_on_what_it_arrived_at():
    """The point of the piece: the chapter after it can count on a speed."""
    piece = _built('plain')
    seen = {speed: _left_at(piece, speed) for speed in (1.0, 4.0, 8.0, 11.0)}
    report = ', '.join('in %.0f -> out %s' % (speed, _say(left))
                       for speed, left in sorted(seen.items()))
    assert None not in seen.values(), report
    spread = max(seen.values()) - min(seen.values())
    assert spread < 3.0, 'the belt left a %.1f m/s spread across arrivals — %s' \
        % (spread, report)
