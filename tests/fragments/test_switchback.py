"""The zigzag: the corners it puts at the foot of its descents, and what they ask.

Two questions are being asked here and they need different instruments. The
corner is a question about arrival speed, so a marble is put on the piece at a
speed and the answer is whether it came round or went over the edge. Whether the
*descents* are what make a marble fast enough to fail is a question about the
run as a whole, and it is asked by comparing what the marble carries on the first
descent against the second.

Driving is done by the autopilot, which leans the board through the same rig a
player's keys do. A right-angle with nothing beyond it is not a corner an
unsteered marble ever rounds — the board leans straight on and nothing else turns
it — so a marble simply released would measure the absence of a driver rather
than anything about the piece.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import Wall

STEP = 1 / 120.0

_SIDE = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}


def _entry():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('switchback', random.Random(seed), _entry(),
                           variant=variant, **named)


def _driven(piece, speed=0.0, seconds=45.0, **named):
    """Play the piece with the autopilot; answer ``(seconds, falls, trail)``.

    ``trail`` is one ``(height, speed)`` a frame, which is what the accumulation
    is read off: the height says which descent the marble is on.  The pilot
    drives as well as steers, since the board is level until it leans one and
    nothing else moves the marble, so it is flown as it is tuned.
    """
    level = piece.level(time_limit=600.0)
    game = MarbleGame(level)
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis, **named)
    world, index = game.scene.world, game.marble.index
    if speed:
        facing = piece.entry.facing
        world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
        world.wake(index)
    trail = []
    played = 0.0
    for _ in range(int(seconds / STEP)):
        game.lean(*driver.lean(world.position[index],
                               world.linear_velocity[index]))
        at = world.position[index]
        moving = world.linear_velocity[index]
        trail.append((float(at[1]), float(np.hypot(moving[0], moving[2]))))
        state = game.advance(STEP)
        played += STEP
        if state != 'playing':
            break
    return (played if game.state == WON else None,
            game.controller.fall_count, trail)


@pytest.fixture(scope='module')
def driven():
    """A free-rolling run down the one-descent piece and the two-descent one."""
    return {legs: _driven(_built(legs=legs)) for legs in (1, 2)}


@pytest.fixture(scope='module')
def at_the_corner():
    """One run at the single corner for each of two arrival speeds."""
    piece = _built(legs=1)
    return {speed: _driven(piece, speed) for speed in (14.0, 16.0)}


# -- what the piece is made of -------------------------------------------------

def test_a_switchback_turns_one_way_and_then_the_other():
    """Down, hard across, down, hard across: the second crossing goes back."""
    piece = _built()
    across = piece.entry.across()
    reach = [cell[0] * across[0] + cell[1] * across[1] for cell in piece.cells]
    assert max(reach) - min(reach) >= 4, \
        'the piece only spans %d cells across the board' % (max(reach) - min(reach))
    home = piece.exit.cell[0] * across[0] + piece.exit.cell[1] * across[1]
    start = piece.entry.cell[0] * across[0] + piece.entry.cell[1] * across[1]
    assert abs(home - start) <= 1, \
        'the way out is %d cells off the way in' % abs(home - start)


@pytest.mark.parametrize('variant',
                         sorted(fragments.library()['switchback'].variants))
def test_a_switchback_descends_the_whole_way(variant):
    piece = _built(variant)
    assert piece.exit.height < piece.entry.height - 1.0, variant


def _openings(piece):
    """The cells with an unwalled face onto nothing that are not a mouth."""
    mouths = set(piece.entry.cells()) | set(piece.exit.cells())
    walled = {(wall.cell, wall.side) for wall in piece.features
              if isinstance(wall, Wall)}
    return {cell for cell in piece.cells
            if cell not in mouths
            and any((cell[0] + step[0], cell[1] + step[1]) not in piece.cells
                    and (cell, side) not in walled
                    for side, step in _SIDE.items())}


@pytest.mark.parametrize('legs', (1, 2))
def test_each_descent_ends_in_an_opening_and_nothing_else_is_open(legs):
    """Straight on at a corner is off the board, and that is the whole piece.
    Everywhere else is fenced, so a marble that turns is a marble that is safe.

    One opening a descent: they are counted by the height they stand at, since
    each descent ends at its own, and the inside of a turn is not open because
    the crossing itself is what covers it.
    """
    piece = _built(legs=legs)
    openings = _openings(piece)
    assert openings, 'the piece is fenced all round: nothing can be carried off it'
    heights = {round(piece.cells[cell], 3) for cell in openings}
    assert len(heights) == legs, \
        '%d cells stand open at %d heights, where %d descents were built: %s' \
        % (len(openings), len(heights), legs, sorted(openings))
    for cell in openings:
        assert piece.cells[cell] < piece.entry.height, \
            'the opening at %r is level with the way in, so it is not the foot ' \
            'of anything' % (cell,)


# -- and what the corner asks --------------------------------------------------

@pytest.mark.xfail(reason='arriving at 16 m/s costs 0 falls now: on the gentler board -- 8 degrees of lean rather than 12.4, which is what made the game aimable -- a marble reaches the corner slowly enough to turn whatever it entered with. Driven by the pilot the piece is crossed at every speed from 3 to 18 m/s and quicker the faster it arrives, 20.5 s down to 17.6.', strict=True)
def test_a_corner_can_be_taken_at_speed_but_not_at_any_speed(at_the_corner):
    """The rule: the descents put the corner where a marble is fastest, and past
    a point no amount of steering brings it round."""
    made, missed = at_the_corner[14.0], at_the_corner[16.0]
    assert made[1] == 0, \
        'arriving at 14 m/s cost %d falls' % made[1]
    assert missed[1] >= 1, \
        'arriving at 16 m/s cost %d falls, so nothing is ever carried past the ' \
        'corner' % missed[1]
    assert missed[0] > made[0] + 1.0, \
        'arriving at 14 m/s took %.2f s and at 16 m/s %.2f s, so going over the ' \
        'edge costs nothing' % (made[0], missed[0])


def test_a_switchback_can_be_driven_from_one_end_to_the_other(driven):
    for legs, (took, falls, _) in driven.items():
        assert took is not None, 'the autopilot never got down the %d-leg one' % legs
        assert falls == 0, 'the %d-leg one cost %d falls free-rolling' % (legs, falls)


def test_the_two_descent_one_costs_more_of_the_clock(driven):
    short, long_way = driven[1][0], driven[2][0]
    assert long_way > short + 4.0, \
        'one descent took %.2f s and two took %.2f s' % (short, long_way)


@pytest.mark.xfail(strict=True, reason=(
    'a driven marble settles at about 5 m/s on a descent and stays there '
    'however far it has fallen, because steering across the board lean spends '
    'the pull that would otherwise be speed: measured on switchback/plain, 5.08 '
    'm/s on the first descent and 5.15 on the second, and 4.6 to 5.1 over every '
    'shelf of a piece built with three, five or seven cells of descent a leg. '
    'The corner does bite at 16 m/s, so what it bites on is the speed a marble '
    'was given by whatever came before rather than any the piece builds up.'))
def test_the_second_descent_hands_the_corner_more_speed_than_the_first(driven):
    """The rule the corners are placed for: the last one is the fast one."""
    foot = _built(legs=1).exit.height
    _, _, trail = driven[2]
    above = max([speed for height, speed in trail if height > foot + 0.6] or [0.0])
    below = max([speed for height, speed in trail if height <= foot + 0.6] or [0.0])
    assert below >= above + 1.0, \
        'the first descent tops out at %.2f m/s and the second at %.2f m/s, ' \
        'measured either side of %.2f m' % (above, below, foot)
