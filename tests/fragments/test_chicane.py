"""The chicane, and whether its rhythm is a rule or a decoration.

A gate a marble can be anywhere for is not a gate, so the questions here are
answered by driving the piece rather than by reading its cells: the game's own
:class:`~openglcontext_marble_demo.pilot.Autopilot` is put on it at one speed
and then at another, and what is counted is how hard it had to meet a wall and
how long it took to come out the far side.

A wall met above the controller's ``wall_impact_speed`` costs nine tenths of the
speed that went into it, which is what the chicane charges for arriving too
fast to be where the next gate is.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Wall

SOUTH = (0, 1)

#: How far along counts as through.  Not 1.0: the exit is the last cell of the
#: piece and a marble arriving at it runs off the end of a piece standing alone.
THROUGH = 0.8

#: The contact impulse per unit mass above which
#: :class:`~openglcontext_marble_demo.controller.MarbleController` calls a wall
#: contact a crash and takes nine tenths of the speed.  Below it, a wall is
#: something to lean on.
CRASH = 3.0


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('chicane', random.Random(seed), _entry(), variant=variant,
                           **named)


def _drive(piece, speed, seconds=25.0, dt=1 / 120.0):
    """Roll a marble on at ``speed`` and let the autopilot take it through.

    A chicane is a rule about being in the right place, so it has to be driven
    by something that is trying to be: an unsteered marble stops at the first
    gate whatever speed it arrived with, which measures nothing.  Answers how
    far it ever got, how long it took to get through, and how many wall contacts
    were hard enough to count as crashes.
    """
    level = piece.level(time_limit=900.0)
    game = MarbleGame(level)
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    begin = np.array(level.cell_center(piece.entry.cell))
    span = np.array(level.cell_center(piece.exit.cell)) - begin
    whole = float(np.dot(span, span)) or 1.0
    furthest, took, crashes = 0.0, None, 0
    for step in range(int(seconds / dt)):
        game.lean(*driver.lean(world.position[index], world.linear_velocity[index]))
        game.advance(dt)
        mass = world.mass[index]
        for contact in world.contacts:
            if index in (contact.a, contact.b) and abs(contact.normal[1]) <= 0.7 \
                    and contact.normal_impulse / mass >= CRASH:
                crashes += 1
        at = world.position[index]
        furthest = max(furthest, float(np.dot(np.array([at[0], at[2]]) - begin, span)
                                       / whole))
        if furthest >= THROUGH:
            took = step * dt
            break
    return dict(furthest=furthest, took=took, crashes=crashes)


# -- the shape ------------------------------------------------------------------

def test_a_chicane_has_no_lane_that_runs_straight_through_it():
    """The whole of a chicane is that the way in is not pointed at the way out."""
    piece = _built()
    rows = {row for _, row in piece.cells}
    for col in {col for col, _ in piece.cells}:
        mine = {row for column, row in piece.cells if column == col}
        assert mine != rows, 'column %d runs the length of the piece' % col


def test_a_chicane_comes_back_to_the_line_it_started_on():
    """A piece that left the marble a lane across from where it took it would be
    a shift rather than a chicane, and the piece after it would be built askew."""
    piece = _built()
    across = piece.entry.across()
    assert (piece.exit.cell[0] - piece.entry.cell[0]) * across[0] \
        + (piece.exit.cell[1] - piece.entry.cell[1]) * across[1] == 0
    assert piece.exit.facing == piece.entry.facing


def test_a_chicane_is_walled_all_the_way_round():
    """The walls are what it costs: a bend with nothing at the outside of it is
    a cliff, and getting the rhythm wrong there would end the run."""
    piece = _built()
    walled = {(feature.cell, feature.side) for feature in piece.features
              if isinstance(feature, Wall)}
    mouths = set(piece.entry.cells()) | set(piece.exit.cells())
    for cell in piece.cells:
        if cell in mouths:
            continue
        for (dcol, drow), side in (((0, 1), 'S'), ((0, -1), 'N'),
                                   ((1, 0), 'E'), ((-1, 0), 'W')):
            if (cell[0] + dcol, cell[1] + drow) not in piece.cells:
                assert (cell, side) in walled, \
                    'cell %r is open to the void on its %s side' % (cell, side)


def test_a_chicane_is_the_same_chicane_twice():
    assert _built().cells == _built().cells


# -- the rule -------------------------------------------------------------------

@pytest.mark.xfail(reason='the walls stopped catching anything when the board became aimable. Rolled straight the marble is stopped at 0.26 of the way at every speed from 4 to 20 m/s -- it cannot cross at all without steering, which is what a chicane is for -- and driven by the pilot it crosses at every speed and faster the quicker it arrives: 23.9 s at 3 m/s down to 21.0 s at 18. So the rhythm is not a rule any more; the piece is a slow way round rather than a fast way that can be got wrong.', strict=True)
def test_a_chicane_costs_nothing_at_a_sensible_speed_and_the_walls_at_a_fast_one():
    """The rule: the time to move over is the length of a leg divided by the
    speed, so past a certain speed there is not enough of it."""
    steady = _drive(_built(), 8.0)
    quick = _drive(_built(), 14.0)
    assert steady['crashes'] == 0, \
        'a steady 8 m/s hit the walls %d times' % steady['crashes']
    assert quick['crashes'] >= 1, \
        '14 m/s went through untouched, so the rhythm is not a rule'
    assert steady['took'] is not None and quick['took'] is not None, \
        'not both got through: 8 m/s %s, 14 m/s %s' % (steady['took'], quick['took'])
    assert quick['took'] > steady['took'], \
        ('the fast entry was not the slower way round: 8 m/s took %.2f s with %d '
         'crashes, 14 m/s took %.2f s with %d'
         % (steady['took'], steady['crashes'], quick['took'], quick['crashes']))


@pytest.mark.xfail(reason='neither leg length catches anything now: plain (four-cell legs) and tight (three-cell) both crash 0 times at 10 m/s. Driven, tight crosses in 18.8 s against plain 21.9, so what the shorter leg buys is a quicker way through rather than a harder one.', strict=True)
def test_a_shorter_leg_breaks_the_rhythm_at_a_lower_speed():
    """What a layout variant is for: the same rule, biting sooner.

    ``tight`` leaves three cells between one gate and the next where ``plain``
    leaves four, and three cells at 10 m/s is not long enough to move over.
    """
    roomy = _drive(_built('plain'), 10.0)
    tight = _drive(_built('tight'), 10.0)
    assert roomy['crashes'] == 0 and tight['crashes'] >= 1, \
        ('10 m/s: plain (four-cell legs) crashed %d times, tight (three-cell) %d'
         % (roomy['crashes'], tight['crashes']))
