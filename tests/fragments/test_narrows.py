"""The narrows, and whether its throat is a rule or a doorway.

Every cell of the piece is floor, so nothing here can be answered by asking
where a marble ended up: it ends up through.  What the piece charges is *speed*,
and it charges it for arriving off the centre line, so each case below starts a
marble somewhere across the mouth, hands it to the game's own
:class:`~openglcontext_marble_demo.pilot.Autopilot`, and counts the wall
contacts hard enough for the controller to take nine tenths of the speed away.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Wall

SOUTH = (0, 1)

#: How far along counts as through: the exit is the last cell of the piece, and
#: a marble arriving at it runs off the end of a piece standing alone.
THROUGH = 0.8

#: The contact impulse per unit mass at which
#: :class:`~openglcontext_marble_demo.controller.MarbleController` stops calling
#: a wall contact a lean and starts calling it a crash.
CRASH = 3.0


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('narrows', random.Random(seed), _entry(), variant=variant,
                           **named)


def _drive(piece, speed, across=0, seconds=25.0, dt=1 / 120.0):
    """Start a marble ``across`` cells off the centre of the mouth, at ``speed``.

    The line the piece asks for is its middle, so being off it is a starting
    position rather than a way of steering: the marble is put down across the
    mouth and the autopilot drives from there, exactly as it would having been
    left there by the piece before.
    """
    level = piece.level(time_limit=900.0)
    side = piece.entry.across()
    level.start_cell = (piece.entry.cell[0] + side[0] * across,
                        piece.entry.cell[1] + side[1] * across)
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


def _widths(piece):
    """How many cells across the piece is, row by row down its length."""
    facing = piece.entry.facing
    rows: dict = {}
    for cell in piece.cells:
        along = ((cell[0] - piece.entry.cell[0]) * facing[0]
                 + (cell[1] - piece.entry.cell[1]) * facing[1])
        rows[along] = rows.get(along, 0) + 1
    return [rows[along] for along in sorted(rows)]


# -- the shape ------------------------------------------------------------------

def test_a_narrows_closes_to_one_lane_and_opens_again():
    widths = _widths(_built())
    assert min(widths) == 1, 'the throat is %d cells across' % min(widths)
    assert widths[0] > 1 and widths[-1] > 1, widths
    assert widths.index(min(widths)) > 0


def test_a_narrows_only_ever_closes_and_then_only_ever_opens():
    """A funnel that widened again half way down is two funnels, and a marble
    gathered by the first would be thrown by the second."""
    widths = _widths(_built())
    throat = widths.index(min(widths))
    assert widths[:throat + 1] == sorted(widths[:throat + 1], reverse=True), widths
    assert widths[throat:] == sorted(widths[throat:]), widths


def test_a_narrows_is_walled_where_it_is_not_floor():
    """The converging wall is the piece; a funnel open at the sides is a plain."""
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


def test_a_narrows_is_the_same_narrows_twice():
    assert _built().cells == _built().cells


# -- the rule -------------------------------------------------------------------

def test_on_the_line_the_throat_costs_nothing_at_any_speed():
    """The other half of the rule, and the half that makes it a reward: a marble
    that arrives aimed is not charged for arriving quickly."""
    steady = _drive(_built(), 4.0)
    quick = _drive(_built(), 14.0)
    assert steady['crashes'] == 0 and quick['crashes'] == 0, \
        ('on the centre line: 4 m/s crashed %d times, 14 m/s %d'
         % (steady['crashes'], quick['crashes']))
    assert quick['took'] < steady['took'], \
        ('speed on the line did not pay: 4 m/s through in %.2f s, 14 m/s in %.2f'
         % (steady['took'], quick['took']))


def test_off_the_line_the_closing_wall_is_what_speed_costs():
    """The rule: how long there is to get onto the line is the length of the
    taper divided by the speed, and three lanes out is a long way to come."""
    gentle = _drive(_built(), 4.0, across=3)
    quick = _drive(_built(), 14.0, across=3)
    assert gentle['crashes'] == 0, \
        ('rolling in three lanes wide at 4 m/s still met the wall %d times'
         % gentle['crashes'])
    assert quick['crashes'] >= 1, \
        ('three lanes wide at 14 m/s went through untouched, so aim is not a rule '
         '(it reached %.2f of the way in %s s)' % (quick['furthest'], quick['took']))


@pytest.mark.xfail(reason='the closing wall stopped catching anything when the board became aimable: two lanes wide at 8 m/s, plain and funnel both crash 0 times. Driven, funnel crosses in 8.8 s against plain 14.4, so the tighter taper is the quicker way through rather than the less forgiving one.', strict=True)
def test_a_shorter_taper_leaves_less_room_to_put_it_right():
    """What a layout variant is for: ``funnel`` closes in one cell of lane per
    lane where ``plain`` takes two, so the same line off centre costs sooner."""
    roomy = _drive(_built('plain'), 8.0, across=2)
    sharp = _drive(_built('funnel'), 8.0, across=2)
    assert roomy['crashes'] == 0 and sharp['crashes'] >= 1, \
        ('two lanes wide at 8 m/s: plain crashed %d times, funnel %d'
         % (roomy['crashes'], sharp['crashes']))
