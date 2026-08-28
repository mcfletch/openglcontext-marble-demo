"""The banked turn, and whether arriving quickly is rewarded or punished.

The other corner in the library is :func:`~openglcontext_marble_demo.pieces.hairpin`,
which is the same right-angle with its outside left open, so it is the control:
both are driven by the game's own
:class:`~openglcontext_marble_demo.pilot.Autopilot`, at the same speeds, and
what is compared is whether the marble came round and how long it took.

A rule that is a reward is measured the other way up from one that is a
punishment.  The question is not "at what speed does this fail" -- it is whether
going faster costs anything at all.
"""
import random

import numpy as np

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Ramp, Wall

SOUTH = (0, 1)

#: How far along counts as round: the exit is the last cell of the piece and a
#: marble arriving at it runs off the end of a piece standing alone.
THROUGH = 0.8

#: The contact impulse per unit mass at which
#: :class:`~openglcontext_marble_demo.controller.MarbleController` stops calling
#: a wall contact a lean and starts calling it a crash.
CRASH = 3.0


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('banked_turn', random.Random(seed), _entry(),
                           variant=variant, **named)


def _drive(piece, speed, seconds=25.0, dt=1 / 120.0):
    """Roll a marble on at ``speed`` and let the autopilot take it round.

    A corner has to be driven to be measured: nothing that is not steering gets
    round one, because the board leans one way and the way out points another.
    Answers how far it ever got, how long it took, whether it left the piece,
    and how many wall contacts were hard enough to count as crashes.
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
    furthest, crashes = 0.0, 0
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
        if game.controller.fall_count:
            return dict(furthest=furthest, took=None, crashes=crashes, fell=True)
        if furthest >= THROUGH:
            return dict(furthest=furthest, took=step * dt, crashes=crashes, fell=False)
    return dict(furthest=furthest, took=None, crashes=crashes, fell=False)


def _along(piece, cell):
    return ((cell[0] - piece.entry.cell[0]) * piece.entry.facing[0]
            + (cell[1] - piece.entry.cell[1]) * piece.entry.facing[1])


# -- the shape ------------------------------------------------------------------

def test_the_outside_of_the_bend_is_built_up():
    """A corner with a flat floor is a corner; the bank is what makes this one."""
    piece = _built()
    furthest = max(_along(piece, cell) for cell in piece.cells)
    outside = [height for cell, height in piece.cells.items()
               if _along(piece, cell) == furthest]
    assert min(outside) > piece.entry.height, \
        'the far side of the bend sits at %r, the mouth at %.2f' \
        % (sorted(set(outside)), piece.entry.height)


def test_every_rise_of_the_bank_is_a_surface_rather_than_a_step():
    """Terraces at different heights are a staircase, and a marble meets each
    riser as a wall: every cell the bank steps up from carries a ramp."""
    piece = _built()
    ramped = {feature.cell for feature in piece.features if isinstance(feature, Ramp)}
    for (col, row), height in piece.cells.items():
        for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            beside = piece.cells.get((col + dcol, row + drow), height)
            if beside > height + 1e-6:
                assert (col, row) in ramped, \
                    'cell %r steps up %.2f with nothing to roll up' % ((col, row),
                                                                       beside - height)


def test_the_mouths_are_at_the_height_the_piece_was_handed():
    """The bank is the outside of the bend and not the way in or the way out: a
    piece raised at its mouth is a step in the middle of a board."""
    entry = _entry(cell=(4, 9), height=-2.7)
    piece = fragments.build('banked_turn', random.Random(3), entry)
    for cell in entry.cells():
        assert piece.cells[cell] == entry.height, cell
    assert piece.cells[piece.exit.cell] == piece.exit.height


def test_a_banked_turn_is_walled_where_it_is_not_floor():
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


def test_the_same_turn_twice():
    assert _built().cells == _built().cells


# -- the rule -------------------------------------------------------------------

def test_a_banked_turn_does_not_ask_a_player_to_brake():
    """The rule, and it is a reward: arriving quickly is the quick way round."""
    gentle = _drive(_built(), 3.0)
    quick = _drive(_built(), 16.0)
    assert not gentle['fell'] and not quick['fell'], \
        'the piece lost a marble: 3 m/s %r, 16 m/s %r' % (gentle, quick)
    assert gentle['took'] is not None and quick['took'] is not None, \
        ('not both came round: 3 m/s reached %.2f, 16 m/s reached %.2f'
         % (gentle['furthest'], quick['furthest']))
    assert quick['took'] < gentle['took'], \
        ('braking would have paid: 3 m/s came round in %.2f s, 16 m/s in %.2f'
         % (gentle['took'], quick['took']))


def test_it_holds_a_marble_that_a_right_angle_loses():
    """The same driver, the same speed, the corner whose outside is left open.

    :func:`~openglcontext_marble_demo.pieces.hairpin` lays two legs across each
    other, so the outside of its bend is not there; this one lays the square
    whole and walls it.  That is the difference the piece exists for.
    """
    speed = 12.0
    open_corner = _drive(pieces.hairpin(random.Random(3), _entry()), speed)
    banked = _drive(_built(), speed)
    assert open_corner['fell'] or open_corner['took'] is None, \
        ('the open corner held a marble at %.0f m/s, so there is nothing to hold: '
         'it reached %.2f in %s s' % (speed, open_corner['furthest'],
                                      open_corner['took']))
    assert banked['took'] is not None and not banked['fell'], \
        ('the banked turn lost it too at %.0f m/s: reached %.2f, fell %s'
         % (speed, banked['furthest'], banked['fell']))


def test_longer_legs_make_a_gentler_corner():
    """What a layout variant is for: ``sweeping`` gives six cells of approach
    where ``plain`` gives four, and the marble arrives already turning."""
    tight = _drive(_built('plain'), 8.0)
    sweeping = _drive(_built('sweeping'), 8.0)
    assert sweeping['crashes'] < tight['crashes'], \
        ('at 8 m/s: plain crashed %d times, sweeping %d'
         % (tight['crashes'], sweeping['crashes']))
