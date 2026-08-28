"""The stepping stones, and whether the air between them has to be paid for.

This piece is answered by rolling rather than by driving: the line across it is
straight, so there is nothing to steer at, and the only question is what the
marble arrived with.  Two things are measured -- the speed below which it goes
in, and what happens to a marble that had enough and then let it go.
"""
import random

import numpy as np

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import PLAYING, MarbleGame

SOUTH = (0, 1)

#: How far along counts as across: the exit is the last cell of the piece and a
#: marble arriving at it runs off the end of a piece standing alone.
THROUGH = 0.8


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('stepping_stones', random.Random(seed), _entry(),
                           variant=variant, **named)


def _roll(piece, speed, ease_off=None, seconds=10.0, dt=1 / 120.0):
    """Roll a marble on at ``speed`` down the middle of the stones.

    ``ease_off`` is how far along to start leaning the board back up the slope,
    which is the only brake a marble has and the thing the piece is about: pass
    a fraction to play it as somebody who committed and then thought better of
    it.  Answers how far it ever got and whether it went in.
    """
    level = piece.level(time_limit=900.0)
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    begin = np.array(level.cell_center(piece.entry.cell))
    span = np.array(level.cell_center(piece.exit.cell)) - begin
    whole = float(np.dot(span, span)) or 1.0
    furthest, took = 0.0, None
    for step in range(int(seconds / dt)):
        if ease_off is not None and furthest >= ease_off:
            game.lean(1.0, 0.0)                     # +forward is back up the slope
        state = game.advance(dt)
        at = world.position[index]
        furthest = max(furthest, float(np.dot(np.array([at[0], at[2]]) - begin, span)
                                       / whole))
        if took is None and furthest >= THROUGH:
            took = step * dt
        if game.controller.fall_count:
            return dict(furthest=furthest, fell=True, took=took)
        if state != PLAYING:                        # the last stone carries the finish
            return dict(furthest=furthest, fell=False, took=took)
    return dict(furthest=furthest, fell=False, took=took)


def _pick(piece, seconds=40.0, dt=1 / 120.0):
    """Let the game's own autopilot pick its way across from a standstill.

    Its route is the one the cells offer, which over these is the lip rather
    than the middle -- so this is the slow way round, played by the thing the
    game plays itself with.
    """
    level = piece.level(time_limit=1800.0)
    game = MarbleGame(level)
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis)
    world, index = game.scene.world, game.marble.index
    begin = np.array(level.cell_center(piece.entry.cell))
    span = np.array(level.cell_center(piece.exit.cell)) - begin
    whole = float(np.dot(span, span)) or 1.0
    furthest, took = 0.0, None
    for step in range(int(seconds / dt)):
        game.lean(*driver.lean(world.position[index], world.linear_velocity[index]))
        state = game.advance(dt)
        at = world.position[index]
        furthest = max(furthest, float(np.dot(np.array([at[0], at[2]]) - begin, span)
                                       / whole))
        if took is None and furthest >= THROUGH:
            took = step * dt
        if game.controller.fall_count:
            return dict(furthest=furthest, fell=True, took=took)
        if state != PLAYING:
            return dict(furthest=furthest, fell=False, took=took)
    return dict(furthest=furthest, fell=False, took=took)


def _along(piece, cell):
    return ((cell[0] - piece.entry.cell[0]) * piece.entry.facing[0]
            + (cell[1] - piece.entry.cell[1]) * piece.entry.facing[1])


# -- the shape ------------------------------------------------------------------

def test_there_is_air_between_the_stones():
    """A run of tiles with no gaps in it is a path."""
    piece = _built()
    across: dict = {}
    for cell in piece.cells:
        across[_along(piece, cell)] = across.get(_along(piece, cell), 0) + 1
    assert min(across.values()) == 1, \
        'the narrowest place is %d cells across' % min(across.values())
    assert max(across.values()) > 1


def test_no_lane_runs_the_length_of_the_stones():
    """The lip that laces one stone to the next changes sides, so keeping to an
    edge is not a way of walking across."""
    piece = _built()
    lengths = {_along(piece, cell) for cell in piece.cells}
    for col in {col for col, _ in piece.cells}:
        mine = {_along(piece, cell) for cell in piece.cells if cell[0] == col}
        assert mine != lengths, 'column %d runs the length of the piece' % col


def test_each_stone_is_a_terrace_below_the_one_before_it():
    piece = _built()
    assert piece.exit.height < piece.entry.height - 1.0
    assert piece.cells[piece.exit.cell] == piece.exit.height


def test_the_stones_are_one_region():
    """A piece that could only be jumped would end a run that arrived at it
    short of speed; the lip is where somebody who did can go."""
    piece = _built()
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell)


def test_nothing_stands_at_the_edge_of_a_stone():
    """A rail would make the gaps a corridor with holes in the floor."""
    assert not _built().features


def test_the_same_stones_twice():
    assert _built().cells == _built().cells


# -- the rule -------------------------------------------------------------------

def test_the_gaps_have_to_be_carried_across():
    """The rule: the gap is four metres of nothing, and how long the marble has
    to cross it is how long it takes to fall a metre."""
    short = _roll(_built(), 6.0)
    enough = _roll(_built(), 10.0)
    assert short['fell'], \
        '6 m/s crossed the gaps, reaching %.2f of the way' % short['furthest']
    assert not enough['fell'] and enough['furthest'] >= THROUGH, \
        ('10 m/s did not get across: %.2f of the way, fell %s'
         % (enough['furthest'], enough['fell']))


def test_easing_off_half_way_is_how_a_marble_goes_in():
    """The rule is about committing, so the interesting failure is not arriving
    slowly -- it is arriving fast and then thinking better of it."""
    held = _roll(_built(), 12.0)
    lifted = _roll(_built(), 12.0, ease_off=0.3)
    assert not held['fell'], \
        '12 m/s did not cross even held: %.2f of the way' % held['furthest']
    assert lifted['fell'], \
        ('leaning back from a third of the way in still got across at 12 m/s '
         '(%.2f of the way)' % lifted['furthest'])
    assert lifted['furthest'] < held['furthest']


def test_the_lip_is_a_way_across_and_it_is_the_slow_one():
    """What makes the gaps a decision rather than a wall: a marble with no speed
    at all can still get over, by weaving the corners the stones are laced at,
    and it takes the time the jump would have saved."""
    picked = _pick(_built())
    carried = _roll(_built(), 10.0)
    assert not picked['fell'] and picked['took'] is not None, \
        ('the autopilot could not pick its way across from a standstill: %.2f of '
         'the way, fell %s' % (picked['furthest'], picked['fell']))
    assert picked['took'] > carried['took'] * 2, \
        ('the slow way round is not slow: picked across in %.2f s, carried across '
         'in %.2f' % (picked['took'], carried['took']))


def test_a_wider_gap_asks_for_more_speed():
    """What an effect variant is for: ``wide`` puts two cells of air between the
    stones where ``plain`` puts one, and eight metres wants a run at it."""
    plain = _roll(_built('plain'), 10.0)
    wide = _roll(_built('wide'), 10.0)
    assert not plain['fell'] and wide['fell'], \
        ('at 10 m/s: plain reached %.2f (fell %s), wide reached %.2f (fell %s)'
         % (plain['furthest'], plain['fell'], wide['furthest'], wide['fell']))
