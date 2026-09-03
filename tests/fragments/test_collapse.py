"""The collapsing floor: rolled across, and rolled into.

The rule is a clock on the tile, not a speed gate -- a marble that is off the
far edge quickly never puts enough time on the floor to matter, and one that
dawdles does, however fast or slow it happens to be moving at the moment the
floor finally gives.  Both crossings are driven on the real physics, through
:class:`~openglcontext_marble_demo.game.MarbleGame`, so the seconds reported
are the seconds a player would spend.
"""
import random

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.fragments.collapse import VARIANTS
from openglcontext_marble_demo.game import PLAYING, WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.collapse import Collapse, CollapseWatch

SOUTH = (0, 1)
DT = 1 / 120.0


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('collapse', random.Random(seed), _entry(), variant=variant,
                           **named)


def _watch(game):
    return next(a for a in game.build.animators if isinstance(a, CollapseWatch))


def _level(piece, exit_name):
    """``piece`` as a playable level whose finish is one of its named exits.

    ``Piece.level()`` always finishes at ``'ok'``, which sits at the floor's
    own far edge here -- too close to measure a fall through it by, since a
    body flung across the hole can clear the finish trigger while still in the
    air.  Finishing at ``'missed'`` instead is what proves a fall through
    actually reaches the slower route, rather than merely not being lost.
    """
    exit_cell = piece.exits[exit_name].cell
    return Level(name='collapse-test', cells=dict(piece.cells), start_cell=piece.entry.cell,
                finish_cell=exit_cell, time_limit=900.0,
                features=[*piece.features, Finish(exit_cell)])


def _cross(piece, speed, exit_name='ok', seconds=8.0, dt=DT):
    """Roll a marble onto the piece at ``speed`` and drive the real game.

    Answers whether the floor gave way, how long the marble dwelt on it, the
    game's state at the end, and how many times it fell or was lost.
    """
    game = MarbleGame(_level(piece, exit_name), base_tilt=fragments.DESIGN_TILT)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    watch = _watch(game)
    dwell = 0.0
    for _ in range(int(seconds / dt)):
        state = game.advance(dt)
        if watch.holds(world.position[index]):
            dwell += dt
        if state != PLAYING:
            break
    return dict(gave_way=watch.panel.released_at is not None, dwell=dwell,
               state=game.state, fall_count=game.controller.fall_count,
               loss_count=game.controller.loss_count)


# -- the shape --------------------------------------------------------------

def test_every_variant_carries_the_collapsing_floor():
    for variant in VARIANTS:
        piece = _built(variant)
        assert any(isinstance(f, Collapse) for f in piece.features), variant


def test_the_floor_is_passable_from_entry_to_the_ok_exit():
    piece = _built()
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell)


def test_falling_through_has_its_own_exit():
    piece = _built()
    assert 'missed' in piece.exits
    assert piece.exits['missed'].cell in piece.cells
    assert piece.exits['missed'].height < piece.entry.height, (
        'the way past a collapsed floor does not go anywhere lower')


def test_the_descent_past_the_floor_holds_the_slope_budget():
    """The registry checks every fragment's own cells; this is the same
    question asked about the one piece this file is about."""
    piece = _built()
    for (col, row), height in piece.cells.items():
        for dcol, drow in ((1, 0), (0, 1)):
            beside = (col + dcol, row + drow)
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9

def test_the_same_floor_for_the_same_seed():
    assert _built().cells == _built().cells


# -- the rule: a clock, not a speed gate -------------------------------------

def test_crossing_at_pace_leaves_the_floor_standing():
    piece = _built('plain')
    result = _cross(piece, speed=12.0)
    hold_time = VARIANTS['plain']['hold_time']
    assert not result['gave_way'], (
        'crossed with %.2f s on the floor and it gave way anyway, against a '
        '%.2f s hold' % (result['dwell'], hold_time))
    assert result['dwell'] < hold_time, (
        'the quick crossing put %.2f s on the floor, at or over the %.2f s hold: '
        'it is not testing what it claims to' % (result['dwell'], hold_time))
    assert result['state'] == WON, (
        'crossed at pace in %.2f s of dwell and the run did not reach the finish '
        '(state=%s)' % (result['dwell'], result['state']))


def test_dawdling_drops_you_through_and_you_carry_on():
    piece = _built('plain')
    result = _cross(piece, speed=0.3, exit_name='missed', seconds=10.0)
    hold_time = VARIANTS['plain']['hold_time']
    assert result['gave_way'], (
        '%.2f s dawdling on the floor and it never gave way, against a %.2f s hold'
        % (result['dwell'], hold_time))
    assert result['dwell'] >= hold_time
    assert result['fall_count'] == 0 and result['loss_count'] == 0, (
        'dwelt %.2f s, the floor gave way, and the run ended instead of '
        'continuing onto the slower route' % result['dwell'])
    assert result['state'] == WON, (
        'the floor gave way after %.2f s of dwell but the marble never reached '
        'the way down past it (state=%s)' % (result['dwell'], result['state']))


def test_the_hold_is_what_separates_them_and_not_the_route():
    """Both crossings take the same line down the middle of the same piece;
    only the speed differs."""
    piece = _built('plain')
    quick = _cross(piece, speed=20.0, exit_name='missed', seconds=10.0)
    slow = _cross(piece, speed=0.3, exit_name='missed', seconds=10.0)
    assert (quick['gave_way'], slow['gave_way']) == (False, True)
    assert slow['dwell'] > quick['dwell'] * 2, (
        'quick dwelt %.2f s, dawdling dwelt %.2f s: too close together to be '
        'about the time on the floor' % (quick['dwell'], slow['dwell']))


# -- reset --------------------------------------------------------------------

def test_a_restarted_run_finds_the_floor_whole_again():
    piece = _built('plain')
    level = piece.level(time_limit=900.0)
    game = MarbleGame(level, base_tilt=fragments.DESIGN_TILT)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * 0.3, 0.0, facing[1] * 0.3)
    world.wake(index)
    watch = _watch(game)
    for _ in range(int(10.0 / DT)):
        game.advance(DT)
        if watch.panel.released_at is not None:
            break
    assert watch.panel.released_at is not None, 'the floor never gave way to reset'
    game.reset()
    assert watch.panel.released_at is None, 'a restarted run still finds the floor open'
    assert not watch.held, 'a restarted run remembers time already spent on the floor'


def test_the_floor_is_something_a_restarted_run_asks_to_forget():
    piece = _built('plain')
    game = MarbleGame(piece.level(time_limit=900.0))
    assert any(isinstance(item, CollapseWatch) for item in game.build.resettable)
