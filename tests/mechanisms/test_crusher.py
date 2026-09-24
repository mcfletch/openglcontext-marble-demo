"""The crusher: a press that strikes on a cycle, and what stopping under it costs.

The mechanism does no crushing of its own -- it swings a kinematic slab up and
down, and :class:`~openglcontext_marble_demo.controller.MarbleController` reads
the squeeze off the marble's contacts, exactly as it would for a marble
caught between any two closing surfaces.  So what these tests hold the press to
is the same rhythm as ``tests/mechanisms/test_burner.py`` holds the fire to:
build the real board, run the real physics, and show both halves of the rule
with the numbers the run produced.
"""
from openglcontext_marble_demo import fragments, levelfile, mechanisms
from openglcontext_marble_demo.controller import CRUSHED, DESTROYED
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.crusher import Crusher

STEP = 1 / 120.0
#: A corridor along +Z (the board's own downhill direction), one press at row 3.
PRESS_CELL = (0, 3)


def _level(features):
    cells = {(0, row): 0.0 for row in range(-2, 12)}
    return Level(name='crusher', cells=cells, start_cell=(0, -2), finish_cell=(0, 11),
                 time_limit=120.0, features=[*features, Finish((0, 11))])


def _game(**press):
    return MarbleGame(_level([Crusher(cells=(PRESS_CELL,), **press)]),
                      base_tilt=fragments.DESIGN_TILT)


def _the_press(game):
    return next(f for f in game.level.features if isinstance(f, Crusher))


# -- the whole point: stopping under the press is what it catches --------------

def test_a_marble_parked_under_the_press_is_crushed():
    game = _game()
    press = _the_press(game)
    world, marble = game.scene.world, game.marble.index
    x, z = game.level.cell_center(PRESS_CELL)
    world.place_body(marble, position=(x, game.radius + 0.001, z))
    world.linear_velocity[marble] = (0.0, 0.0, 0.0)
    world.wake(marble)

    limit = press.strike * 2.0 + press.dwell + 0.5
    crushed_at = None
    for i in range(int(limit / STEP)):
        game.advance(STEP)
        if game.controller.state == DESTROYED:
            crushed_at = i * STEP
            break
    assert crushed_at is not None, (
        'never crushed in %.2f s parked under the press (period %.2f s, strike '
        '%.2f s, dwell %.2f s)' % (limit, press.period, press.strike, press.dwell))
    assert game.controller.last_loss == CRUSHED, (
        'lost to %r instead of being crushed' % (game.controller.last_loss,))
    # A blow is one down-stroke, the dwell, and one up-stroke; a marble that
    # never leaves has to be caught well inside that, or the dwell is not
    # doing the job ``crush_time`` needs of it.
    assert crushed_at < press.strike * 2.0 + press.dwell, (
        'took %.3f s to crush a marble that never moved, longer than the '
        '%.2f s blow that should have done it'
        % (crushed_at, press.strike * 2.0 + press.dwell))
    assert crushed_at >= game.controller.crush_time * 0.5, (
        'crushed after %.3f s, suspiciously quicker than half of crush_time '
        '(%.2f s) -- the squeeze may be getting counted before it is real'
        % (crushed_at, game.controller.crush_time))


def test_a_marble_crossing_while_the_press_is_freshly_up_survives_it():
    """Started just after a blow, the press is not due again for a while --
    a marble rolling through has nothing to read and nothing to fear."""
    game = _game()
    level = game.level
    for animator in game.build.animators:
        animator.time = 0.0     # the instant a blow ends: the whole period ahead
    world, marble = game.scene.world, game.marble.index
    x, z = level.cell_center(PRESS_CELL)
    world.place_body(marble, position=(x, game.radius + 0.001, z - 3 * level.cell_size))
    world.linear_velocity[marble] = (0.0, 0.0, 3.0)
    world.wake(marble)
    for _ in range(int(20.0 / STEP)):
        game.advance(STEP)
        if game.controller.state == DESTROYED or game.state == WON:
            break
    assert game.controller.state != DESTROYED, (
        'a marble crossing at 3.0 m/s just after a blow was crushed anyway')
    assert game.state == WON, 'never reached the finish either: stuck, not caught'


# -- what the mechanism is, structurally ----------------------------------------

def test_the_press_hangs_above_the_track_rather_than_replacing_it():
    assert Crusher(cells=(PRESS_CELL,)).owned_cells() == set()


def test_the_press_lies_only_where_the_track_has_a_floor():
    """The far cell has no floor at all -- the same guard ``Burner`` carries,
    and for the same reason: a press needs a floor to lie in."""
    game = MarbleGame(_level([Crusher(cells=(PRESS_CELL, (0, 40)))]),
                      base_tilt=fragments.DESIGN_TILT)
    assert len(game.build.animators) == 1


def test_the_press_needs_no_reset_of_its_own():
    """Purely cyclic, like ``Elevator`` and ``RotatingArm`` -- nothing about a
    marble's own run changes when the next blow lands, so there is nothing a
    restarted run has to be told to forget."""
    assert _game().build.resettable == []


def test_the_press_is_registered_under_its_own_name():
    assert mechanisms.registry()['crusher'] is Crusher


def test_the_press_round_trips_through_the_file_format():
    level = _level([Crusher(cells=(PRESS_CELL,), period=2.5, strike=0.2, dwell=0.35,
                            clearance=1.4, gap=0.15)])
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features == level.features
