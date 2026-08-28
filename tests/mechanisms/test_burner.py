"""The burner: a hazard that takes a moment, so it is survivable at speed.

A trap that destroys on contact is a trap you either time perfectly or lose to.
The burner gives the player the moment instead: it destroys after a dwell, so
what kills a marble is being slow across it rather than being there at all.  The
crossing tests drive the real physics on a board leaning the way the game's
board leans, so the seconds they report are the seconds a player spends.
"""
import math

from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import levelfile, materials, mechanisms
from openglcontext_marble_demo.controller import ACTIVE, BURNED, DESTROYED
from openglcontext_marble_demo.game import BASE_TILT, GRAVITY, ROLL_DAMPING, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.burner import Burner, BurnerHeat

CELL = 4.0
RADIUS = 0.5
STEP = 1 / 60.0
#: Two cells of burner, and the X of their near and far edges.
BURN_CELLS = ((3, 0), (4, 0))
ENTRY, EXIT = 2.5 * CELL, 4.5 * CELL


def _level(features):
    cells = {(col, 0): 0.0 for col in range(-4, 14)}
    return Level(name='burner', cells=cells, start_cell=(-4, 0), finish_cell=(13, 0),
                 time_limit=600.0, features=[*features, Finish((13, 0))])


def _build(level):
    """Build ``level`` into a world leaning downhill along +X, as the game's does."""
    direction = (math.sin(BASE_TILT), -math.cos(BASE_TILT), 0.0)
    scene = DemoScene(gravity=model.Gravity(gravity=GRAVITY, direction=direction),
                      debug_flags=0,
                      default_linear_damping=ROLL_DAMPING[0],
                      default_angular_damping=ROLL_DAMPING[1])
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, index, level.build_into(scene, index)


def _marble(scene, index, x, velocity=(0.0, 0.0, 0.0), material='steel'):
    ball = scene.world.add_shape(model.Shape.sphere(RADIUS))
    return scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=(x, RADIUS + 0.001, 0.0))


def _heat_of(result):
    return next(a for a in result.animators if isinstance(a, BurnerHeat))


def _cross(speed, features=None, limit=20.0):
    """Send a marble across the burner at ``speed``.

    Returns what it was lost to (``None`` if it got out) and the seconds it
    spent standing in the fire.  The crossing is driven to the far edge whatever
    happens on the way, so the dwell is the whole time in the fire rather than
    the time up to the moment it caught.
    """
    scene, index, result = _build(_level(features or [Burner(cells=BURN_CELLS)]))
    heat = _heat_of(result)
    marble = _marble(scene, index, ENTRY - 2.0, (speed, 0.0, 0.0))
    dwell, cause = 0.0, None
    for _ in range(int(limit / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
        position = scene.world.position[marble]
        if heat.holds(position):
            dwell += STEP
        cause = cause or heat.lost(marble)
        if position[0] > EXIT + CELL:
            break
    return cause, dwell


# -- the whole point: quick is survivable, slow is not -------------------------

def test_crossing_the_burner_at_speed_survives_it():
    cause, dwell = _cross(12.0)
    burn_time = Burner.burn_time
    assert cause is None, (
        'across in %.2f s and still burned, against a %.2f s dwell' % (dwell, burn_time))
    assert dwell < burn_time, (
        'the quick crossing took %.2f s, at or over the %.2f s dwell: it is not '
        'testing what it claims to' % (dwell, burn_time))


def test_dawdling_in_the_burner_destroys_the_marble():
    cause, dwell = _cross(2.0)
    assert cause == BURNED, (
        '%.2f s in the fire and the marble came out (%s), against a %.2f s dwell'
        % (dwell, cause, Burner.burn_time))
    assert dwell >= Burner.burn_time


def test_the_dwell_is_what_separates_them_and_not_the_route():
    """Both crossings take the same line; only the speed differs."""
    quick, quick_dwell = _cross(12.0)
    slow, slow_dwell = _cross(2.0)
    assert (quick, slow) == (None, BURNED), (quick, slow)
    assert slow_dwell > quick_dwell * 2, (
        'the slow crossing dwelt %.2f s against the quick one\'s %.2f s: too close '
        'together to be about the speed' % (slow_dwell, quick_dwell))


# -- the heat itself -----------------------------------------------------------

def test_heat_builds_only_while_the_marble_is_in_the_fire():
    scene, index, result = _build(_level([Burner(cells=BURN_CELLS)]))
    heat = _heat_of(result)
    marble = _marble(scene, index, 3 * CELL)
    for _ in range(30):
        heat.update(STEP)
    inside = heat.heat[marble]
    assert inside > 0.0
    scene.world.place_body(marble, position=(0.0, RADIUS, 0.0))    # out on the stone
    for _ in range(30):
        heat.update(STEP)
    assert heat.heat.get(marble, 0.0) < inside, (
        'the marble left the fire and stayed at %.2f s of heat' % heat.heat.get(marble, 0.0))


def test_heat_cools_all_the_way_off_once_the_marble_is_clear():
    scene, index, result = _build(_level([Burner(cells=BURN_CELLS)]))
    heat = _heat_of(result)
    marble = _marble(scene, index, 3 * CELL)
    for _ in range(30):
        heat.update(STEP)
    scene.world.place_body(marble, position=(0.0, RADIUS, 0.0))
    for _ in range(240):
        heat.update(STEP)
    assert marble not in heat.heat, 'a cooled marble is still on the burner\'s books'


def test_the_edge_of_the_fire_is_a_line():
    scene, index, result = _build(_level([Burner(cells=BURN_CELLS)]))
    heat = _heat_of(result)
    assert heat.holds((3 * CELL, RADIUS, 0.0))
    assert heat.holds((ENTRY + 0.05, RADIUS, 0.0))
    assert not heat.holds((ENTRY - 0.05, RADIUS, 0.0))
    assert not heat.holds((EXIT + 0.05, RADIUS, 0.0))
    assert not heat.holds((3 * CELL, RADIUS, CELL))


def test_a_marble_carried_over_the_fire_is_clear_of_it():
    """The ramp route pays here too: an arc over the burner never heats up."""
    scene, index, result = _build(_level([Burner(cells=BURN_CELLS, catch_height=1.0)]))
    heat = _heat_of(result)
    assert heat.holds((3 * CELL, 0.9, 0.0))
    assert not heat.holds((3 * CELL, 1.1, 0.0))


# -- the run it belongs to -----------------------------------------------------

def _game():
    cells = {(col, 0): 0.0 for col in range(-4, 14)}
    level = Level(name='burner', cells=cells, start_cell=(-4, 0), finish_cell=(13, 0),
                  time_limit=600.0,
                  features=[Burner(cells=BURN_CELLS), Finish((13, 0))])
    return MarbleGame(level)


def _stand_in_the_fire(game, seconds):
    world, marble = game.scene.world, game.marble.index
    for _ in range(int(seconds / STEP)):
        world.place_body(marble, position=(3 * CELL, RADIUS + 0.001, 0.0))
        world.linear_velocity[marble] = (0.0, 0.0, 0.0)
        game.advance(STEP)


def test_the_game_loses_a_marble_left_in_the_fire():
    game = _game()
    _stand_in_the_fire(game, Burner.burn_time * 0.5)
    assert game.controller.state == ACTIVE, 'burned before the dwell was up'
    _stand_in_the_fire(game, Burner.burn_time)
    assert game.controller.state == DESTROYED
    assert game.controller.last_loss == BURNED


def test_a_restarted_run_puts_the_fire_out():
    game = _game()
    _stand_in_the_fire(game, Burner.burn_time * 0.5)
    heat = _heat_of(game.build)
    assert heat.heat, 'the marble never picked up any heat to forget'
    game.reset()
    assert not heat.heat


def test_the_burner_is_something_a_restarted_run_asks_to_forget():
    game = _game()
    assert any(isinstance(item, BurnerHeat) for item in game.build.resettable)


# -- the level it belongs to ---------------------------------------------------

def test_the_burner_owns_the_cells_it_covers():
    assert Burner(cells=BURN_CELLS).owned_cells() == set(BURN_CELLS)


def test_the_burner_lies_only_where_the_track_has_a_floor():
    scene, index, result = _build(_level([Burner(cells=(*BURN_CELLS, (40, 40)))]))
    assert set(_heat_of(result).ceilings) == set(BURN_CELLS)


def test_the_burner_is_registered_under_its_own_name():
    assert mechanisms.registry()['burner'] is Burner


def test_the_burner_round_trips_through_the_file_format():
    level = _level([Burner(cells=BURN_CELLS, burn_time=0.8, cool_rate=2.0,
                           catch_height=0.75)])
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features == level.features
