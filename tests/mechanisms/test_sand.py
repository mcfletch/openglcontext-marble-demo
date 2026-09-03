"""The sand field: what it costs to roll through, and what it costs to fly over.

The crossing tests run the real physics on a board leaning the way the game's
board leans, so the times they report are the times a player waits.  The row of
cells runs along X and the lean is along X with it, which keeps the marble on one
row and the measurement to one number: the seconds between the near edge of the
sand and the far one.
"""
import math

import pytest
from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import fragments, levelfile, materials, mechanisms
from openglcontext_marble_demo.game import GRAVITY, ROLL_DAMPING, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.sand import Sand, SandDrag

CELL = 4.0
RADIUS = 0.5
STEP = 1 / 60.0
#: Two cells of sand, and the X of their near and far edges.
SAND_CELLS = ((3, 0), (4, 0))
ENTRY, EXIT = 2.5 * CELL, 4.5 * CELL


def _level(features):
    cells = {(col, 0): 0.0 for col in range(-2, 12)}
    return Level(name='sand', cells=cells, start_cell=(-2, 0), finish_cell=(11, 0),
                 time_limit=120.0, features=[*features, Finish((11, 0))])


def _build(level):
    """Build ``level`` into a world leaning downhill along +X.

    The lean is the one this mechanism's rule was measured against
    (:data:`~openglcontext_marble_demo.fragments.DESIGN_TILT`) rather than the
    game's own, which is level: what the rule says happens to a marble being
    carried through the piece, and a world that carries nothing says nothing.
    """
    direction = (math.sin(fragments.DESIGN_TILT), -math.cos(fragments.DESIGN_TILT), 0.0)
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


def _crossing(scene, result, marble, limit=40.0):
    """Seconds the marble takes to get from the near edge of the sand to the far one.

    ``None`` if it has not made it inside ``limit`` seconds of simulation.
    """
    entered = None
    elapsed = 0.0
    while elapsed < limit:
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
        elapsed += STEP
        x = scene.world.position[marble][0]
        if entered is None and x >= ENTRY:
            entered = elapsed
        if entered is not None and x >= EXIT:
            return elapsed - entered
    return None


def _run(scene, result, seconds):
    """Drive the world for ``seconds``, as the game drives it."""
    for _ in range(int(seconds / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)


def _drag_of(result):
    return next(a for a in result.animators if isinstance(a, SandDrag))


# -- what it costs -------------------------------------------------------

def test_sand_takes_real_time_to_churn_out_of():
    """Rolled into at speed, the sand is seconds rather than a moment."""
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS)]))
    through_sand = _crossing(scene, result, _marble(scene, index, -8.0, (8.0, 0, 0)))
    scene, index, result = _build(_level([]))
    over_stone = _crossing(scene, result, _marble(scene, index, -8.0, (8.0, 0, 0)))

    assert through_sand is not None, 'the marble never churned out of the far side'
    assert through_sand > 4 * over_stone, (
        'two cells of sand cost %.2f s against %.2f s of plain floor: not the '
        'delay a player is meant to be steering around' % (through_sand, over_stone))


def test_the_marble_always_churns_out_the_far_side():
    """Slowed to a crawl, not stopped: the board's lean still carries it through."""
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS)]))
    marble = _marble(scene, index, -8.0, (8.0, 0, 0))
    assert _crossing(scene, result, marble) is not None
    _run(scene, result, 2.0)                  # and on, out onto the stone beyond
    assert not _drag_of(result).caught, 'the sand kept its hold on a marble that left'
    assert scene.world.linear_velocity[marble][0] > 2.0, (
        'out of the sand at %.2f m/s: the marble has to pick up again once it is out'
        % scene.world.linear_velocity[marble][0])


def test_a_marble_launched_over_the_sand_crosses_as_if_it_were_stone():
    """The ramp route pays: an arc over the field is not touched by it."""
    launch = (10.0, 7.0, 0.0)
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS)]))
    over_sand = _crossing(scene, result, _marble(scene, index, ENTRY - 1.0, launch))
    scene, index, result = _build(_level([]))
    over_stone = _crossing(scene, result, _marble(scene, index, ENTRY - 1.0, launch))

    assert over_sand == pytest.approx(over_stone, abs=0.05), (
        'launched over the sand took %.2f s against %.2f s over plain floor: the '
        'ramp has to be a way past it' % (over_sand, over_stone))


# -- the edge ------------------------------------------------------------

def test_the_sand_edge_is_a_line_and_not_a_gradient():
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS)]))
    drag = _drag_of(result)
    assert drag.holds((3 * CELL, RADIUS, 0.0))            # the middle of it
    assert drag.holds((ENTRY + 0.05, RADIUS, 0.0))        # just over the near edge
    assert not drag.holds((ENTRY - 0.05, RADIUS, 0.0))    # just short of it
    assert not drag.holds((EXIT + 0.05, RADIUS, 0.0))     # just past the far edge
    assert not drag.holds((3 * CELL, RADIUS, CELL))       # one row over


def test_a_body_clear_of_the_surface_is_clear_of_the_sand():
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS, catch_height=1.0)]))
    drag = _drag_of(result)
    assert drag.holds((3 * CELL, 0.9, 0.0))
    assert not drag.holds((3 * CELL, 1.1, 0.0))


# -- what it does to a body ----------------------------------------------

def test_the_sand_gives_back_exactly_the_damping_it_took():
    scene, index, result = _build(_level([Sand(cells=SAND_CELLS)]))
    drag = _drag_of(result)
    world = scene.world
    marble = _marble(scene, index, 3 * CELL)
    before = (float(world.linear_damping[marble]), float(world.angular_damping[marble]))

    drag.update(STEP)
    assert world.linear_damping[marble] > before[0]
    assert world.angular_damping[marble] > before[1]

    world.place_body(marble, position=(0.0, RADIUS, 0.0))     # back out on the stone
    drag.update(STEP)
    assert (float(world.linear_damping[marble]),
            float(world.angular_damping[marble])) == before


def test_the_game_drags_a_marble_it_spawned_after_the_level():
    """The field is built before there is a marble, and still finds one."""
    game = MarbleGame(_level([Sand(cells=SAND_CELLS)]),
                      base_tilt=fragments.DESIGN_TILT)
    world, marble = game.scene.world, game.marble.index
    before = float(world.linear_damping[marble])
    world.place_body(marble, position=(3 * CELL, RADIUS + 0.001, 0.0))
    game.advance(STEP)
    assert world.linear_damping[marble] > before


# -- the level it belongs to ---------------------------------------------

def test_sand_owns_the_cells_it_covers():
    """The level builds no stone tile under the sand's own."""
    sand = Sand(cells=SAND_CELLS)
    assert sand.owned_cells() == set(SAND_CELLS)


def test_sand_lies_only_where_the_track_has_a_floor():
    scene, index, result = _build(_level([Sand(cells=(*SAND_CELLS, (40, 40)))]))
    assert set(_drag_of(result).ceilings) == set(SAND_CELLS)


def test_sand_is_registered_under_its_own_name():
    assert mechanisms.registry()['sand'] is Sand


def test_sand_round_trips_through_the_file_format():
    level = _level([Sand(cells=SAND_CELLS, linear_drag=3.5, angular_drag=4.5,
                         catch_height=0.75)])
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features == level.features
