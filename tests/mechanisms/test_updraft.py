"""The updraft field: what it does to a body inside its box, and never does.

The crossing behaviour that makes the shaft worth building into a piece is
measured in ``tests/fragments/test_updraft.py``, against the real board lean.
What belongs here is the field on its own: the hard edges of the box it acts
in, and the one property the whole mechanism depends on -- that it is always
weaker than gravity, so nothing held inside it ever stops coming down.
"""
import dataclasses
import math

from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import levelfile, materials, mechanisms
from openglcontext_marble_demo.game import BASE_TILT, GRAVITY, ROLL_DAMPING
from openglcontext_marble_demo.level import Level
from openglcontext_marble_demo.mechanisms.updraft import Updraft, UpdraftLift

CELL = 4.0
RADIUS = 0.5
STEP = 1 / 60.0

#: One cell's worth of shaft, so a test can place a body anywhere in or around
#: it without also having to reason about a second cell's box.
SHAFT_CELLS = ((3, 0),)
FLOOR = -6.0
REACH = 7.5
STRENGTH = 9.2


def _level(features):
    # The shaft needs no floor of its own -- the one cell of real track here is
    # only somewhere for a marble to be dropped from, unrelated to the shaft.
    cells = {(-2, 0): 0.0}
    return Level(name='updraft', cells=cells, start_cell=(-2, 0), finish_cell=(-2, 0),
                 time_limit=120.0, features=list(features))


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


def _marble(scene, index, position, velocity=(0.0, 0.0, 0.0), material='steel'):
    ball = scene.world.add_shape(model.Shape.sphere(RADIUS))
    return scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)


def _lift_of(result):
    return next(a for a in result.animators if isinstance(a, UpdraftLift))


def _run(scene, result, seconds, dt=STEP):
    for _ in range(int(seconds / dt)):
        for animator in result.animators:
            animator.update(dt)
        scene.world.step(dt)


def _shaft_point(offset=0.0):
    """A world point over the middle of :data:`SHAFT_CELLS`, ``offset`` up from
    the floor of the shaft."""
    return (3 * CELL, FLOOR + offset, 0.0)


# -- the box --------------------------------------------------------------

def test_the_shaft_is_a_box_with_hard_edges():
    """Sand's own edge is a line and not a gradient; the shaft's is the same."""
    scene, index, result = _build(_level([Updraft(cells=SHAFT_CELLS, strength=STRENGTH,
                                                  floor=FLOOR, reach=REACH)]))
    lift = _lift_of(result)
    assert lift.holds(_shaft_point(1.0))               # well inside
    assert lift.holds(_shaft_point(0.0))                # right at the floor
    assert lift.holds(_shaft_point(REACH))              # right at the top
    assert not lift.holds(_shaft_point(-0.05))          # just under the floor
    assert not lift.holds(_shaft_point(REACH + 0.05))   # just over the reach
    assert not lift.holds((3 * CELL, FLOOR + 1.0, CELL))    # one row over
    assert not lift.holds((4 * CELL, FLOOR + 1.0, 0.0))     # one column over


def test_updraft_owns_no_cells():
    """The shaft's cells carry no floor of their own -- there is nothing to own."""
    assert Updraft(cells=SHAFT_CELLS).owned_cells() == set()


# -- the one thing the whole mechanism depends on --------------------------

def test_a_stationary_marble_still_sinks():
    """Left in the shaft with nothing carrying it forward, a marble comes down.

    This is the fact the piece's whole rule rests on: the lift is weaker than
    gravity, so hovering is never an option, only a slower fall.
    """
    scene, index, result = _build(_level([Updraft(cells=SHAFT_CELLS, strength=STRENGTH,
                                                  floor=FLOOR, reach=REACH)]))
    marble = _marble(scene, index, _shaft_point(REACH - RADIUS - 0.1))
    start_y = float(scene.world.position[marble][1])
    _run(scene, result, 4.0)
    end_y = float(scene.world.position[marble][1])
    velocity_y = float(scene.world.linear_velocity[marble][1])
    assert end_y < start_y - 1.0, (
        'a marble left in the shaft for 4 s only sank from %.2f to %.2f m: '
        'the draft is holding it up rather than merely slowing it' % (start_y, end_y))
    assert velocity_y < -0.5, (
        'after 4 s in the shaft the marble is still falling at %.2f m/s, which '
        'is not the same as having stopped -- but it must still be coming down'
        % velocity_y)


def test_the_lift_slows_the_sink_without_stopping_it():
    """Compared to open air, the same marble falls less far in the same time --
    the draft costs the pull some of its edge without cancelling it."""
    scene, index, result = _build(_level([Updraft(cells=SHAFT_CELLS, strength=STRENGTH,
                                                  floor=FLOOR, reach=REACH)]))
    lifted = _marble(scene, index, _shaft_point(REACH - RADIUS - 0.1))
    open_air = _marble(scene, index, (30 * CELL, FLOOR + REACH - RADIUS - 0.1, 0.0))
    start_y = float(scene.world.position[lifted][1])
    _run(scene, result, 1.5)
    fell_lifted = start_y - float(scene.world.position[lifted][1])
    fell_open = start_y - float(scene.world.position[open_air][1])
    assert 0.0 < fell_lifted < fell_open, (
        'in 1.5 s a marble in the shaft fell %.2f m against %.2f m in open air: '
        'the shaft has to fall between "as fast as open air" and "not at all"'
        % (fell_lifted, fell_open))


def test_a_body_moving_sideways_through_the_shaft_still_feels_the_lift():
    """The box is horizontal as well as vertical: passing through it counts,
    not only being dropped straight down the middle of it."""
    scene, index, result = _build(_level([Updraft(cells=SHAFT_CELLS, strength=STRENGTH,
                                                  floor=FLOOR, reach=REACH)]))
    start_x = 3 * CELL - CELL / 2.0        # the shaft's own near edge
    start_y = FLOOR + REACH - RADIUS - 0.1
    marble = _marble(scene, index, (start_x, start_y, 0.0), velocity=(8.0, 0.0, 0.0))
    # One row clear of the shaft's footprint, never inside its box at all.
    without = _marble(scene, index, (start_x, start_y, 3 * CELL), velocity=(8.0, 0.0, 0.0))
    _run(scene, result, 0.4)               # 3.2 m at 8 m/s -- still inside the 4 m shaft
    fell_through = start_y - float(scene.world.position[marble][1])
    fell_beside = start_y - float(scene.world.position[without][1])
    assert fell_through < fell_beside, (
        'crossing the shaft sideways fell %.2f m against %.2f m one row clear of '
        'it: the lift should reach a body passing through, not only one dropped '
        'straight down its centre' % (fell_through, fell_beside))


# -- registration and the file format ---------------------------------------

def test_updraft_is_registered_under_its_own_name():
    assert mechanisms.registry()['updraft'] is Updraft


def test_updraft_round_trips_through_the_file_format():
    level = _level([Updraft(cells=SHAFT_CELLS, strength=8.4, floor=-5.5, reach=6.5)])
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features == level.features


def test_updraft_cells_survive_a_list_round_trip():
    """A file gives back lists where a level gave tuples; the mechanism's own
    ``__post_init__`` is what makes a loaded board the same board."""
    made = Updraft(cells=[[3, 0], [3, 1]])
    assert made.cells == ((3, 0), (3, 1))
    assert dataclasses.replace(made, strength=1.0).cells == made.cells
