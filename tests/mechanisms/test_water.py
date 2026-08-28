"""The water trap: a pool that takes the marble in and lets it out underneath.

Every case runs on the real physics world, so what is measured is the marble the
mechanism actually produces rather than the bodies it created.  The trigger
listener the game wires is reproduced in :func:`_wire`, because the effects are
only reached through it and a mechanism tested with its effects called by hand
would not show that they are reached at all.
"""
import math

import numpy as np
import pytest
from omi_physics import model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import levelfile, materials
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.water import Water, _Plug

FRAME = 1 / 60.0
#: The pool used throughout: cell ``(3, 0)`` of the strip below, so its centre is
#: at ``x = 12``, ``z = 0`` and its surface is at ``y = 0``.
POOL_CELL = (3, 0)
POOL_X, POOL_Z = 12.0, 0.0
DEPTH = 3.0
#: Well away from the strip of tiles, where nothing is under the marble.
OPEN_AIR_Z = 40.0


# -- fixtures -------------------------------------------------------------------

def _level(feature):
    cells = {(col, 0): 0.0 for col in range(-3, 6)}
    return Level(name='t', cells=cells, start_cell=(0, 0), finish_cell=(5, 0),
                 time_limit=60.0, features=[feature, Finish((5, 0))])


def _build(feature):
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, index, _level(feature).build_into(scene, index)


def _marble(scene, index, position, material='steel'):
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    return scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)


def _wire(world, result, marble):
    """The trigger listener :meth:`MarbleGame._wire_triggers` wires, for one marble.

    Effects fire on entry and on every frame of overlap; ``exit`` is ignored, as
    the game ignores it.
    """
    def on_trigger(event_type, trigger_body, other_body):
        if other_body != marble or event_type == 'exit':
            return
        effect = result.effects.get(trigger_body)
        if effect is not None:
            effect(world, marble)

    world.add_trigger_listener(on_trigger)


def _step(scene, result, frames):
    """Advance ``frames`` frames, driving the mechanisms as the game does."""
    for _ in range(frames):
        for animator in result.animators:
            animator.update(FRAME)
        scene.world.step(FRAME)


def _fall_time(scene, result, marble, to_y, frames=900):
    """Seconds until the marble's centre first reaches ``to_y``."""
    for frame in range(frames):
        for animator in result.animators:
            animator.update(FRAME)
        scene.world.step(FRAME)
        if scene.world.position[marble][1] <= to_y:
            return (frame + 1) * FRAME
    return None


# -- sinking --------------------------------------------------------------------

def test_the_water_takes_the_marble_down_more_slowly_than_air():
    """Two metres of the same fall, one through the pool and one beside the board."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    sinking = _marble(scene, index, position=(POOL_X, 0.0, POOL_Z))
    _wire(scene.world, result, sinking)
    wet = _fall_time(scene, result, sinking, to_y=-2.0)

    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    falling = _marble(scene, index, position=(POOL_X, 0.0, OPEN_AIR_Z))
    _wire(scene.world, result, falling)
    dry = _fall_time(scene, result, falling, to_y=-2.0)

    assert wet is not None and dry is not None
    assert wet > 2.0 * dry, 'two metres down: %.2fs in the water, %.2fs in air' % (wet, dry)


def test_the_pool_only_slows_what_is_in_it():
    """A marble falling beside the board keeps the gravity and damping it had."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    marble = _marble(scene, index, position=(POOL_X, 0.0, OPEN_AIR_Z))
    _wire(scene.world, result, marble)
    _step(scene, result, 60)
    assert scene.world.gravity_factor[marble] == 1.0
    assert scene.world.linear_damping[marble] == 0.0


# -- the plug -------------------------------------------------------------------

def test_the_plug_is_a_floor_until_something_reaches_it():
    """Nothing about the pool says it will open: a marble put on the plug rests there."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    marble = _marble(scene, index, position=(POOL_X, -2.4, POOL_Z))
    _step(scene, result, 120)
    assert scene.world.position[marble][1] > -DEPTH


def test_the_plug_gives_when_the_marble_reaches_the_bottom():
    """Dropped in at the surface, the marble leaves through the floor of the pool."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    marble = _marble(scene, index, position=(POOL_X, 0.0, POOL_Z))
    _wire(scene.world, result, marble)
    _step(scene, result, 600)
    assert scene.world.position[marble][1] < -DEPTH - 1.0, \
        'the marble is at y=%.2f, and the pool floor is at y=%.2f' % (
            scene.world.position[marble][1], -DEPTH)


def test_the_plug_swings_clear_rather_than_vanishing():
    """The floor of the pool is a body that moves, so a player sees it give."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    marble = _marble(scene, index, position=(POOL_X, 0.0, POOL_Z))
    _wire(scene.world, result, marble)
    plug = next(a for a in result.animators if isinstance(a, KinematicAnimator))
    shut = scene.world.orientation[plug.index].copy()
    _step(scene, result, 600)
    assert not np.allclose(shut, scene.world.orientation[plug.index])


def test_the_flap_turns_about_its_hinge_and_stops_at_a_quarter_turn():
    """The swing on its own, without a world: shut, halfway, hanging, and still."""
    plug = _Plug(hinge=(0.0, -3.0, 0.0), reach=2.0, duration=0.8)
    assert plug.pose(0.0)[0] == (2.0, -3.0, 0.0)      # flat, reaching out from the hinge
    plug.release(_Waking(), 0)
    plug.pose(0.4)                                     # halfway through the swing
    assert plug.angle == pytest.approx(math.pi / 4.0)
    position, _ = plug.pose(0.8)
    assert position == pytest.approx((0.0, -5.0, 0.0))  # hanging straight down
    plug.pose(40.0)
    assert plug.angle == pytest.approx(math.pi / 2.0)   # and no further


class _Waking:
    """Enough of a world for :meth:`_Plug.release`, which wakes what reached it."""

    def wake(self, body):
        self.woke = body


# -- coming out the other side --------------------------------------------------

def test_the_marble_is_itself_again_once_it_is_through():
    """What the water lends it, it takes back: the rest of the run plays normally."""
    scene, index, result = _build(Water(cell=POOL_CELL, depth=DEPTH))
    marble = _marble(scene, index, position=(POOL_X, 0.0, POOL_Z))
    _wire(scene.world, result, marble)
    before = (float(scene.world.gravity_factor[marble]),
              float(scene.world.linear_damping[marble]))

    was_slowed = False
    for _ in range(600):
        for animator in result.animators:
            animator.update(FRAME)
        scene.world.step(FRAME)
        if scene.world.gravity_factor[marble] < before[0]:
            was_slowed = True

    assert was_slowed, 'the marble never sank, so there is nothing to have restored'
    after = (float(scene.world.gravity_factor[marble]),
             float(scene.world.linear_damping[marble]))
    assert after == before, 'left the pool with %r, went in with %r' % (after, before)


# -- the file format ------------------------------------------------------------

def test_a_pool_round_trips_through_the_file_format():
    water = Water(cell=(2, 1), depth=4.5, sink_gravity=0.3, sink_damping=2.0,
                  open_time=1.25, rim=0.4, plug_thickness=0.2)
    level = _level(water)
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features[0] == water


def test_the_pool_is_a_mechanism_the_registry_knows():
    from openglcontext_marble_demo import mechanisms
    assert mechanisms.registry().get('water') is Water
    assert levelfile.FEATURES.get('water') is Water
