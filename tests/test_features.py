"""Tests for level features: ramps (with speed boost / launch) and walls.

Features are the things that sit on the track and change how the marble moves.
Each is a small dataclass that builds its own bodies; a ramp additionally owns its
cell (it replaces the flat tile with a sloped one) and registers a boost effect
that the game fires when the marble rolls over it.
"""

from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import materials
from openglcontext_marble_demo.level import Finish, Level, Ramp, Wall


def _level_with(feature, extra_cells=None):
    cells = {(c, 0): 0.0 for c in range(0, 5)}
    if extra_cells:
        cells.update(extra_cells)
    return Level(name="t", cells=cells, start_cell=(0, 0), finish_cell=(4, 0),
                 time_limit=30.0, features=[feature, Finish((4, 0))])


def _built(level):
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    return scene, index, level.build_into(scene, index)


# -- ramps ---------------------------------------------------------------

def test_ramp_owns_its_cell_so_no_flat_tile_is_built_there():
    ramp = Ramp(cell=(2, 0), direction=(0, 1))
    assert ramp.owned_cells() == {(2, 0)}


def test_ramp_registers_a_boost_effect():
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), boost_speed=8.0))
    scene, index, result = _built(level)
    assert len(result.effects) == 1


def test_ramp_boost_accelerates_a_slow_marble_along_its_direction():
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), boost_speed=8.0))
    scene, index, result = _built(level)
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    marble = scene.world.add_body(model.Motion(type=model.DYNAMIC, mass=2.0,
                                               linearVelocity=(1.0, 0, 0)),
                                  collider=model.Collider(shape=ball), position=(8, 0.6, 0))
    effect = next(iter(result.effects.values()))
    effect(scene.world, marble)
    assert scene.world.linear_velocity[marble][0] >= 7.5      # boosted toward +X


def test_ramp_boost_does_not_slow_an_already_fast_marble():
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), boost_speed=6.0))
    scene, index, result = _built(level)
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    marble = scene.world.add_body(model.Motion(type=model.DYNAMIC, mass=2.0,
                                               linearVelocity=(10.0, 0, 0)),
                                  collider=model.Collider(shape=ball), position=(8, 0.6, 0))
    effect = next(iter(result.effects.values()))
    effect(scene.world, marble)
    assert scene.world.linear_velocity[marble][0] >= 10.0     # unchanged (cap, not brake)


def test_launch_ramp_adds_upward_impulse():
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), boost_speed=8.0, launch=True))
    scene, index, result = _built(level)
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    marble = scene.world.add_body(model.Motion(type=model.DYNAMIC, mass=2.0,
                                               linearVelocity=(6.0, 0, 0)),
                                  collider=model.Collider(shape=ball), position=(8, 0.6, 0))
    effect = next(iter(result.effects.values()))
    effect(scene.world, marble)
    assert scene.world.linear_velocity[marble][1] > 1.0       # popped upward


# -- walls ---------------------------------------------------------------

def test_wall_builds_a_static_collider_body():
    level = _level_with(Wall(cell=(2, 0), side="E"))
    scene, index, result = _built(level)
    # The wall is a static body with a collider (not a trigger).
    walls = [k for k in range(scene.world.body_count)
             if scene.world.motion_type[k] == 0 and scene.world.collider_shape[k] >= 0]
    assert len(walls) >= 1


def test_wall_does_not_own_a_cell():
    assert Wall(cell=(2, 0), side="E").owned_cells() == set()
