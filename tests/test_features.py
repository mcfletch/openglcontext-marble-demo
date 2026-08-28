"""Tests for level features: ramps (with speed boost / launch) and walls.

Features are the things that sit on the track and change how the marble moves.
Each is a small dataclass that builds its own bodies; a ramp additionally owns its
cell (it replaces the flat tile with a sloped one) and registers a boost effect
that the game fires when the marble rolls over it.

A ramp's *shape* is held to as tightly as its effect, because the shape is what
makes a run of stepped cells a surface: the face has to leave the cell behind it
at that cell's height and meet the cell ahead at its, or what a marble finds at
the join is a step.  :func:`_surface_along` asks the built world where the
surface is, by casting a ray down at it.
"""

import pytest
from omi_physics import model
from omi_physics.raycast import raycast
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


def _surface_along(level, cell, direction, offsets):
    """Where the surface is, ``offsets`` metres along ``direction`` from ``cell``.

    A ray straight down at each point; None where there is nothing under it.
    """
    scene, index, result = _built(level)
    x, z = level.cell_center(cell)
    heights = []
    for offset in offsets:
        origin = (x + direction[0] * offset, 20.0, z + direction[1] * offset)
        hit = raycast(scene.world, origin, (0.0, -1.0, 0.0), 40.0)
        heights.append(None if hit is None else float(hit.point[1]))
    return heights


# -- ramps ---------------------------------------------------------------

def test_ramp_owns_its_cell_so_no_flat_tile_is_built_there():
    ramp = Ramp(cell=(2, 0), direction=(0, 1))
    assert ramp.owned_cells() == {(2, 0)}


@pytest.mark.parametrize('rise', (0.9, -0.9))
def test_a_ramp_climbs_toward_the_way_it_faces(rise):
    """Its far edge is ``rise`` above its near one, and the face between them is
    the straight line joining the two.

    This is the whole of what a ramp is for.  A run of cells at stepped heights
    is a flight of steps, which a marble rolls down and cannot roll up; the ramp
    on each one turns the two heights into a surface.
    """
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), rise=rise),
                        extra_cells={(3, 0): rise})
    offsets = (-1.9, -1.0, 0.0, 1.0, 1.9)
    wanted = [rise * (offset + 2.0) / 4.0 for offset in offsets]
    assert _surface_along(level, (2, 0), (1, 0), offsets) \
        == pytest.approx(wanted, abs=0.02)


@pytest.mark.parametrize('rise', (0.9, -0.9))
def test_a_ramp_meets_the_cells_on_either_side_of_it(rise):
    """No step where the slope begins, and none where it ends.

    A lip at either join is a wall to a marble arriving at speed, which is what
    the cells being at different heights was supposed to stop being.
    """
    level = _level_with(Ramp(cell=(2, 0), direction=(1, 0), rise=rise),
                        extra_cells={(3, 0): rise})
    behind, near, far, ahead = _surface_along(
        level, (2, 0), (1, 0), (-2.4, -1.99, 1.99, 2.4))
    assert near == pytest.approx(behind, abs=0.02)
    assert far == pytest.approx(ahead, abs=0.02)


def test_a_ramp_can_be_a_slope_and_nothing_else():
    """``boost_speed=None`` is a ramp that only changes the shape of the floor."""
    ramp = Ramp(cell=(2, 0), direction=(1, 0), boost_speed=None)
    scene, index, result = _built(_level_with(ramp))
    assert result.effects == {}


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
