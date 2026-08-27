"""Tests for :class:`Level` and building it into a scene.

A ``Level`` is pure data (cells + features + metadata); ``build_into`` realizes it
as physics bodies and render nodes in a :class:`DemoScene` and returns the
:class:`TrackMap` the controller uses.  Tests build headlessly and assert on the
resulting world, so no window is needed.
"""
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import materials
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.track import TrackMap


def _simple_level():
    cells = {(0, 0): 0.0, (1, 0): -0.3, (2, 0): -0.6}
    return Level(name="t", cells=cells, start_cell=(0, 0), finish_cell=(2, 0),
                 time_limit=30.0, features=[Finish((2, 0))])


def test_track_map_matches_cells():
    level = _simple_level()
    track = level.track_map()
    assert isinstance(track, TrackMap)
    assert track.cells == level.cells


def test_marble_start_sits_above_the_start_cell_surface():
    level = _simple_level()
    x, y, z = level.marble_start(radius=0.5)
    assert (x, z) == (0.0, 0.0)               # start cell centre
    assert y > 0.0                             # resting above surface (0.0) + radius


def test_build_adds_a_floor_body_per_cell():
    level = _simple_level()
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    before = scene.world.body_count
    level.build_into(scene, index)
    # One static floor per cell (finish trigger is extra).
    assert scene.world.body_count >= before + len(level.cells)


def test_build_registers_a_finish_trigger():
    level = _simple_level()
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    result = level.build_into(scene, index)
    assert result.finish_body is not None
    # The finish trigger body carries a trigger shape, not a collider.
    assert scene.world.trigger_shape[result.finish_body.index] >= 0


def test_build_returns_track_map_for_the_controller():
    level = _simple_level()
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    result = level.build_into(scene, index)
    assert result.track.cells == level.cells
