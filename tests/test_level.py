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


# -- the kill plane ----------------------------------------------------------

def test_the_kill_plane_goes_under_the_deepest_tile():
    """A board that descends far enough would otherwise run out through it.

    The plane was a constant -8 metres, and a board descending 0.9 m a cell
    passes that after nine cells of slope.  Measured over twelve generated
    boards, six had floor below it and one had 134 of its 192 cells there —
    every one a tile a player can see, stand on, and be killed for standing on,
    with no way to tell why.
    """
    from openglcontext_marble_demo.level import KILL_MARGIN, Level
    deep = {(col, row): -0.9 * row for col in range(-1, 2) for row in range(20)}
    level = Level(name='deep', cells=deep, start_cell=(0, 0), finish_cell=(0, 19),
                  time_limit=90.0)
    floor = min(deep.values())
    assert level.kill_y <= floor - KILL_MARGIN, \
        'the board reaches %.2f and the kill plane sits at %.2f' % (floor, level.kill_y)
    assert not [cell for cell, height in level.cells.items()
                if height <= level.kill_y], 'board below the kill plane'


def test_a_shallow_board_keeps_the_plane_where_it_was():
    """Under the board, not glued to it: a marble that goes over the edge of a
    flat board still has somewhere to fall before it is caught."""
    from openglcontext_marble_demo.level import Level
    flat = {(col, row): 0.0 for col in range(-1, 2) for row in range(4)}
    level = Level(name='flat', cells=flat, start_cell=(0, 0), finish_cell=(0, 3),
                  time_limit=90.0)
    assert level.kill_y == -8.0


def test_a_board_may_ask_for_a_lower_plane_than_its_own_depth():
    from openglcontext_marble_demo.level import Level
    flat = {(col, row): 0.0 for col in range(-1, 2) for row in range(4)}
    level = Level(name='flat', cells=flat, start_cell=(0, 0), finish_cell=(0, 3),
                  time_limit=90.0, kill_y=-40.0)
    assert level.kill_y == -40.0


def test_a_marble_resting_on_the_deepest_tile_has_not_fallen():
    """Which is what the plane being above it meant: the board you were standing
    on was outside the world."""
    from openglcontext_marble_demo.game import MarbleGame
    deep = {(col, row): -0.9 * row for col in range(-1, 2) for row in range(20)}
    level = Level(name='deep', cells=deep, start_cell=(0, 0), finish_cell=(0, 19),
                  time_limit=90.0)
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    world.place_body(index, (0.0, min(deep.values()) + 0.55, 19 * level.cell_size))
    world.wake(index)
    for _ in range(240):
        game.advance(1 / 120.0)
    assert game.controller.fall_count == 0, \
        'a marble sitting on the lowest tile was counted as having fallen'
