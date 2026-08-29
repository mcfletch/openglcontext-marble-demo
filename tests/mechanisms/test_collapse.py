"""The collapsing floor: a dwell, not an impact, and a trapdoor once it gives.

Every case runs on the real physics world, so what is measured is the marble
the mechanism actually produces.  The crossing tests drive a marble across a
strip carrying the floor and report the seconds it dwelt there, matching the
style :mod:`tests.mechanisms.test_burner` measures its own dwell-based trap in.
"""
import math

import numpy as np
import pytest
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import levelfile, materials, mechanisms
from openglcontext_marble_demo.game import BASE_TILT, GRAVITY, ROLL_DAMPING
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.collapse import Collapse, CollapseWatch, _Panel

CELL = 4.0
RADIUS = 0.5
STEP = 1 / 60.0
#: Two cells of floor, and the X of their near and far edges.
FLOOR_CELLS = ((3, 0), (4, 0))
ENTRY, EXIT = 2.5 * CELL, 4.5 * CELL


def _level(features):
    cells = {(col, 0): 0.0 for col in range(-4, 14)}
    return Level(name='collapse', cells=cells, start_cell=(-4, 0), finish_cell=(13, 0),
                time_limit=600.0, features=[*features, Finish((13, 0))])


def _build(level):
    """Build ``level`` with the game's own damping and a lean along +X, as the
    strip runs, so a crossing measures the same seconds a player would spend."""
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


def _watch_of(result):
    return next(a for a in result.animators if isinstance(a, CollapseWatch))


def _cross(speed, hold_time=0.9, limit=20.0):
    """Send a marble across the floor at ``speed``.

    Returns whether the floor gave way and the seconds the marble spent on it
    -- the crossing runs to the far edge whatever happens, so the dwell is the
    whole time on the floor rather than the time up to the moment it gave.
    """
    scene, index, result = _build(_level(
        [Collapse(cells=FLOOR_CELLS, direction=(1, 0), hold_time=hold_time)]))
    watch = _watch_of(result)
    marble = _marble(scene, index, ENTRY - 2.0, (speed, 0.0, 0.0))
    dwell = 0.0
    for _ in range(int(limit / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
        position = scene.world.position[marble]
        if watch.holds(position):
            dwell += STEP
        if position[0] > EXIT + 3 * CELL:
            break
    return watch.panel.released_at is not None, dwell


# -- the whole point: quick is survivable, dawdling is not ---------------------

def test_crossing_the_floor_at_speed_leaves_it_standing():
    gave_way, dwell = _cross(12.0)
    hold_time = Collapse.hold_time
    assert not gave_way, (
        'across in %.2f s and the floor still gave way, against a %.2f s hold'
        % (dwell, hold_time))
    assert dwell < hold_time, (
        'the quick crossing took %.2f s, at or over the %.2f s hold: it is not '
        'testing what it claims to' % (dwell, hold_time))


def test_dawdling_on_the_floor_drops_it_out_from_under():
    gave_way, dwell = _cross(2.0)
    assert gave_way, (
        '%.2f s on the floor and it never gave way, against a %.2f s hold'
        % (dwell, Collapse.hold_time))
    assert dwell >= Collapse.hold_time


def test_the_hold_is_what_separates_them_and_not_the_route():
    """Both crossings take the same line; only the speed differs."""
    quick_gave, quick_dwell = _cross(20.0)
    slow_gave, slow_dwell = _cross(2.0)
    assert (quick_gave, slow_gave) == (False, True), (quick_gave, slow_gave)
    assert slow_dwell > quick_dwell * 2, (
        'the slow crossing dwelt %.2f s against the quick one\'s %.2f s: too close '
        'together to be about the speed' % (slow_dwell, quick_dwell))


def test_two_quick_taps_never_add_up_like_one_long_stand():
    """Dwell drains once the weight is off, so tapping across twice is not the
    same as standing for the sum of the taps."""
    scene, index, result = _build(_level(
        [Collapse(cells=FLOOR_CELLS, direction=(1, 0), hold_time=0.9,
                 recover_rate=4.0)]))
    watch = _watch_of(result)
    marble = _marble(scene, index, ENTRY + 0.5)
    for _ in range(int(0.4 / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
    on_floor = watch.held.get(marble, 0.0)
    assert on_floor > 0.0, 'the marble never registered any weight on the floor'
    scene.world.place_body(marble, position=(-3 * CELL, RADIUS, 0.0))   # off the floor
    for _ in range(int(1.0 / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
    assert marble not in watch.held, 'the floor never forgot the earlier tap'


# -- the floor itself ------------------------------------------------------

def test_weight_builds_only_while_the_marble_is_on_the_floor():
    scene, index, result = _build(_level([Collapse(cells=FLOOR_CELLS, direction=(1, 0))]))
    watch = _watch_of(result)
    marble = _marble(scene, index, 3 * CELL)
    for _ in range(30):
        watch.update(STEP)
    on_floor = watch.held[marble]
    assert on_floor > 0.0
    scene.world.place_body(marble, position=(0.0, RADIUS, 0.0))   # off the floor
    for _ in range(30):
        watch.update(STEP)
    assert watch.held.get(marble, 0.0) < on_floor, (
        'the marble left the floor and its reading stayed at %.2f s'
        % watch.held.get(marble, 0.0))


def test_the_edge_of_the_floor_is_a_line():
    scene, index, result = _build(_level([Collapse(cells=FLOOR_CELLS, direction=(1, 0))]))
    watch = _watch_of(result)
    assert watch.holds((3 * CELL, RADIUS, 0.0))
    assert watch.holds((ENTRY + 0.05, RADIUS, 0.0))
    assert not watch.holds((ENTRY - 0.05, RADIUS, 0.0))
    assert not watch.holds((EXIT + 0.05, RADIUS, 0.0))


def test_a_body_arced_over_the_floor_is_clear_of_it():
    """The ramp route past a burner is a real route; the same is true here."""
    scene, index, result = _build(_level(
        [Collapse(cells=FLOOR_CELLS, direction=(1, 0), catch_height=1.0)]))
    watch = _watch_of(result)
    assert watch.holds((3 * CELL, 0.9, 0.0))
    assert not watch.holds((3 * CELL, 1.1, 0.0))


def test_the_floor_swings_clear_rather_than_vanishing():
    """A player sees the tile give way: it is a body that moves, not a hole
    that appears."""
    scene, index, result = _build(_level([Collapse(cells=FLOOR_CELLS, direction=(1, 0))]))
    watch = _watch_of(result)
    panel_index = next(a for a in result.animators
                       if isinstance(a, KinematicAnimator)).index
    shut = scene.world.orientation[panel_index].copy()
    marble = _marble(scene, index, ENTRY + 0.5)
    watch.panel.release(scene.world, marble)
    for _ in range(60):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
    assert not np.allclose(shut, scene.world.orientation[panel_index])


def test_the_floor_carries_a_slow_body_toward_the_way_down():
    """A marble that dawdled has, by definition, little speed of its own; the
    release itself is what puts it moving toward the far side."""
    scene, index, result = _build(_level(
        [Collapse(cells=FLOOR_CELLS, direction=(1, 0), carry_speed=6.0)]))
    watch = _watch_of(result)
    marble = _marble(scene, index, 3 * CELL, velocity=(0.05, 0.0, 0.0))
    watch.panel.release(scene.world, marble)
    scene.world.step(STEP)
    assert scene.world.linear_velocity[marble][0] >= 5.0, (
        'released at rest and still moving at %.2f m/s toward the far side'
        % scene.world.linear_velocity[marble][0])


def test_release_only_carries_a_body_once():
    """The floor gives way once; a second call is not a second shove."""
    scene, index, result = _build(_level(
        [Collapse(cells=FLOOR_CELLS, direction=(1, 0), carry_speed=6.0)]))
    watch = _watch_of(result)
    marble = _marble(scene, index, 3 * CELL)
    watch.panel.release(scene.world, marble)
    scene.world.step(STEP)
    once = float(scene.world.linear_velocity[marble][0])
    watch.panel.release(scene.world, marble)
    scene.world.step(STEP)
    assert float(scene.world.linear_velocity[marble][0]) < once + 1.0, (
        'a second release added speed on top of the first')


def test_the_angle_swings_from_shut_to_a_quarter_turn_and_stops():
    panel = _Panel(hinge=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), wdir=(0.0, 0.0, 1.0),
                  reach=2.0, duration=0.8)
    assert panel.angle == 0.0
    panel.release(_Waking(), 0)
    panel.pose(0.4)
    assert panel.angle == pytest.approx(math.pi / 4.0)
    panel.pose(0.8)
    assert panel.angle == pytest.approx(math.pi / 2.0)
    panel.pose(40.0)
    assert panel.angle == pytest.approx(math.pi / 2.0)   # and no further


def test_the_panel_hangs_from_its_hinge_once_fully_open():
    panel = _Panel(hinge=(0.0, -1.0, 0.0), axis=(1.0, 0.0, 0.0), wdir=(0.0, 0.0, 1.0),
                  reach=2.0, duration=0.5)
    shut, _ = panel.pose(0.0)
    assert shut == pytest.approx((0.0, -1.0, 2.0))       # flat, reaching out
    panel.release(_Waking(), 0)
    hanging, quat = panel.pose(0.5)
    assert hanging == pytest.approx((0.0, -3.0, 0.0))     # straight down at the hinge
    assert mathutil.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0) \
        == pytest.approx(np.asarray(quat))


class _Waking:
    """Enough of a world for :meth:`_Panel.release`: no velocity, and a body
    it can wake."""

    def __init__(self):
        self.linear_velocity = {0: np.zeros(3)}
        self.mass = {0: 1.0}

    def wake(self, body):
        self.woke = body


# -- reset, the registry and the file format --------------------------------

def test_a_restarted_run_shuts_the_floor_again():
    scene, index, result = _build(_level([Collapse(cells=FLOOR_CELLS, direction=(1, 0))]))
    watch = _watch_of(result)
    marble = _marble(scene, index, ENTRY + 0.5)
    watch.panel.release(scene.world, marble)
    assert watch.panel.released_at is not None
    watch.held[marble] = 0.5
    for holder in result.resettable:
        holder.reset(scene.world)
    assert watch.panel.released_at is None
    assert not watch.held


def test_the_floor_is_something_a_restarted_run_asks_to_forget():
    scene, index, result = _build(_level([Collapse(cells=FLOOR_CELLS, direction=(1, 0))]))
    assert any(isinstance(item, CollapseWatch) for item in result.resettable)
    assert any(isinstance(item, _Panel) for item in result.resettable)


def test_the_floor_owns_the_cells_it_covers():
    assert Collapse(cells=FLOOR_CELLS).owned_cells() == set(FLOOR_CELLS)


def test_the_floor_is_registered_under_its_own_name():
    assert mechanisms.registry()['collapse'] is Collapse


def test_the_floor_round_trips_through_the_file_format():
    level = _level([Collapse(cells=FLOOR_CELLS, direction=(1, 0), hold_time=0.8,
                             recover_rate=2.0, drop_time=0.4, catch_height=0.75,
                             carry_speed=7.0)])
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features == level.features
