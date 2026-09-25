"""The turntable mechanism: the bar's reach, the disc's spin, and the file format.

Every physical claim runs on the real physics world, exactly as
``tests/mechanisms/test_water.py`` and ``tests/mechanisms/test_rockfall.py`` do,
so what is measured is the marble the mechanism actually produces rather than
the bodies it created.
"""
import numpy as np
import pytest
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import levelfile, materials, mechanisms
from openglcontext_marble_demo.level import CELL_SIZE, Finish, Level
from openglcontext_marble_demo.mechanisms.turntable import Turntable, _Spin

FRAME = 1 / 120.0
HUB = (0, 0)


# -- fixtures ---------------------------------------------------------------

def _level(feature):
    cells = {(col, row): 0.0 for col in range(-6, 7) for row in range(-6, 7)}
    return Level(name='t', cells=cells, start_cell=(0, 0), finish_cell=(6, 0),
                 time_limit=60.0, features=[feature, Finish((6, 0))])


def _build(feature):
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, index, _level(feature).build_into(scene, index)


def _marble(scene, index, position, material='steel'):
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    body = scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)
    # Gravity would settle the marble onto the floor and jitter its speed on
    # its own; zeroing it isolates what the bar itself does to a marble sitting
    # beside it.
    scene.world.gravity_factor[body] = 0.0
    return body


def _spin_for_one_turn(scene, result, rpm, extra_frames=5):
    """Advance a little over one full revolution at ``rpm``."""
    frames = int((60.0 / abs(rpm)) / FRAME) + extra_frames
    top_speed = 0.0
    for _ in range(frames):
        for animator in result.animators:
            animator.update(FRAME)
        scene.world.step(FRAME)
        yield top_speed


def _top_speed(scene, result, marble, rpm):
    top = 0.0
    for _ in _spin_for_one_turn(scene, result, rpm):
        v = scene.world.linear_velocity[marble]
        top = max(top, float(np.hypot(v[0], v[2])))
    return top


# -- shape --------------------------------------------------------------------

def test_a_turntable_owns_no_cells():
    """The floor under it is ordinary tile; only the bar and the disc are its own."""
    assert Turntable(cell=HUB, radius=2).owned_cells() == set()


def test_the_bar_reaches_exactly_twice_its_radius():
    level = _level(Turntable(cell=HUB, radius=2))
    assert Turntable(cell=HUB, radius=2).span(level) == 2 * 2 * CELL_SIZE
    assert Turntable(cell=HUB, radius=3.5).span(level) == 2 * 3.5 * CELL_SIZE


# -- what it builds -------------------------------------------------------------

def test_the_disc_adds_no_body_of_its_own():
    """Drawn, not collided with -- see the module docstring on why."""
    cells = {(col, row): 0.0 for col in range(-6, 7) for row in range(-6, 7)}
    level = Level(name='t', cells=cells, start_cell=(0, 0), finish_cell=(6, 0),
                 time_limit=60.0, features=[Turntable(cell=HUB, radius=2)])
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    result = level.build_into(scene, index)
    assert len(result.feature_bodies) == 1, \
        '%d bodies for one bar and one (undriven) disc' % len(result.feature_bodies)


def test_the_disc_turns_at_the_bars_own_rate():
    scene, index, result = _build(Turntable(cell=HUB, radius=2, rpm=30.0))
    spins = [a for a in result.animators if isinstance(a, _Spin)]
    assert len(spins) == 1
    arms = [a for a in result.animators if isinstance(a, KinematicAnimator)]
    assert len(arms) == 1
    before = spins[0].transform.rotation
    for _ in range(30):
        for animator in result.animators:
            animator.update(FRAME)
        scene.world.step(FRAME)
    assert not np.allclose(spins[0].transform.rotation, before), \
        'the disc has not moved after half a second at 30 rpm'


def test_a_negative_rpm_turns_the_bar_the_other_way():
    reference = np.array([1.0, 0.0, 0.0])

    def _heading(rpm):
        scene, index, result = _build(Turntable(cell=HUB, radius=2, rpm=rpm))
        arm = next(a for a in result.animators if isinstance(a, KinematicAnimator))
        for _ in range(6):
            for animator in result.animators:
                animator.update(FRAME)
            scene.world.step(FRAME)
        return mathutil.quat_rotate(scene.world.orientation[arm.index], reference)

    forward = _heading(20.0)
    backward = _heading(-20.0)
    assert not np.allclose(forward, backward), \
        'the bar ended up at %r whichever way it was told to turn' % (forward,)
    assert forward[2] == pytest.approx(-backward[2], abs=1e-3), \
        'forward=%r backward=%r are not mirror images after the same time' \
        % (forward, backward)


# -- what it does to a marble ---------------------------------------------------

def test_a_marble_well_clear_of_the_hub_is_never_touched():
    """The reach a fragment plans a lead-in lane around is a real limit.

    Three cells out is a cell and a half past the tip of a radius-2 bar; a
    marble sitting there for a full turn should feel nothing.
    """
    scene, index, result = _build(Turntable(cell=HUB, radius=2, rpm=20.0))
    clear = _marble(scene, index, (12.0, 0.5, 0.0))
    top = _top_speed(scene, result, clear, 20.0)
    assert top == 0.0, \
        'a marble 12 m from a radius-2 (8 m half-length) bar reached %.3f m/s' % top


def test_a_marble_within_reach_is_carried():
    """The positive control: something the bar can reach, it does reach."""
    scene, index, result = _build(Turntable(cell=HUB, radius=2, rpm=20.0))
    reachable = _marble(scene, index, (6.0, 0.5, 0.0))
    top = _top_speed(scene, result, reachable, 20.0)
    assert top > 1.0, \
        'a marble 6 m from a radius-2 (8 m half-length) bar only reached %.3f m/s' % top


# -- registration and the file format --------------------------------------------

def test_a_turntable_is_a_mechanism_the_registry_knows():
    assert mechanisms.registry().get('turntable') is Turntable
    assert levelfile.FEATURES.get('turntable') is Turntable


def test_a_turntable_round_trips_through_the_file_format():
    disc = Turntable(cell=(2, 1), radius=1.5, rpm=-12.5, bar_thickness=0.4,
                     clearance=0.25, disc_thickness=0.1)
    level = _level(disc)
    again = levelfile.from_json(levelfile.to_json(level))
    assert again.features[0] == disc


def test_a_turntable_is_saved_under_its_own_name():
    document = levelfile.to_json(_level(Turntable(cell=HUB, radius=2)))
    assert document['features'][0]['kind'] == 'turntable'
