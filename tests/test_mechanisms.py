"""Tests for the moving/reactive mechanisms: bumper, spring trap, elevator, arm.

Bumpers and spring traps act through contact/impulse; elevators and rotating arms
are kinematic bodies driven by :class:`KinematicAnimator`.  All are exercised on the
real physics world so the tests prove the mechanism actually moves the marble (or
itself), not merely that a body was created.
"""
import numpy as np
from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import materials
from openglcontext_marble_demo.level import Bumper, Elevator, Finish, Level, RotatingArm, SpringTrap


def _level(feature):
    cells = {(c, 0): 0.0 for c in range(-3, 6)}
    return Level(name="t", cells=cells, start_cell=(0, 0), finish_cell=(5, 0),
                 time_limit=60.0, features=[feature, Finish((5, 0))])


def _build(level):
    scene = DemoScene(debug_flags=0)
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, index, level.build_into(scene, index)


def _marble(scene, index, material="rubber", position=(0, 0.6, 0), velocity=(0, 0, 0)):
    ball = scene.world.add_shape(model.Shape.sphere(0.5))
    return scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)


# -- bumper --------------------------------------------------------------

def test_bumper_bounces_the_marble_back():
    scene, index, result = _build(_level(Bumper(cell=(3, 0))))
    # Start right in front of the bumper (at cell (3,0) ≈ x=12) so the rebound is
    # what's measured, not the friction of a long roll-in.
    marble = _marble(scene, index, velocity=(8.0, 0, 0), position=(10.5, 0.6, 0))
    rebound = 0.0
    for _ in range(120):
        scene.world.step(1 / 60.0)
        rebound = min(rebound, scene.world.linear_velocity[marble][0])
    assert rebound < -2.0      # bounced back with real energy (peak rebound speed)


# -- spring trap ---------------------------------------------------------

def test_spring_trap_launches_the_marble_upward():
    scene, index, result = _build(_level(SpringTrap(cell=(3, 0), impulse=(0, 9, 0))))
    marble = _marble(scene, index, position=(12.0, 0.6, 0))
    effect = next(iter(result.effects.values()))
    effect(scene.world, marble)
    assert scene.world.linear_velocity[marble][1] > 5.0


def test_spring_trap_rearms_only_after_its_cooldown():
    scene, index, result = _build(_level(SpringTrap(cell=(3, 0), impulse=(0, 9, 0),
                                                    rearm=1.0)))
    marble = _marble(scene, index, position=(12.0, 0.6, 0))
    effect = next(iter(result.effects.values()))

    scene.world.time = 10.0
    effect(scene.world, marble)
    first = scene.world.linear_velocity[marble][1]
    scene.world.linear_velocity[marble] = (0, 0, 0)

    effect(scene.world, marble)                # same instant → still on cooldown
    assert scene.world.linear_velocity[marble][1] == 0.0

    scene.world.time = 11.5                     # past the 1.0s rearm
    effect(scene.world, marble)
    assert scene.world.linear_velocity[marble][1] > 5.0
    assert first > 5.0


# -- elevator ------------------------------------------------------------

def test_elevator_registers_an_animator_and_moves_vertically():
    scene, index, result = _build(_level(Elevator(cell=(3, 0), travel=3.0, period=2.0)))
    assert len(result.animators) == 1
    anim = result.animators[0]
    heights = []
    for _ in range(180):
        anim.update(1 / 60.0)
        scene.world.step(1 / 60.0)
        heights.append(scene.world.position[anim.index][1])
    assert max(heights) - min(heights) > 1.5      # it travels up and down


def test_elevator_carries_a_marble_upward():
    scene, index, result = _build(_level(Elevator(cell=(3, 0), travel=3.0, period=4.0)))
    anim = result.animators[0]
    plat_xz = (12.0, 0.0)
    marble = _marble(scene, index, position=(plat_xz[0], 0.8, plat_xz[1]))
    for _ in range(90):                           # first quarter of the rise
        anim.update(1 / 60.0)
        scene.world.step(1 / 60.0)
    assert scene.world.position[marble][1] > 0.9   # lifted off the start height


# -- rotating arm --------------------------------------------------------

def test_rotating_arm_spins():
    scene, index, result = _build(_level(RotatingArm(cell=(3, 0), rpm=30)))
    assert len(result.animators) == 1
    anim = result.animators[0]
    before = scene.world.orientation[anim.index].copy()
    for _ in range(30):
        anim.update(1 / 60.0)
        scene.world.step(1 / 60.0)
    after = scene.world.orientation[anim.index]
    assert not np.allclose(before, after)          # it rotated


def test_rotating_arm_knocks_a_marble():
    scene, index, result = _build(_level(RotatingArm(cell=(3, 0), rpm=90, length=3.5)))
    anim = result.animators[0]
    # Marble sitting just off the pivot, within the arm's sweep.
    marble = _marble(scene, index, position=(12.0, 0.6, 1.2))
    for _ in range(120):
        anim.update(1 / 60.0)
        scene.world.step(1 / 60.0)
    assert np.linalg.norm(scene.world.linear_velocity[marble][[0, 2]]) > 0.5
