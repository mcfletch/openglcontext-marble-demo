"""Tests for the wall-hit and hard-landing speed-kill rules.

Marble Madness punishes crashing: slam a wall or drop from height onto a hard
floor and you lose nearly all your speed, while a springy (rubber) surface gives it
back.  The controller reads the solver's contact impulses each frame to spot a hard
impact and, on a non-elastic surface, bleeds the horizontal speed — leaving elastic
surfaces to the physics (restitution already returns their energy).
"""

from omi_physics import model
from omi_physics.world import PhysicsWorld

from openglcontext_marble_demo import materials
from openglcontext_marble_demo.controller import MarbleController
from openglcontext_marble_demo.track import TrackMap


def _world(surface="stone"):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    index = materials.register_materials(world)
    materials.apply_pair_frictions(world, index)
    ground = world.add_shape(model.Shape.box((60, 1, 60)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground, physicsMaterial=index["stone"]),
                   position=(0, -0.5, 0))
    return world, index


def _marble(world, index, material="steel", position=(0, 0.6, 0), velocity=(0, 0, 0)):
    ball = world.add_shape(model.Shape.sphere(0.5))
    return world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)


def _controller(world, i, cells=None):
    if cells is None:
        cells = {(c, r): 0.0 for c in range(-4, 8) for r in range(-4, 4)}
    track = TrackMap(cells, cell_size=4.0)
    return MarbleController(world, i, track, marble_radius=0.5)


def _add_wall(world, index, x, material="stone"):
    wall = world.add_shape(model.Shape.box((0.5, 3, 8)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=wall, physicsMaterial=index[material]),
                   position=(x, 1.0, 0))


def _run(world, ctrl, frames=120, dt=1 / 60.0):
    for _ in range(frames):
        world.step(dt)
        ctrl.update(dt)


def test_hitting_a_stone_wall_kills_horizontal_speed():
    world, index = _world()
    _add_wall(world, index, x=6.0)
    i = _marble(world, index, velocity=(9.0, 0, 0), position=(0, 0.6, 0))
    ctrl = _controller(world, i)
    _run(world, ctrl, frames=120)
    assert ctrl.speed < 1.5              # crashed into the wall, nearly stopped


def test_hitting_a_rubber_wall_returns_energy():
    world, index = _world()
    _add_wall(world, index, x=6.0, material="rubber_pad")
    i = _marble(world, index, material="rubber", velocity=(9.0, 0, 0), position=(0, 0.6, 0))
    ctrl = _controller(world, i)
    _run(world, ctrl, frames=120)
    # A springy bumper sends it back the other way with real speed retained.
    assert world.linear_velocity[i][0] < -2.0


def test_hard_landing_kills_horizontal_speed():
    world, index = _world()
    # Falling fast while also moving sideways; the landing should scrub the slide.
    i = _marble(world, index, velocity=(6.0, -9.0, 0), position=(0, 8.0, 0))
    ctrl = _controller(world, i)
    _run(world, ctrl, frames=120)
    assert ctrl.speed < 1.5


def test_gentle_roll_is_not_treated_as_an_impact():
    world, index = _world()
    i = _marble(world, index, material="rubber", velocity=(3.0, 0, 0), position=(0, 0.6, 0))
    ctrl = _controller(world, i)
    _run(world, ctrl, frames=30)
    # No wall, no fall — a normal roll keeps most of its speed (friction only).
    assert ctrl.speed > 1.5
