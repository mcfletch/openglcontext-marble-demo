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


# -- rolling is not landing ----------------------------------------------------

def test_rolling_down_a_slope_and_up_again_keeps_its_speed():
    """A marble that never left the ground has not landed on it.

    The speed-kill exists so a drop from a height costs something. Applied to a
    marble merely rolling downhill it also makes a dip impossible to carry
    through, which is the whole of what a kicker asks -- so what it keys on is
    whether the marble was in the air, not how hard the floor pushed back.
    """
    import random

    from openglcontext_marble_demo import pieces
    from openglcontext_marble_demo.game import MarbleGame

    piece = pieces.kicker(random.Random(3),
                          pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                      width=3), depth=3.6)
    game = MarbleGame(piece.level(time_limit=600.0))
    world, index = game.scene.world, game.marble.index
    world.linear_velocity[index] = (0.0, 0.0, 18.0)
    world.wake(index)
    # Through the dip and up the far side, which is a second of rolling; what
    # happens after that is whether it *climbs* out, which is another question.
    through = []
    for step in range(int(1.0 / (1 / 120.0))):
        game.advance(1 / 120.0)
        if step % 30 == 0:
            through.append(game.controller.speed)
    # Carrying most of what it arrived with the whole way, rather than the fifth
    # a hard landing leaves.
    assert min(through) > 18.0 * 0.7, 'the dip scrubbed a rolling marble: %r' % (
        [round(speed, 1) for speed in through],)


def test_a_marble_dropped_from_a_height_still_loses_its_speed():
    """The rule it is keyed on still has to fire when it should: the existing
    hard-landing test above is a real drop, and this says the two are told
    apart rather than both being let through."""
    world, index = _world()
    dropped = _marble(world, index, velocity=(6.0, -9.0, 0), position=(0, 8.0, 0))
    ctrl = _controller(world, dropped)
    _run(world, ctrl, frames=120)
    assert ctrl.speed < 1.5
    assert not ctrl.airborne          # it is on the ground again, and knows it
