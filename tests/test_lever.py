"""A lever goes over for the marble's hard blow, and for nothing else's.

Run against a whole game on the real solver: the paddle hears contact events
inside the physics step, so what reaches it is what the world reports.
"""
import numpy as np
from omi_physics import model

from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms import lever as lever_module

DT = 1 / 60.0


def _game():
    cells = {(col, row): 0.0 for col in (-1, 0, 1) for row in range(8)}
    level = Level(name='lever', cells=cells, start_cell=(0, 0), finish_cell=(0, 7),
                  time_limit=600.0,
                  features=[lever_module.Lever(cell=(0, 3), channel='a', facing=(0, 1)),
                            lever_module.Door(cell=(0, 5), channel='a'),
                            Finish((0, 7))])
    return MarbleGame(level)


def _run_at_the_paddle(game, body, speed=9.0, frames=90):
    """Send ``body`` along +Z into the lever's cell at ``speed``."""
    world = game.scene.world
    x, z = game.level.cell_center((0, 3))
    world.place_body(body, position=(x, 0.6, z - 3.0))
    world.linear_velocity[body] = (0.0, 0.0, speed)
    world.wake(body)
    for _ in range(frames):
        game.advance(DT)


def _park_the_marble(game):
    world = game.scene.world
    x, z = game.level.cell_center((1, 0))
    world.place_body(game.marble.index, position=(x, 0.6, z))
    world.linear_velocity[game.marble.index] = (0.0, 0.0, 0.0)


def test_the_marble_throws_it():
    game = _game()
    _run_at_the_paddle(game, game.marble.index)
    assert lever_module.channels(game.build)['a'].thrown


def test_something_else_striking_it_as_hard_does_not():
    game = _game()
    _park_the_marble(game)
    world = game.scene.world
    crate = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                           collider=model.Collider(
                               shape=world.add_shape(model.Shape.box((0.8, 0.8, 0.8)))))
    _run_at_the_paddle(game, crate)
    assert not lever_module.channels(game.build)['a'].thrown


def test_after_a_reset_the_paddle_stands_and_can_be_thrown_again():
    game = _game()
    world = game.scene.world
    paddle = next(body.index for body in game.build.feature_bodies
                  if np.allclose(world.position[body.index][[0, 2]],
                                 game.level.cell_center((0, 3)), atol=0.1))
    upright = world.orientation[paddle].copy()
    _run_at_the_paddle(game, game.marble.index)
    assert not np.allclose(world.orientation[paddle], upright)
    game.reset()
    assert np.allclose(world.orientation[paddle], upright)
    _run_at_the_paddle(game, game.marble.index)
    assert lever_module.channels(game.build)['a'].thrown


def test_the_world_keeps_no_log_of_the_contacts_it_reports():
    game = _game()
    _run_at_the_paddle(game, game.marble.index, frames=240)
    assert len(game.scene.world.contact_log) == 0
    assert game.scene.world.contact_log.dropped == 0
