"""Tests for the lever and the door it opens.

The threshold is the mechanism, so it is what these measure: the same board,
the same marble, the same place, and only the speed of arrival different.  It
runs on the real physics through :class:`MarbleGame`, whose world reports the
blow to the lever's contact listener, so what is asserted is the blow the
solver recorded rather than a number handed to the lever by the test.
"""
import numpy as np
import pytest

from openglcontext_marble_demo import levelfile
from openglcontext_marble_demo.game import MARBLE_RADIUS, SPIN, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.lever import Door, Lever, channels

#: Cell (2, 0) is the lever, (3, 0) carries the door on its far side.
LEVER_CELL = (2, 0)
DOOR_CELL = (3, 0)
HARDNESS = 4.0


def _board(lever_cell=LEVER_CELL):
    """A flat corridor along +X with a lever at (2, 0) and its door past (3, 0)."""
    cells = {(col, 0): 0.0 for col in range(0, 7)}
    features = [Lever(cell=lever_cell, channel='gate', hardness=HARDNESS),
                Door(cell=DOOR_CELL, side='E', channel='gate'),
                Finish((6, 0))]
    return Level(name='lever', cells=cells, start_cell=(0, 0), finish_cell=(6, 0),
                 time_limit=120.0, features=features)


def _game(level=None):
    """The board as a headless game: no window, and no lean to drag the marble off it."""
    return MarbleGame(level if level is not None else _board(),
                      marble_material='steel', control=SPIN,
                      base_tilt=0.0, player_tilt=0.0)


def _roll(game, speed, x=5.0, frames=150):
    """Send the marble along +X from ``x`` at ``speed``, rolling rather than sliding."""
    world, marble = game.scene.world, game.marble.index
    world.place_body(marble, position=(x, MARBLE_RADIUS + 0.05, 0.0))
    world.linear_velocity[marble] = (speed, 0.0, 0.0)
    # Rolling without slipping: about -Z for travel along +X, so the contact
    # patch is still and friction is not spending the speed spinning it up.
    world.angular_velocity[marble] = (0.0, 0.0, -speed / MARBLE_RADIUS)
    world.wake(marble)
    for _ in range(frames):
        game.advance(1 / 60.0)
    return world.position[marble]


def _door_height(game):
    channel = channels(game.build)['gate']
    return float(game.scene.world.position[channel.doors[0]][1])


# -- the threshold -------------------------------------------------------

def test_a_gentle_marble_leaves_the_lever_alone():
    game = _game()
    shut = _door_height(game)
    _roll(game, 2.0)
    assert not channels(game.build)['gate'].thrown
    assert _door_height(game) == shut


def test_a_fast_marble_throws_the_lever_and_opens_the_door():
    game = _game()
    shut = _door_height(game)
    _roll(game, 9.0)
    assert channels(game.build)['gate'].thrown
    assert _door_height(game) < shut - 1.0


def test_the_lever_records_the_blow_it_answered_to():
    """The throw happens above the threshold and not below it, either side of one number."""
    below, above = _game(_board()), _game(_board())
    _roll(below, HARDNESS * 0.5)
    _roll(above, HARDNESS * 2.5)
    assert not channels(below.build)['gate'].thrown
    assert channels(above.build)['gate'].thrown


def test_the_lever_lies_over_when_it_is_thrown():
    game = _game()
    lever = game.build.feature_bodies[0]
    upright = game.scene.world.orientation[lever.index].copy()
    _roll(game, 9.0)
    assert not np.allclose(upright, game.scene.world.orientation[lever.index])
    # And it lies low enough that the marble is not stopped by what it threw.
    assert game.scene.world.position[lever.index][1] < 0.5


def test_a_thrown_lever_stays_thrown():
    game = _game()
    _roll(game, 9.0)
    opened = _door_height(game)
    _roll(game, 2.0, x=5.0, frames=60)      # a gentle second pass changes nothing
    assert _door_height(game) == opened


def test_the_way_opens_once_the_lever_is_thrown():
    """The blow that throws it also stops the marble, so the run at it costs a pass."""
    game = _game()
    struck = _roll(game, 9.0)
    assert channels(game.build)['gate'].thrown
    assert struck[0] < 14.0                  # bounced off the lever, still short
    assert _roll(game, 5.0, frames=240)[0] > 14.0    # and now the doorway is clear


def test_one_channel_opens_every_door_on_it():
    cells = {(col, 0): 0.0 for col in range(0, 7)}
    level = Level(name='pair', cells=cells, start_cell=(0, 0), finish_cell=(6, 0),
                  time_limit=120.0,
                  features=[Lever(cell=LEVER_CELL, channel='gate', hardness=HARDNESS),
                            Door(cell=DOOR_CELL, side='E', channel='gate'),
                            Door(cell=(4, 0), side='E', channel='gate'),
                            Finish((6, 0))])
    game = _game(level)
    world = game.scene.world
    doors = channels(game.build)['gate'].doors
    assert len(doors) == 2
    shut = [float(world.position[door][1]) for door in doors]
    _roll(game, 9.0)
    assert all(world.position[door][1] < was - 1.0
               for door, was in zip(doors, shut, strict=True))


# -- the door as a barrier ----------------------------------------------

def test_a_shut_door_stops_the_marble():
    # The lever is out of the way, so what the door does is all that is measured.
    game = _game(_board(lever_cell=(0, 0)))
    position = _roll(game, 9.0, x=9.0)
    assert position[0] < 14.0                # never got past the doorway at x=14


def test_an_open_door_lets_the_marble_through():
    game = _game(_board(lever_cell=(0, 0)))
    channels(game.build)['gate'].throw()
    position = _roll(game, 9.0, x=9.0)
    assert position[0] > 14.0


# -- the file format -----------------------------------------------------

def test_a_lever_and_its_door_round_trip_through_a_file():
    level = _board()
    again = levelfile.from_json(levelfile.to_json(level))
    lever, door = again.features[0], again.features[1]
    assert isinstance(lever, Lever) and isinstance(door, Door)
    assert lever.cell == LEVER_CELL and lever.channel == 'gate'
    assert lever.hardness == HARDNESS
    assert door.cell == DOOR_CELL and door.channel == 'gate' and door.side == 'E'


def test_the_registry_knows_both_of_them():
    from openglcontext_marble_demo import mechanisms
    assert mechanisms.registry()['lever'] is Lever
    assert mechanisms.registry()['door'] is Door


def test_a_door_wants_a_side_of_the_cell_it_knows():
    with pytest.raises(KeyError):
        _game(Level(name='x', cells={(0, 0): 0.0}, start_cell=(0, 0),
                    finish_cell=(0, 0), time_limit=10.0,
                    features=[Door(cell=(0, 0), side='up')]))
