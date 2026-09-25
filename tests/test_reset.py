"""What a restarted run has to forget.

Pressing R puts the marble back at the start. A board whose levers were still
thrown and whose plugs were still open would be a different board from the one
the player started on, and the second attempt would not be a second attempt at
the same thing.

So a mechanism that holds state says how to put it back, and the game asks
every one of them on reset.
"""
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms import lever as lever_module
from openglcontext_marble_demo.mechanisms import water as water_module


def _board(features):
    cells = {(col, row): 0.0 for col in (-1, 0, 1) for row in range(8)}
    return Level(name='reset', cells=cells, start_cell=(0, 0), finish_cell=(0, 7),
                 time_limit=600.0, features=list(features) + [Finish((0, 7))])


def _game(features):
    return MarbleGame(_board(features))


# -- the game collects what can be put back -------------------------------------

def test_a_build_gathers_the_things_that_hold_state():
    game = _game([lever_module.Lever(cell=(0, 3), channel='a'),
                  lever_module.Door(cell=(0, 5), channel='a')])
    assert game.build.resettable


def test_a_board_with_nothing_stateful_gathers_nothing():
    assert not _game([]).build.resettable


# -- levers ---------------------------------------------------------------------

def test_resetting_shuts_a_thrown_door_again():
    game = _game([lever_module.Lever(cell=(0, 3), channel='a'),
                  lever_module.Door(cell=(0, 5), channel='a')])
    channel = lever_module.channels(game.build)['a']
    channel.throw()
    assert channel.thrown
    game.reset()
    assert not channel.thrown


def test_a_reset_lever_can_be_thrown_again():
    game = _game([lever_module.Lever(cell=(0, 3), channel='a'),
                  lever_module.Door(cell=(0, 5), channel='a')])
    channel = lever_module.channels(game.build)['a']
    channel.throw()
    game.reset()
    assert channel.throw() is True


def test_the_door_goes_back_where_it_was():
    game = _game([lever_module.Lever(cell=(0, 3), channel='a'),
                  lever_module.Door(cell=(0, 5), channel='a')])
    channel = lever_module.channels(game.build)['a']
    shut = [float(game.scene.world.position[door][1]) for door in channel.doors]
    channel.throw()
    opened = [float(game.scene.world.position[door][1]) for door in channel.doors]
    assert opened < shut
    game.reset()
    back = [float(game.scene.world.position[door][1]) for door in channel.doors]
    assert back == shut


# -- water ----------------------------------------------------------------------

def test_resetting_shuts_the_plug_again():
    game = _game([water_module.Water(cell=(0, 4))])
    plugs = [item for item in game.build.resettable
             if isinstance(item, water_module.Plug)]
    assert plugs
    plugs[0].release(game.scene.world, game.marble.index)
    assert plugs[0].released_at is not None
    game.reset()
    assert plugs[0].released_at is None


# -- and the run itself ----------------------------------------------------------

def test_resetting_still_does_what_it_always_did():
    game = _game([])
    game.time_left = 3.0
    game.advance(0.1)
    game.reset()
    assert game.time_left == game.level.time_limit
    assert game.controller.fall_count == 0
