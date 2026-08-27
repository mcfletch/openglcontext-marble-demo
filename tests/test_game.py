"""Tests for :class:`MarbleGame` — the orchestrator that makes a level playable.

The game ties a level, its physics scene, the marble controller, the countdown
timer, and the win/lose rules together behind one ``advance(dt)`` call.  All of it
is headless: the finish is detected through the real physics trigger, the clock is
real time, and win/lose are asserted directly.
"""
import numpy as np

from openglcontext_marble_demo.game import LOST, PLAYING, WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Level


def _flat_level(time_limit=30.0):
    cells = {(c, 0): 0.0 for c in range(0, 5)}
    return Level(name="flat", cells=cells, start_cell=(0, 0), finish_cell=(4, 0),
                 time_limit=time_limit, features=[Finish((4, 0))])


def test_game_starts_playing_with_full_time():
    game = MarbleGame(_flat_level(time_limit=42.0))
    assert game.state == PLAYING
    assert game.time_left == 42.0


def test_time_counts_down():
    game = MarbleGame(_flat_level())
    game.advance(0.5)
    assert game.time_left < 30.0


def test_running_out_of_time_loses():
    game = MarbleGame(_flat_level(time_limit=1.0))
    for _ in range(70):
        game.advance(1 / 30.0)
    assert game.state == LOST


def test_reaching_the_finish_wins():
    game = MarbleGame(_flat_level())
    # Drop the marble onto the finish cell; a step registers the trigger overlap.
    finish_xz = game.level.cell_center((4, 0))
    game.scene.world.position[game.marble.index] = (finish_xz[0], 0.6, finish_xz[1])
    for _ in range(5):
        game.advance(1 / 60.0)
    assert game.state == WON


def test_winning_freezes_the_clock():
    game = MarbleGame(_flat_level())
    finish_xz = game.level.cell_center((4, 0))
    game.scene.world.position[game.marble.index] = (finish_xz[0], 0.6, finish_xz[1])
    for _ in range(5):
        game.advance(1 / 60.0)
    assert game.state == WON
    frozen = game.time_left
    game.advance(1.0)
    assert game.time_left == frozen        # clock stopped once the run ended


def test_reset_restores_playing_state_and_time():
    game = MarbleGame(_flat_level(time_limit=20.0))
    for _ in range(30):
        game.advance(1 / 30.0)
    game.reset()
    assert game.state == PLAYING
    assert game.time_left == 20.0
    start = game.level.marble_start()
    assert np.allclose(game.scene.world.position[game.marble.index][[0, 2]],
                       (start[0], start[2]), atol=0.3)


def test_set_marble_material_changes_feel_and_look():
    game = MarbleGame(_flat_level(), marble_material="steel")
    i = game.marble.index
    steel_mass = game.scene.world.mass[i]
    game.set_marble_material("glass")
    assert game.marble_material == "glass"
    assert game.scene.world.mass[i] != steel_mass          # glass is lighter
    assert game.scene.world.collider_material[i] == game.material_index["glass"]
    assert game.controller.marble_material == "glass"


def test_kick_moves_the_marble():
    # A wide flat plate so a few right kicks have room to show lateral motion.
    cells = {(c, r): 0.0 for c in range(-3, 7) for r in range(-4, 4)}
    level = Level(name="plate", cells=cells, start_cell=(0, 0), finish_cell=(5, 0),
                  time_limit=60.0, features=[Finish((5, 0))])
    game = MarbleGame(level, marble_material="rubber")
    for _ in range(6):
        game.kick(0.0, 1.0)                    # steer right
    for _ in range(90):
        game.advance(1 / 60.0)
    assert game.scene.world.position[game.marble.index][0] > 1.0
