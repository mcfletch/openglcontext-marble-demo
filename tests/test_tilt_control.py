"""Leaning the board, end to end: held input reaches the marble through gravity.

:mod:`~openglcontext_marble_demo.tilt` is tested on its own; this asks the
question that only the whole game can answer — does a held lean actually move the
marble, and does the board that gets drawn agree with the one being simulated.
All against the real physics world, with no window.
"""
import math

import numpy as np
import pytest
from OpenGLContext.scenegraph import basenodes

from openglcontext_marble_demo.game import SPIN, TILT, MarbleGame
from openglcontext_marble_demo.level import Level

CELL = 4.0


def _plain_board(cols=7, rows=7):
    """A wide flat board, so a lean has somewhere to take the marble."""
    cells = {(c, r): 0.0 for c in range(-cols // 2, cols // 2 + 1)
             for r in range(-1, rows)}
    return Level(name="plain", cells=cells, start_cell=(0, 0), finish_cell=(0, rows - 1),
                 time_limit=600.0, features=[], cell_size=CELL)


def _game(control=TILT, **named):
    named.setdefault('base_tilt', math.atan(0.22))
    return MarbleGame(_plain_board(), control=control, **named)


def _run(game, forward=0.0, right=0.0, seconds=2.0, dt=1 / 120.0):
    """Hold a lean for ``seconds``; return how far the marble moved."""
    start = game.scene.world.position[game.marble.index].copy()
    for _ in range(int(round(seconds / dt))):
        game.lean(forward, right)
        game.advance(dt)
    return game.scene.world.position[game.marble.index] - start


# -- the lean reaches the marble -----------------------------------------------

def test_an_untouched_board_rolls_the_marble_downhill():
    moved = _run(_game(), seconds=2.0)
    assert moved[2] > 0.5           # +Z, toward the finish


def test_holding_right_takes_the_marble_right():
    moved = _run(_game(), right=1.0, seconds=2.0)
    assert moved[0] > 0.5


def test_holding_left_takes_the_marble_left():
    moved = _run(_game(), right=-1.0, seconds=2.0)
    assert moved[0] < -0.5


def test_holding_forward_holds_the_marble_up_against_the_slope():
    """Full lean up-slope beats the board's own lean: the marble comes back."""
    moved = _run(_game(player_tilt=math.radians(26)), forward=1.0, seconds=2.0)
    assert moved[2] < 0


def test_a_longer_hold_goes_further_than_a_shorter_one():
    """The lean is an acceleration, so holding it is how far you go."""
    brief = _run(_game(), right=1.0, seconds=0.5)
    sustained = _run(_game(), right=1.0, seconds=1.5)
    assert abs(sustained[0]) > abs(brief[0]) * 2


def test_half_a_lean_travels_less_than_a_full_one():
    """Every size in between exists — the thing a kick-per-keypress cannot offer."""
    half = _run(_game(), right=0.5, seconds=1.5)
    full = _run(_game(), right=1.0, seconds=1.5)
    assert 0 < half[0] < full[0]


def test_letting_go_stops_adding_sideways_pull():
    """Released, the board settles back to level and stops pulling sideways.

    Settles rather than snaps: the lean is damped, so what is asserted is that
    what is left is too small to steer with.
    """
    game = _game()
    _run(game, right=1.0, seconds=1.0)
    _run(game, right=0.0, seconds=3.0)
    assert abs(math.degrees(game.tilt.roll)) < 0.05
    assert abs(game.scene.world.gravity.direction[0]) < 1e-3


# -- grip changes the feel, not the authority ----------------------------------

def test_every_marble_can_still_be_steered():
    """Gravity acts whatever the ball is made of, so no marble loses control.

    Under the spin model steering authority came from friction, so a slick
    marble on a slick surface had almost none.
    """
    for marble in ('steel', 'rubber', 'ice', 'glass', 'wood', 'chrome'):
        moved = _run(MarbleGame(_plain_board(), control=TILT, marble_material=marble),
                     right=1.0, seconds=2.0)
        assert moved[0] > 0.5, marble


def test_a_slick_marble_still_slides_further_than_a_grippy_one():
    """Friction stays a difference the player feels; it just is not the control."""
    slick = _run(MarbleGame(_plain_board(), marble_material='ice'), right=1.0, seconds=2.0)
    grippy = _run(MarbleGame(_plain_board(), marble_material='rubber'), right=1.0, seconds=2.0)
    assert abs(slick[0]) > abs(grippy[0])


# -- the spin model is the same class with the lean bounded at zero ------------

def test_the_spin_model_ignores_a_lean():
    game = _game(control=SPIN)
    moved = _run(game, right=1.0, seconds=2.0)
    assert game.tilt.roll == 0.0
    assert abs(moved[0]) < 1e-6


def test_the_spin_model_keeps_the_board_leaning_downhill():
    game = _game(control=SPIN)
    _run(game, seconds=1.0)
    assert game.scene.world.gravity.direction[2] > 0


def test_the_spin_model_still_steers_by_kicking():
    game = _game(control=SPIN)
    start = game.scene.world.position[game.marble.index].copy()
    for i in range(240):
        if i % 12 == 0:
            game.kick(0.0, 1.0)
        game.advance(1 / 120.0)
    assert game.scene.world.position[game.marble.index][0] - start[0] > 0.5


# -- the board that is drawn is the board that is simulated --------------------

def test_the_drawn_board_leans_around_the_marble():
    """The rotation centre follows the ball, so the ball stays put on screen."""
    game = _game()
    _run(game, right=1.0, seconds=1.0)
    assert np.allclose(game.board.center,
                       game.scene.world.position[game.marble.index])


def test_the_drawn_board_is_level_until_the_player_leans_it():
    game = _game()
    _run(game, seconds=0.5)
    assert game.board.rotation[3] == pytest.approx(0.0)


def test_the_drawn_board_leans_once_the_player_does():
    game = _game()
    _run(game, right=1.0, seconds=1.0)
    assert game.board.rotation[3] > 0


def test_resetting_levels_the_board():
    game = _game()
    _run(game, right=1.0, seconds=1.0)
    game.reset()
    assert game.tilt.roll == 0.0
    assert game.board.rotation[3] == pytest.approx(0.0)


def test_a_finished_run_stops_taking_input():
    """Nothing the player holds after the clock runs out moves the board."""
    game = MarbleGame(_plain_board(), control=TILT)
    game.time_left = 0.05
    game.advance(0.1)
    game.lean(0.0, 1.0)
    game.advance(0.1)
    assert game.tilt.roll == 0.0


# -- the scene graph --------------------------------------------------------

def test_the_scene_graph_hangs_the_world_off_the_leaning_board():
    game = _game()
    graph = game.scene_graph()
    assert game.board in graph.children


def test_the_light_and_the_sky_do_not_lean_with_the_board():
    """A sun that swung across the sky on every steer would be the board's lean
    read as the world's."""
    game = _game()
    sun = basenodes.DirectionalLight(direction=(-0.4, -1, -0.5))
    graph = game.scene_graph(extra=[sun])
    assert sun in graph.children
    assert sun not in game.board.children


# -- how freely the marble rolls -----------------------------------------------
#
# The scenegraph physics manager damps linear and angular motion by default
# (0.3 and 1.5 per second), which settles a scene of boxes and is a continuous
# brake on a ball whose whole job is to roll.  A rolling game names its own.

def test_the_marble_rolls_under_the_games_own_damping_not_the_managers():
    game = _game()
    world = game.scene.world
    assert world.default_linear_damping < 0.3
    assert world.default_angular_damping < 1.5


def test_less_damping_carries_the_marble_further():
    """The number is a real one: it is most of how fast the game feels.

    Under a held lean rather than an untouched board.  The board is level until
    somebody leans it, so an untouched one moves the marble nowhere at all and
    both dampings carry it exactly as far: nought.  Damping is a brake, and a
    brake is only measurable against something driving.
    """
    slow = MarbleGame(_plain_board(rows=14), damping=(0.3, 1.5))
    quick = MarbleGame(_plain_board(rows=14), damping=(0.05, 0.2))
    far = _run(quick, forward=-1.0, seconds=4.0)[2]
    near = _run(slow, forward=-1.0, seconds=4.0)[2]
    assert far > near * 1.5, \
        'the game\'s own damping carried it %.1f m and the manager\'s %.1f' % (far, near)
    assert Level        # imported for the board the helper builds


def test_the_damping_the_game_asks_for_is_the_damping_it_gets():
    game = MarbleGame(_plain_board(), damping=(0.11, 0.22))
    assert game.scene.world.default_linear_damping == pytest.approx(0.11)
    assert game.scene.world.default_angular_damping == pytest.approx(0.22)
