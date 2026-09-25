"""Tests for the HUD's text content (the pure part).

The HUD's *drawing* is GL (verified by the rendered captures); its *content* — what
the timer, speed, material, and status lines say for a given game state — is pure
string formatting and is unit tested here.
"""
import math

from openglcontext_marble_demo import hud
from openglcontext_marble_demo.game import LOST, SPIN, TILT, WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Level


def _game(time_limit=42.0, marble="steel"):
    cells = {(c, 0): 0.0 for c in range(0, 5)}
    level = Level(name="t", cells=cells, start_cell=(0, 0), finish_cell=(4, 0),
                  time_limit=time_limit, features=[Finish((4, 0))])
    return MarbleGame(level, marble_material=marble)


def test_timer_line_shows_remaining_seconds():
    game = _game(time_limit=42.0)
    lines = hud.hud_lines(game)
    assert any("42.0" in line for line in lines)


def test_material_name_is_shown():
    game = _game(marble="chrome")
    assert any("chrome" in line.lower() for line in hud.hud_lines(game))


def test_timer_line_is_marked_urgent_when_low():
    game = _game(time_limit=5.0)
    assert not hud.time_is_urgent(game)          # 5s over the 4s threshold... check default
    game.time_left = 3.0
    assert hud.time_is_urgent(game)


def test_won_state_shows_a_win_banner():
    game = _game()
    game.state = WON
    assert hud.banner(game) is not None
    assert "win" in hud.banner(game).lower() or "finish" in hud.banner(game).lower()


def test_lost_state_shows_a_lose_banner():
    game = _game()
    game.state = LOST
    assert hud.banner(game) is not None
    assert "time" in hud.banner(game).lower() or "lose" in hud.banner(game).lower()


def test_no_banner_while_playing():
    game = _game()
    assert hud.banner(game) is None


# -- what the spike needs to see -----------------------------------------------

def test_the_lean_line_reports_the_board_and_the_control_model():
    """Judging a control model means seeing what it is doing."""
    game = _game()
    game.tilt.roll = math.radians(17)
    line = [l for l in hud.hud_lines(game) if "LEAN" in l]
    assert line and "17" in line[0]


def test_the_lean_line_says_which_model_is_driving():
    assert any(TILT in l.lower() for l in hud.hud_lines(_game()))
    spun = _game()
    spun.control = SPIN
    assert any(SPIN in l.lower() for l in hud.hud_lines(spun))
