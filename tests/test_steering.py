"""Can a player actually aim the marble?

Every other test of the controls asks whether a lean produces motion. That is
not the question. The question is whether a player can put the marble *where
they want it* before it has gone somewhere else, and the measure of that is one
number:

    **how far the marble travels forward while it moves one cell sideways.**

A ratio near 1 means a lane change costs a lane of travel, which is a board a
player can thread. At 2.77 -- what the game did before this was measured -- a
lane change costs three cells, so by the time the marble is where it was aimed
it is three cells past the thing it was being aimed at. Every fragment about aim
was being asked of a control that could not aim.
"""
import math

import pytest

from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level

DT = 1 / 120.0
CELL = 4.0

#: The worst ratio the controls may have and still be aimable.  Chosen from the
#: measurement rather than from taste: at 1.3 a lane change costs a lane and a
#: third, which threads a three-wide lane; at 2.77 it does not.
AIMABLE = 1.6


def _open_board(across=14, along=60):
    cells = {(col, row): 0.0 for col in range(-across, across + 1)
             for row in range(-2, along)}
    return Level(name='open', cells=cells, start_cell=(0, 0),
                 finish_cell=(0, along - 1), time_limit=900.0,
                 features=[Finish((0, along - 1))], cell_size=CELL)


#: Which way to hold the board to travel toward the finish.  The steering axes
#: are the camera's and +forward is up-screen, so going *down* the board is -1.
ONWARD = -1.0


def _control_ratio(**named):
    """Cells travelled forward per cell moved sideways, at cruising speed.

    The marble is driven the whole way -- forward held, right added on top of it
    -- because that is what a player does, and because the board no longer moves
    anything by itself: measured against an untouched board this is nought over
    nought, which flatters the controls rather than testing them.
    """
    game = MarbleGame(_open_board(), **named)
    world, index = game.scene.world, game.marble.index
    for _ in range(int(4.0 / DT)):          # reach a cruising speed
        game.lean(ONWARD, 0.0)
        game.advance(DT)
    cruise = game.controller.speed
    began = (float(world.position[index][0]), float(world.position[index][2]))
    while abs(float(world.position[index][0]) - began[0]) < CELL:
        game.lean(ONWARD, 1.0)
        game.advance(DT)
        if float(world.position[index][2]) - began[1] > 40 * CELL:
            break
    return (float(world.position[index][2]) - began[1]) / CELL, cruise


# -- the property ------------------------------------------------------------------

def test_a_lane_change_costs_about_a_lane_of_travel():
    """The whole of whether the game can be aimed."""
    ratio, cruise = _control_ratio()
    assert ratio <= AIMABLE, \
        ('a lane change costs %.2f cells of travel at %.1f m/s: a player cannot '
         'put the marble anywhere' % (ratio, cruise))


def test_the_marble_still_rolls_along_at_a_useful_pace():
    """Aimable must not mean becalmed: the fix for steering is not to stop."""
    _, cruise = _control_ratio()
    assert cruise >= 2.5, 'cruising at %.1f m/s is a crawl' % cruise


def test_leaning_further_buys_more_authority():
    tight, _ = _control_ratio(player_tilt=math.radians(20))
    wide, _ = _control_ratio(player_tilt=math.radians(45))
    assert wide < tight, 'a bigger lean bought nothing: %.2f against %.2f' % (
        wide, tight)


def test_a_steeper_board_costs_authority():
    """What the base lean trades: pace against being able to aim."""
    gentle, _ = _control_ratio(base_tilt=math.radians(6))
    steep, _ = _control_ratio(base_tilt=math.radians(16))
    assert gentle < steep


@pytest.mark.parametrize('marble', ['steel', 'rubber', 'ice', 'glass'])
def test_every_marble_can_be_aimed(marble):
    ratio, _ = _control_ratio(marble_material=marble)
    assert ratio <= AIMABLE * 1.5, '%s: %.2f cells a lane' % (marble, ratio)


def test_the_board_answers_a_lean_within_a_fraction_of_a_second():
    """A control that takes half a second to begin is one a player fights."""
    from openglcontext_marble_demo.tilt import TiltRig
    rig = TiltRig()
    for _ in range(int(0.2 / DT)):
        rig.update(DT, 0.0, 1.0)
    assert rig.roll > rig.limit * 0.4, \
        'after 0.2 s the board has only reached %.0f%% of its lean' % (
            100 * rig.roll / rig.limit)
