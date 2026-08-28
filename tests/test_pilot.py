"""The autopilot: a board played by nobody, well enough to watch.

A demo that demos itself needs a player. :class:`~openglcontext_marble_demo.pilot.Autopilot`
is one — it reads the board, works out a line, and leans the board toward it —
and because it holds no GL and is handed its own inputs, how well it plays is
something a test can measure rather than something to watch and hope.

The bar is not "plays perfectly". It is **finishes a board it is put on**, which
is what an attract mode and a recorded video both need.
"""
import math

import numpy as np
import pytest

from openglcontext_marble_demo import boards, generator, pilot
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import Level

FORWARD = np.array([0.0, 0.0, -1.0])
RIGHT = np.array([1.0, 0.0, 0.0])


def _straight(depth=8, width=3):
    from openglcontext_marble_demo.level import Finish
    cells = {(col, row): 0.0
             for col in range(-(width // 2), width - width // 2)
             for row in range(depth)}
    finish = (0, depth - 1)
    return Level(name='straight', cells=cells, start_cell=(0, 0),
                 finish_cell=finish, time_limit=90.0, features=[Finish(finish)])


def _pilot(level, **named):
    named.setdefault('forward_axis', FORWARD)
    named.setdefault('right_axis', RIGHT)
    return pilot.Autopilot(level, **named)


# -- the line it means to take -------------------------------------------------

def test_the_route_runs_from_the_start_to_the_finish():
    route = _pilot(_straight()).route
    assert route[0] == (0, 0)
    assert route[-1] == (0, 7)


def test_the_route_keeps_off_the_hazards_where_it_can():
    """The clean line exists on every board; a pilot that ignored it would spend
    the run being flung about by springs."""
    for seed in range(8):
        level = generator.generate(seed=seed, difficulty=3)
        driver = _pilot(level)
        hazards = pilot.hazard_cells(level)
        assert not (set(driver.route) & hazards), 'seed %d drives into a hazard' % seed


def test_a_board_whose_clean_line_is_blocked_still_gets_a_route():
    """Better to drive over a bumper than to stop: a route that does not exist
    is a pilot that does nothing at all."""
    level = _straight(depth=5, width=1)
    level.features = [__import__(
        'openglcontext_marble_demo.level', fromlist=['Bumper']).Bumper(cell=(0, 2))]
    assert _pilot(level).route[-1] == level.finish_cell


# -- where it is aiming --------------------------------------------------------

def test_it_aims_along_the_route_rather_than_at_the_finish():
    """Aiming straight at the finish would cut every corner, including the ones
    with nothing under them."""
    level = generator.generate(seed=1, difficulty=2)
    driver = _pilot(level)
    start = np.array(level.marble_start())
    target = driver.target(start)
    assert np.linalg.norm(target - start) < level.cell_size * 6


def test_the_aim_moves_up_the_route_as_the_marble_does():
    level = _straight(depth=10)
    driver = _pilot(level)
    early = driver.target(np.array([0.0, 0.0, 0.0]))
    later = driver.target(np.array([0.0, 0.0, 6 * level.cell_size]))
    assert later[2] > early[2]


def test_it_aims_at_the_finish_once_there_is_nothing_further_on():
    level = _straight(depth=6)
    driver = _pilot(level)
    near_the_end = np.array([0.0, 0.0, 5 * level.cell_size])
    x, z = level.cell_center(level.finish_cell)
    assert driver.target(near_the_end)[2] == pytest.approx(z)


# -- what it asks the board to do ----------------------------------------------

def test_a_marble_behind_the_line_is_steered_back_onto_it():
    level = _straight(depth=10)
    driver = _pilot(level)
    forward, right = driver.lean(np.array([2.0, 0.0, 4.0]), np.zeros(3))
    assert right < 0                              # pull back toward -X


def test_a_marble_left_of_the_line_is_steered_the_other_way():
    level = _straight(depth=10)
    driver = _pilot(level)
    _, right = driver.lean(np.array([-2.0, 0.0, 4.0]), np.zeros(3))
    assert right > 0


def test_a_demand_too_big_to_hold_keeps_the_direction_it_asked_for():
    """Clipping each axis on its own turns two-forward-one-right into
    one-and-one: a different direction, chosen by the clip."""
    level = _straight(depth=10)
    driver = _pilot(level, steer_gain=50.0)          # certain to be over the stop
    forward, right = driver.lean(np.array([-8.0, 0.0, 0.0]), np.zeros(3))
    assert math.hypot(forward, right) == pytest.approx(1.0)


def test_it_reaches_the_stop_when_it_is_in_trouble_and_not_otherwise():
    """A pilot that asks for everything the board has as a matter of course is a
    demo of the extremes rather than of the game."""
    level = generator.generate(seed=1, difficulty=2)
    game = MarbleGame(level)
    driver = _pilot(level, forward_axis=game.tilt.forward_axis,
                    right_axis=game.tilt.right_axis)
    world = game.scene.world
    at_stop = frames = 0
    for _ in range(int(20.0 / (1 / 120.0))):
        demand = driver.lean(world.position[game.marble.index],
                             world.linear_velocity[game.marble.index])
        at_stop += math.hypot(*demand) >= 0.999
        frames += 1
        game.lean(*demand)
        if game.advance(1 / 120.0) != 'playing':
            break
    assert at_stop / frames < 0.15


def test_the_demand_never_exceeds_what_a_player_could_hold():
    """A pilot that asked for more lean than a key gives would be cheating, and
    a recording of it would not be a recording of the game."""
    level = generator.generate(seed=2, difficulty=3)
    driver = _pilot(level)
    for x in range(-20, 21, 3):
        for z in range(-20, 40, 5):
            forward, right = driver.lean(np.array([float(x), 0.0, float(z)]),
                                         np.array([3.0, 0.0, 5.0]))
            assert -1.0 <= forward <= 1.0
            assert -1.0 <= right <= 1.0


def test_it_leans_back_when_it_is_running_at_the_line_too_fast():
    """The one thing a marble needs that a car does not: brakes made of slope."""
    level = _straight(depth=12)
    driver = _pilot(level)
    quick = driver.lean(np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, 14.0]))[0]
    steady = driver.lean(np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, 2.0]))[0]
    assert quick > steady          # +forward is up-slope, which is the brake


def test_sideways_drift_is_damped_rather_than_chased():
    """Steering only at the error makes a pilot that weaves; steering at the
    error *and* the drift makes one that settles."""
    level = _straight(depth=10)
    driver = _pilot(level)
    still = driver.lean(np.array([1.0, 0.0, 4.0]), np.zeros(3))[1]
    already_returning = driver.lean(np.array([1.0, 0.0, 4.0]),
                                    np.array([-6.0, 0.0, 0.0]))[1]
    assert already_returning > still


# -- does it actually play? ----------------------------------------------------

def _drive(level, seconds=90.0, dt=1 / 120.0, **named):
    game = MarbleGame(level, **named)
    driver = _pilot(level, forward_axis=game.tilt.forward_axis,
                    right_axis=game.tilt.right_axis)
    world = game.scene.world
    steps = int(seconds / dt)
    for _ in range(steps):
        game.lean(*driver.lean(world.position[game.marble.index],
                               world.linear_velocity[game.marble.index]))
        if game.advance(dt) != 'playing':
            break
    return game


@pytest.mark.slow
def test_it_finishes_a_straight_board():
    assert _drive(_straight(depth=12)).state == WON


@pytest.mark.slow
def test_it_finishes_generated_boards():
    """The bar: put it on a board and it gets to the end of it."""
    finished = 0
    for seed in range(12):
        level = generator.generate(seed=seed, difficulty=2)
        level.time_limit = 240.0          # the clock is not what is under test
        if _drive(level, seconds=180.0).state == WON:
            finished += 1
    assert finished >= 10, 'only %d of 12 boards finished' % finished


@pytest.mark.slow
def test_it_finishes_hard_boards_too():
    finished = 0
    for seed in range(8):
        level = generator.generate(seed=seed, difficulty=4)
        level.time_limit = 240.0
        if _drive(level, seconds=180.0).state == WON:
            finished += 1
    assert finished >= 6, 'only %d of 8 boards finished' % finished


@pytest.mark.slow
def test_it_beats_the_clock_it_is_given():
    """A run that finishes after the timer has run out is a lost run."""
    won = 0
    for seed in range(12):
        if _drive(generator.generate(seed=seed, difficulty=2),
                  seconds=120.0).state == WON:
            won += 1
    assert won >= 8, 'only %d of 12 runs beat the clock' % won


@pytest.mark.slow
def test_it_does_not_spend_the_run_falling_off():
    """Falling off is what a recording of a demo must not mostly show."""
    falls = 0
    for seed in range(8):
        level = generator.generate(seed=seed, difficulty=2)
        level.time_limit = 240.0
        falls += _drive(level, seconds=120.0).controller.fall_count
    assert falls <= 8, '%d falls over 8 boards' % falls


@pytest.mark.slow
def test_the_same_board_is_played_the_same_way_twice():
    """A pilot is a function of the board and the marble, so a recording of one
    run and a recording of the next are the same recording."""
    level = generator.generate(seed=5, difficulty=2)
    first = _drive(level, seconds=40.0)
    again = _drive(level, seconds=40.0)
    assert np.allclose(first.scene.world.position[first.marble.index],
                       again.scene.world.position[again.marble.index])


# -- the pieces ----------------------------------------------------------------

def test_hazard_cells_are_the_ones_a_marble_cannot_roll_across():
    level = generator.generate(seed=3, difficulty=4)
    found = pilot.hazard_cells(level)
    assert found <= set(level.cells)
    assert all(isinstance(cell, tuple) for cell in found)


def test_a_route_over_a_board_with_no_way_through_is_empty():
    level = _straight(depth=4)
    level.cells = {(0, 0): 0.0, (0, 3): 0.0}
    level.finish_cell = (0, 3)
    assert _pilot(level).route == []


def test_a_pilot_with_no_route_asks_for_nothing():
    level = _straight(depth=4)
    level.cells = {(0, 0): 0.0, (0, 3): 0.0}
    level.finish_cell = (0, 3)
    assert _pilot(level).lean(np.zeros(3), np.zeros(3)) == (0.0, 0.0)


def test_the_route_is_the_boards_own_shortest_way():
    level = generator.generate(seed=4, difficulty=2)
    driver = _pilot(level)
    shortest = boards.distances(set(level.cells), level.start_cell)[level.finish_cell]
    assert len(driver.route) >= shortest + 1
    assert math.isfinite(len(driver.route))
