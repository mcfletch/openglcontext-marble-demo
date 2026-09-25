"""Gates the finish waits for: what makes a route a route rather than a direction.

A board whose finish takes anyone who reaches it has no route in it -- every way
across is as good as every other, and the shape of the board is decoration. A
board with gates has an order to it, and the places a designer built are places
you have to go.

The rule is one sentence: **the finish refuses until every gate has been passed**,
and the game says how many are left so a player is never guessing.
"""

import json

from openglcontext_marble_demo import hud, levelfile
from openglcontext_marble_demo.game import PLAYING, WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Gate, Level


def _board(gates=(), depth=10):
    cells = {(col, row): 0.0 for col in (-1, 0, 1) for row in range(depth)}
    finish = (0, depth - 1)
    features = [Gate(cell=cell, order=index) for index, cell in enumerate(gates)]
    return Level(name='gated', cells=cells, start_cell=(0, 0), finish_cell=finish,
                 time_limit=600.0, features=features + [Finish(finish)])


def _walk(game, cells, dt=1 / 120.0):
    """Put the marble on each cell in turn, letting the triggers see it."""
    world, index = game.scene.world, game.marble.index
    for cell in cells:
        x, z = game.level.cell_center(cell)
        world.position[index] = (x, game.level.cells[cell] + 0.6, z)
        world.linear_velocity[index] = (0.0, 0.0, 0.0)
        world.wake(index)
        for _ in range(12):
            game.advance(dt)


# -- a board with no gates behaves as it always did -----------------------------

def test_a_board_with_no_gates_is_won_by_reaching_the_finish():
    game = MarbleGame(_board())
    _walk(game, [(0, 9)])
    assert game.state == WON


def test_a_board_with_no_gates_says_there_is_nothing_left_to_reach():
    assert MarbleGame(_board()).gates_left == 0


# -- gates ---------------------------------------------------------------------

def test_the_finish_refuses_while_a_gate_is_unpassed():
    game = MarbleGame(_board(gates=[(0, 4)]))
    _walk(game, [(0, 9)])
    assert game.state == PLAYING


def test_the_finish_takes_you_once_every_gate_has_been_passed():
    game = MarbleGame(_board(gates=[(0, 4)]))
    _walk(game, [(0, 4), (0, 9)])
    assert game.state == WON


def test_passing_a_gate_counts_it_off():
    game = MarbleGame(_board(gates=[(0, 3), (0, 6)]))
    assert game.gates_left == 2
    _walk(game, [(0, 3)])
    assert game.gates_left == 1
    _walk(game, [(0, 6)])
    assert game.gates_left == 0


def test_passing_the_same_gate_twice_counts_once():
    game = MarbleGame(_board(gates=[(0, 3), (0, 6)]))
    _walk(game, [(0, 3), (0, 3), (0, 3)])
    assert game.gates_left == 1


def test_gates_can_be_passed_in_any_order():
    """The order they are numbered in is what a story means by them, not a rule
    the board enforces: a player who finds the second one first has found it."""
    game = MarbleGame(_board(gates=[(0, 3), (0, 6)]))
    _walk(game, [(0, 6), (0, 3), (0, 9)])
    assert game.state == WON


def test_resetting_a_run_shuts_the_gates_again():
    game = MarbleGame(_board(gates=[(0, 4)]))
    _walk(game, [(0, 4)])
    assert game.gates_left == 0
    game.reset()
    assert game.gates_left == 1
    _walk(game, [(0, 9)])
    assert game.state == PLAYING


def test_running_out_of_time_still_loses_whatever_the_gates_say():
    game = MarbleGame(_board(gates=[(0, 4)]))
    game.time_left = 0.05
    game.advance(0.1)
    assert game.state == 'lost'


# -- what the player is told ----------------------------------------------------

def test_the_hud_says_how_many_are_left():
    game = MarbleGame(_board(gates=[(0, 3), (0, 6)]))
    assert any('2' in line for line in hud.hud_lines(game) if 'GATE' in line)
    _walk(game, [(0, 3)])
    assert any('1' in line for line in hud.hud_lines(game) if 'GATE' in line)


def test_the_hud_leaves_the_line_out_when_a_board_has_no_gates():
    """A read-out that always says zero is a read-out nobody reads."""
    assert not [line for line in hud.hud_lines(MarbleGame(_board()))
                if 'GATE' in line]


# -- the file format ------------------------------------------------------------

def test_a_gate_survives_being_saved_and_read_back():

    before = _board(gates=[(0, 3), (0, 6)])
    after = levelfile.from_json(json.loads(json.dumps(levelfile.to_json(before))))
    assert after.features == before.features


def test_a_gate_is_placeable_in_the_editor():
    """A waypoint a designer cannot put down is a waypoint no board has."""
    assert 'gate' in levelfile.FEATURES
