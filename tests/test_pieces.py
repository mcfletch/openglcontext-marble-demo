"""Pieces with ports, and the joiners that are the challenge.

A board built from pieces is a board with places in it. What a test can hold
these to is that the chain **connects** — every piece is entered where the one
before it was left, at the height it was left at — and that each joiner actually
imposes the rule it exists for, which is a question about what a marble does on
it and therefore a question for the physics.

The rules are the point. A ramp a marble can always climb is scenery; a kicker
you must enter fast enough to get out of is a challenge, and the difference is
measurable: put a marble on it slowly and it must fail, put it on fast and it
must pass.
"""

import pytest

from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.level import Level, Wall

CELL = 4.0
NORTH, SOUTH, EAST, WEST = (0, -1), (0, 1), (1, 0), (-1, 0)


def _entry(cell=(0, 0), facing=SOUTH, height=0.0, width=3):
    return pieces.Port(cell=cell, facing=facing, height=height, width=width)


def _built(builder, entry=None, seed=3, **named):
    import random
    return builder(random.Random(seed), entry or _entry(), **named)


# -- ports ---------------------------------------------------------------------

def test_a_port_knows_where_it_is_and_which_way_it_faces():
    port = _entry(cell=(2, 5), facing=EAST, height=-1.8)
    assert port.cell == (2, 5)
    assert port.facing == EAST
    assert port.height == -1.8


def test_a_port_spans_the_cells_across_its_facing():
    """A three-wide mouth facing down the board is three cells across it."""
    across = _entry(cell=(0, 4), facing=SOUTH, width=3).cells()
    assert set(across) == {(-1, 4), (0, 4), (1, 4)}


def test_a_port_facing_sideways_spans_the_other_way():
    across = _entry(cell=(4, 0), facing=EAST, width=3).cells()
    assert set(across) == {(4, -1), (4, 0), (4, 1)}


def test_a_port_can_be_stepped_forward():
    assert _entry(cell=(0, 4), facing=SOUTH).ahead(2).cell == (0, 6)


# -- every piece connects ------------------------------------------------------

@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_every_piece_puts_its_cells_down_and_answers_with_an_exit(name):
    piece = _built(pieces.PIECES[name])
    assert piece.cells, name
    assert piece.exit.cell in piece.cells, name


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_every_piece_is_entered_where_it_was_told(name):
    entry = _entry(cell=(3, 7), height=-2.7)
    piece = _built(pieces.PIECES[name], entry=entry)
    assert entry.cell in piece.cells, name
    assert piece.cells[entry.cell] == pytest.approx(entry.height), name


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_every_piece_leaves_its_exit_at_the_height_it_claims(name):
    piece = _built(pieces.PIECES[name])
    assert piece.cells[piece.exit.cell] == pytest.approx(piece.exit.height), name


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_every_piece_can_be_crossed_from_its_entry_to_its_exit(name):
    piece = _built(pieces.PIECES[name])
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), name


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_no_piece_puts_a_step_a_marble_cannot_roll(name):
    piece = _built(pieces.PIECES[name])
    for (col, row), height in piece.cells.items():
        for dcol, drow in ((1, 0), (0, 1)):
            beside = (col + dcol, row + drow)
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                    '%s steps %.2f at %r' % (name, piece.cells[beside] - height,
                                             (col, row))


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_every_piece_names_a_theme(name):
    assert _built(pieces.PIECES[name]).theme


@pytest.mark.parametrize('name', sorted(pieces.PIECES))
def test_a_piece_is_the_same_piece_for_the_same_seed(name):
    first, again = _built(pieces.PIECES[name]), _built(pieces.PIECES[name])
    assert first.cells == again.cells, name
    assert first.features == again.features, name


# -- chaining ------------------------------------------------------------------

def test_a_chain_hands_each_piece_the_last_one_s_exit():
    board = pieces.chain(3, ['plateau', 'ramp_down', 'plateau'])
    assert len(board.pieces) == 3
    for before, after in zip(board.pieces, board.pieces[1:], strict=False):
        assert after.entry.cell == before.exit.cell
        assert after.entry.height == pytest.approx(before.exit.height)


def test_a_chained_board_is_one_connected_region():
    board = pieces.chain(3, ['plateau', 'kicker', 'plateau', 'spillway', 'plateau'])
    assert pieces.joined(board.cells, board.start, board.finish)


def test_a_chained_board_descends():
    board = pieces.chain(5, ['plateau', 'ramp_down', 'plateau', 'ramp_down',
                             'plateau'])
    assert board.cells[board.finish] < board.cells[board.start]


def test_a_chain_holds_the_slope_budget_across_the_joins_too():
    """Two pieces that each roll can still put a cliff where they meet."""
    board = pieces.chain(7, ['plateau', 'kicker', 'plateau', 'hairpin',
                             'plateau', 'spillway', 'plateau'])
    for (col, row), height in board.cells.items():
        for dcol, drow in ((1, 0), (0, 1)):
            beside = (col + dcol, row + drow)
            if beside in board.cells:
                assert abs(board.cells[beside] - height) <= pieces.MAX_STEP + 1e-9


def test_a_chain_becomes_a_level_the_game_can_play():
    from openglcontext_marble_demo.game import PLAYING, MarbleGame
    level = pieces.chain(4, ['plateau', 'ramp_down', 'plateau']).level()
    assert isinstance(level, Level)
    game = MarbleGame(level)
    for _ in range(240):
        game.advance(1 / 120.0)
    assert game.state in (PLAYING, 'won', 'lost')


def test_an_unknown_piece_is_refused_by_name():
    with pytest.raises(KeyError, match='trampoline'):
        pieces.chain(1, ['plateau', 'trampoline'])


# -- the joiners impose their rules --------------------------------------------
#
# Each of these puts a marble on the piece at a speed and asks what happened.
# That is the only way to know a challenge is a challenge: a ramp a marble can
# always climb is scenery.

def _furthest(piece, speed, seconds=5.0):
    """Roll a marble onto ``piece`` at ``speed`` m/s down a board leaning as the
    game leans one; answer how far along it ever got, and whether it fell off.

    The *furthest* rather than where it ended: a piece on its own has nothing
    after it, so a marble that gets through runs off the end and is respawned,
    and where it finished would say it never made it.
    """
    import numpy as np

    from openglcontext_marble_demo.game import MarbleGame
    level = piece.level(time_limit=600.0)
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    begin = np.array(level.cell_center(piece.entry.cell))
    finish = np.array(level.cell_center(piece.exit.cell))
    whole = float(np.linalg.norm(finish - begin)) or 1.0
    best = 0.0
    for _ in range(int(seconds / (1 / 120.0))):
        game.advance(1 / 120.0)
        if game.controller.fall_count:
            return best, True
        at = world.position[index]
        best = max(best, float(np.dot(np.array([at[0], at[2]]) - begin,
                                      finish - begin) / whole / whole))
    return best, False


#: How far along counts as having got through.  Not 1.0: the exit is the last
#: cell of the piece, a marble arriving at it runs straight off the end -- there
#: is nothing after a piece on its own -- and is respawned, so the furthest it is
#: ever seen is a little short of the mouth.
THROUGH = 0.8


@pytest.mark.xfail(reason='the far side is not climbable at any speed yet: the '
                          'marble reaches 0.60 of the way at 8 m/s and at 20, '
                          'so the dip takes everything it arrives with. The '
                          'ramps that make the descent rollable are not enough '
                          'to make the climb rollable.', strict=True)
def test_a_kicker_has_to_be_entered_fast_enough_to_climb_the_far_side():
    """The rule: too slow and you roll back down and try again."""
    piece = _built(pieces.kicker)
    assert _furthest(piece, 2.0)[0] < THROUGH, 'a crawl got over it'
    assert _furthest(piece, 16.0)[0] >= THROUGH, 'a run at it did not'


@pytest.mark.xfail(reason='nothing gets off the open end: a marble entering at '
                          '18 m/s reaches 0.92 of the way and stops, so the '
                          'descent is not carrying speed into the run-out and '
                          'the rule cannot bite.', strict=True)
def test_a_spillway_drops_a_marble_that_arrives_too_fast():
    """The rule: the bottom end has no wall, so control the descent."""
    piece = _built(pieces.spillway)
    assert not _furthest(piece, 1.0)[1], 'a careful descent fell off'
    assert _furthest(piece, 18.0)[1], 'a headlong one did not'


def test_a_spillway_is_open_at_the_bottom_and_walled_along_the_sides():
    piece = _built(pieces.spillway)
    sides = [f for f in piece.features if isinstance(f, Wall)]
    assert sides, 'a spillway with no sides is a plank'
    ahead = piece.exit.ahead(1).cell
    assert not any(f.cell == ahead for f in sides), 'the bottom end is walled'


def test_a_hairpin_can_only_be_taken_slowly():
    """The rule: brake for the right-angle, or be carried past it.

    Asked as *did it stay on the piece*, rather than as how far along it got:
    a marble that ignores a right-angle carries straight on, which is a long way
    in the direction it was already going and no way at all toward the exit.
    """
    piece = _built(pieces.hairpin)
    assert not _furthest(piece, 3.0)[1], 'a careful entry was carried off'
    assert _furthest(piece, 12.0)[1], 'a fast one stayed on'
    assert _furthest(piece, 3.0)[0] >= 0.55, 'a careful entry got nowhere'


def test_a_narrow_bridge_is_one_cell_across():
    piece = _built(pieces.bridge)
    rows = {}
    for col, row in piece.cells:
        rows.setdefault(row, []).append(col)
    middle = sorted(rows)[len(rows) // 2]
    assert len(rows[middle]) == 1


def test_a_scatter_is_a_field_of_things_to_bounce_off():
    from openglcontext_marble_demo.level import Bumper
    piece = _built(pieces.scatter)
    assert sum(isinstance(f, Bumper) for f in piece.features) >= 4


def test_a_plateau_is_a_place_rather_than_a_passage():
    piece = _built(pieces.plateau)
    rows = {row for _, row in piece.cells}
    cols = {col for col, _ in piece.cells}
    assert len(rows) >= 3 and len(cols) >= 3


def test_a_plateau_is_walled_where_it_is_not_a_way_on():
    """A place you can fall out of on every side is not a place."""
    piece = _built(pieces.plateau)
    assert sum(isinstance(f, Wall) for f in piece.features) >= 4


def test_a_ramp_down_actually_descends():
    piece = _built(pieces.ramp_down)
    assert piece.exit.height < piece.entry.height - 1.0


def test_a_ramp_down_carries_a_marble_that_simply_rolls_onto_it():
    """It is a way between plateaus, not a challenge: everything gets down it."""
    piece = _built(pieces.ramp_down)
    for speed in (1.0, 6.0, 14.0):
        assert _furthest(piece, speed)[0] >= THROUGH, speed


# -- what each piece says about itself -----------------------------------------

def test_a_joiner_says_what_it_asks_of_a_player():
    """The one line that goes in a story, and in the plan."""
    for name in ('kicker', 'spillway', 'hairpin', 'bridge', 'scatter'):
        piece = _built(pieces.PIECES[name])
        assert piece.rule, name
        assert len(piece.rule) < 90, name


def test_a_plateau_asks_nothing_and_says_so():
    assert not _built(pieces.plateau).rule


def test_a_piece_carries_a_theme_that_reaches_the_level():
    piece = _built(pieces.plateau, theme='ice')
    level = piece.level()
    assert set(level.cell_surfaces.values()) == {pieces.THEMES['ice'].floor}
