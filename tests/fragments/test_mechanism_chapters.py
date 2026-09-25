"""The four chapters that make the mechanisms reachable from a generated board.

Seven mechanisms were merged and not one of them could appear on a board the
generator built, because no fragment in the library used one. These are those
fragments, and what a test can hold them to is first that they *contain* the
mechanism — which is the whole gap being closed — and then that each imposes the
rule it was built around.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces, storygen
from openglcontext_marble_demo.game import PLAYING, MarbleGame
from openglcontext_marble_demo.mechanisms import registry
from openglcontext_marble_demo.mechanisms.lever import Door, Lever, channels
from openglcontext_marble_demo.mechanisms.pegs import PegBoard
from openglcontext_marble_demo.mechanisms.rockfall import Rockfall
from openglcontext_marble_demo.mechanisms.water import Water

CHAPTERS = {'plinko': PegBoard, 'scree': Rockfall, 'cistern': Water,
            'locked_room': Lever}


def _built(name, variant=None, seed=3):
    return fragments.build(name, random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


# -- the gap being closed ----------------------------------------------------------

@pytest.mark.parametrize('name', sorted(CHAPTERS))
def test_the_chapter_contains_its_mechanism(name):
    """A mechanism no fragment uses is a mechanism no generated board can hold."""
    assert any(isinstance(f, CHAPTERS[name]) for f in _built(name).features)


@pytest.mark.parametrize('name', sorted(CHAPTERS))
def test_every_variant_carries_it_too(name):
    for variant in fragments.library()[name].variants:
        assert any(isinstance(f, CHAPTERS[name])
                   for f in _built(name, variant).features), variant


def test_a_generated_board_can_now_hold_a_mechanism():
    """The point of the four: the generator draws from the library, so a
    mechanism in the library is a mechanism a generated board can contain."""
    kinds = tuple(registry().values())
    seen = set()
    for seed in range(20):
        for piece in storygen.compose(seed, chapters=10).build(seed).placed.values():
            seen |= {type(f).__name__ for f in piece.features
                     if isinstance(f, kinds)}
    assert seen, 'twenty generated boards and not one mechanism among them'


# -- each chapter's own rule --------------------------------------------------------

def test_a_locked_room_puts_a_door_on_the_lever_s_channel():
    """A lever with no door is a lever nothing listens to."""
    piece = _built('locked_room')
    levers = [f for f in piece.features if isinstance(f, Lever)]
    doors = [f for f in piece.features if isinstance(f, Door)]
    assert levers and doors
    assert {lever.channel for lever in levers} == {door.channel for door in doors}


def test_a_locked_room_gives_you_room_to_get_a_run_at_it():
    """The blow wants four metres a second, and a marble builds that on floor."""
    piece = _built('locked_room')
    rows = {row for _, row in piece.cells}
    assert len(rows) >= 6, 'only %d cells of run-up' % len(rows)


@pytest.mark.slow
def test_the_door_opens_when_the_lever_is_thrown():
    """End to end, through the game: the door is a wall until it is not."""
    game = MarbleGame(_built('locked_room').level(time_limit=900.0))
    channel = list(channels(game.build).values())[0]
    shut = [float(game.scene.world.position[door][1]) for door in channel.doors]
    channel.throw()
    opened = [float(game.scene.world.position[door][1]) for door in channel.doors]
    assert opened < shut, 'the door did not move: %r then %r' % (shut, opened)


def test_a_cistern_has_somewhere_under_the_plug_to_land():
    """The water's own documentation asks for this: the marble leaves through
    the floor and keeps falling."""
    piece = _built('cistern')
    pool = [f for f in piece.features if isinstance(f, Water)][0]
    under = [height for cell, height in piece.cells.items()
             if height < piece.cells[pool.cell] - 1.0]
    assert under, 'nothing below the pool to catch what comes out of it'


def test_a_cistern_can_be_walked_as_well_as_fallen_through():
    """The stair beside it, which is the long way to the same place — and what
    makes the two halves of the piece one region rather than two."""
    piece = _built('cistern')
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell)


def test_a_plinko_gathers_every_slot_back_onto_one_way_on():
    """A fragment only passable through the right slot is one most runs fail for
    no reason a player can see."""
    piece = _built('plinko')
    board = [f for f in piece.features if isinstance(f, PegBoard)][0]
    for slot in range(board.width):
        cell = board.cell_at(board.rows() - 1, slot - board.width // 2)
        if cell in piece.cells:
            assert pieces.joined(piece.cells, cell, piece.exit.cell), slot


def test_a_scree_lands_you_somewhere_wide():
    """The whole point is arriving where you did not aim, so the landing has to
    be wider than the thing that threw you at it."""
    piece = _built('scree')
    fall = [f for f in piece.features if isinstance(f, Rockfall)][0]
    rows = {}
    for col, row in piece.cells:
        rows.setdefault(row, set()).add(col)
    widest = max(len(cols) for cols in rows.values())
    assert widest > fall.width, 'the landing is no wider than the fall'


@pytest.mark.parametrize('name', sorted(CHAPTERS))
def test_the_chapter_plays(name):
    game = MarbleGame(_built(name).level(time_limit=900.0))
    for _ in range(240):
        game.advance(1 / 120.0)
    assert game.state in (PLAYING, 'won', 'lost'), name
