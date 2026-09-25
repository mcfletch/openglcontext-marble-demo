"""A board a marble can actually get across — walls included.

Every test of connectivity until now asked whether the *cells* joined up, and a
wall is not a cell. So a board could be reported connected, and playable, and
have a wall standing across the only way forward — which is what every generated
board turned out to have.

The check that matters is the one a marble makes: from the start, over cells,
**through the gaps between walls**. And the check above that is the autopilot
actually driving it, because a route existing and a route being drivable are two
different claims.
"""
from collections import deque

import pytest

from openglcontext_marble_demo import pieces, pilot, storygen
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level, Wall

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _walled_pairs(level):
    """Every ordered pair of cells a wall stands between."""
    blocked = set()
    for feature in level.features:
        if isinstance(feature, Wall):
            step = Wall.OFFSET[feature.side]
            beyond = (feature.cell[0] + step[0], feature.cell[1] + step[1])
            blocked.add((feature.cell, beyond))
            blocked.add((beyond, feature.cell))
    return blocked


def _reachable(level):
    """Cells a marble can reach from the start, walls taken into account."""
    walls = _walled_pairs(level)
    seen = {level.start_cell}
    queue = deque(seen)
    while queue:
        cell = queue.popleft()
        for dcol, drow in NEIGHBOURS:
            beside = (cell[0] + dcol, cell[1] + drow)
            if (beside in level.cells and beside not in seen
                    and (cell, beside) not in walls):
                seen.add(beside)
                queue.append(beside)
    return seen


# -- the property itself ----------------------------------------------------------

def test_the_engine_offers_a_walled_reachability_check():
    """Cell-only reachability is the thing that hid this; the walled one is what
    every caller should be asking."""
    assert hasattr(pieces, 'navigable')


def test_a_wall_across_a_lane_makes_it_unnavigable():
    """The check has to be able to *fail*, or it is not a check."""
    cells = {(0, row): 0.0 for row in range(5)}
    level = Level(name='blocked', cells=cells, start_cell=(0, 0),
                  finish_cell=(0, 4), time_limit=60.0,
                  features=[Wall(cell=(0, 2), side='S'), Finish((0, 4))])
    assert pieces.joined(level.cells, (0, 0), (0, 4))       # the cells do join
    assert not pieces.navigable(level)                      # a marble does not


def test_an_unwalled_lane_is_navigable():
    cells = {(0, row): 0.0 for row in range(5)}
    level = Level(name='clear', cells=cells, start_cell=(0, 0),
                  finish_cell=(0, 4), time_limit=60.0, features=[Finish((0, 4))])
    assert pieces.navigable(level)


def test_a_rail_along_the_side_does_not_block_anything():
    """Rails face outward: they keep a marble on the board, not off it."""
    cells = {(col, row): 0.0 for col in (-1, 0, 1) for row in range(5)}
    rails = [Wall(cell=(-1, row), side='W') for row in range(5)]
    rails += [Wall(cell=(1, row), side='E') for row in range(5)]
    level = Level(name='railed', cells=cells, start_cell=(0, 0),
                  finish_cell=(0, 4), time_limit=60.0,
                  features=rails + [Finish((0, 4))])
    assert pieces.navigable(level)


# -- what it holds the board-builders to -------------------------------------------

def test_no_wall_ever_stands_between_two_cells_of_a_finished_board():
    """The rule that fixes this class of bug for good.

    A piece walls its own edge against the cells *it* knows about, and the piece
    that joins onto it arrives later — so the wall ends up between two floors.
    Whatever lays a board must drop those.
    """
    for seed in range(8):
        level = storygen.compose(seed, chapters=8).build(seed).level()
        for feature in level.features:
            if isinstance(feature, Wall):
                step = Wall.OFFSET[feature.side]
                beyond = (feature.cell[0] + step[0], feature.cell[1] + step[1])
                assert beyond not in level.cells, \
                    ('seed %d: a wall on %r faces %r, which is board'
                     % (seed, feature.cell, beyond))


def test_every_generated_board_is_navigable():
    for seed in range(12):
        level = storygen.compose(seed, chapters=8).build(seed).level()
        assert pieces.navigable(level), \
            ('seed %d: %d of %d cells reachable'
             % (seed, len(_reachable(level)), len(level.cells)))


def test_every_chained_board_is_navigable():
    for seed in range(8):
        board = pieces.chain(seed, ['plateau', 'ramp_down', 'plateau', 'kicker',
                                    'plateau', 'hairpin', 'plateau'])
        assert pieces.navigable(board.level()), seed


# -- and the one that matters: can it be driven ------------------------------------

@pytest.mark.slow
def test_the_autopilot_gets_across_a_generated_board():
    """A route existing and a route being drivable are two different claims.

    This is the acceptance test for a generated board: the game's own pilot,
    playing it the way a player would, reaching the end.
    """

    finished = []
    for seed in range(6):
        level = storygen.compose(seed, chapters=6).build(seed).level()
        level.time_limit = 600.0
        game = MarbleGame(level)
        driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                                 right_axis=game.tilt.right_axis)
        world, index = game.scene.world, game.marble.index
        for _ in range(int(180.0 / (1 / 120.0))):
            game.lean(*driver.lean(world.position[index],
                                   world.linear_velocity[index]))
            if game.advance(1 / 120.0) != 'playing':
                break
        finished.append(game.state == 'won')
    assert sum(finished) >= 4, \
        'the pilot finished %d of 6 generated boards' % sum(finished)
