"""The fork: two ways on, and what each of them costs.

A fragment with one exit asks a player to do a thing. This one asks them to
choose, so what a test has to hold it to is that the choice is real — that both
ways exist, that both arrive at the same place, and that the quick one is quick
enough to be worth the plank it is made of.

The last of those is a question for the physics rather than for the cell grid,
so it is asked by putting the autopilot on the piece twice, once aimed at each
exit, and timing what it managed.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import Finish, Wall

STEP = 1 / 120.0


def _entry():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('split', random.Random(seed), _entry(),
                           variant=variant, **named)


def _aimed_at(piece, exit_name):
    """The piece as a level whose finish is the end of one of its two ways."""
    level = piece.level(time_limit=600.0)
    target = piece.exits[exit_name].cell
    level.finish_cell = target
    level.features = [Finish(target) if isinstance(feature, Finish) else feature
                      for feature in level.features]
    return level


def _driven(level, seconds=30.0):
    """Play ``level`` with the autopilot; answer the seconds it took, or None.

    The pilot leans the board through the same rig a player's keys do, so what
    it manages is what the piece asks rather than what a path-follower could be
    dragged through.
    """
    game = MarbleGame(level)
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis)
    world, index = game.scene.world, game.marble.index
    played = 0.0
    for _ in range(int(seconds / STEP)):
        game.lean(*driver.lean(world.position[index],
                               world.linear_velocity[index]))
        state = game.advance(STEP)
        played += STEP
        if state != 'playing':
            break
    return played if game.state == WON else None


@pytest.fixture(scope='module')
def runs():
    """What the autopilot managed down each of the two ways, in seconds."""
    piece = _built()
    return {name: _driven(_aimed_at(piece, name)) for name in ('ok', 'long')}


# -- the two ways --------------------------------------------------------------

def test_a_split_offers_a_quick_way_and_a_long_one():
    piece = _built()
    assert set(piece.exits) == {'ok', 'long'}
    for name, port in piece.exits.items():
        assert port.cell in piece.cells, name


@pytest.mark.parametrize('variant', sorted(fragments.library()['split'].variants))
def test_both_ways_are_reachable_and_end_at_the_same_place(variant):
    """A fork whose two ways end in different places is two fragments.

    The same place means the same height and the same distance down the board:
    a story sends both exits to one chapter and the second to arrive is joined
    to where the first laid it.
    """
    piece = _built(variant)
    ahead = piece.entry.facing
    reached = {}
    for name, port in piece.exits.items():
        assert pieces.joined(piece.cells, piece.entry.cell, port.cell), name
        assert port.height == pytest.approx(piece.entry.height), name
        reached[name] = port.cell[0] * ahead[0] + port.cell[1] * ahead[1]
    assert reached['ok'] == reached['long'], \
        'the ways end %d cells apart down the board' % abs(reached['ok']
                                                           - reached['long'])


@pytest.mark.parametrize('variant', sorted(fragments.library()['split'].variants))
def test_the_quick_way_really_is_the_shorter_one(variant):
    piece = _built(variant)
    quick = pieces.distance(piece.cells, piece.entry.cell, piece.exits['ok'].cell)
    round_about = pieces.distance(piece.cells, piece.entry.cell,
                                  piece.exits['long'].cell)
    assert quick + 3 <= round_about, \
        '%s: the plank is %d cells and the way round %d' % (variant, quick,
                                                            round_about)


# -- what each way is made of --------------------------------------------------

def _plank(piece):
    """The cells of the piece that have no cell on either side of them."""
    across = piece.entry.across()
    return {cell for cell in piece.cells
            if (cell[0] + across[0], cell[1] + across[1]) not in piece.cells
            and (cell[0] - across[0], cell[1] - across[1]) not in piece.cells}


@pytest.mark.parametrize('variant', sorted(fragments.library()['split'].variants))
def test_the_quick_way_is_one_cell_across(variant):
    """Narrow is the price of short: the plank is what the choice is about."""
    piece = _built(variant)
    plank = _plank(piece)
    assert len(plank) >= 4, 'only %d cells of the piece stand on their own' \
        % len(plank)
    for cell in plank:
        assert pieces.joined(piece.cells, cell, piece.exits['ok'].cell), \
            'the single-cell run at %r is not on the way to the quick exit' % (cell,)


def test_the_plank_has_nothing_beside_it():
    """A bridge with rails is a corridor."""
    piece = _built()
    railed = {feature.cell for feature in piece.features
              if isinstance(feature, Wall)}
    walled = _plank(piece) & railed
    assert not walled, 'the plank is walled at %s' % sorted(walled)


def test_the_long_way_is_walled_the_whole_distance():
    """It is the kind one, and the kindness is the walls."""
    piece = _built()
    walls = [feature for feature in piece.features if isinstance(feature, Wall)]
    beside_the_long_way = [wall for wall in walls
                           if wall.cell[0] * piece.exits['long'].cell[0] > 0]
    assert len(beside_the_long_way) >= 8, \
        'only %d walls stand along the way round' % len(beside_the_long_way)


# -- and the choice is worth making --------------------------------------------

def test_the_plank_is_measurably_the_quicker_run(runs):
    """The rule: the short way is short enough to be worth its narrowness."""
    assert runs['ok'] is not None, 'the autopilot never got over the plank'
    assert runs['long'] is not None, 'the autopilot never got round the long way'
    assert runs['ok'] < runs['long'] - 2.0, \
        'the plank took %.2f s and the way round %.2f s' % (runs['ok'],
                                                            runs['long'])
