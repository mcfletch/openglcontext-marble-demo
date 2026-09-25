"""The sand pit: pick your way round it, or be thrown clear over it.

Sand exists as a mechanism and, until this fragment, no board could contain it.
What makes the pit a chapter rather than a patch of floor is that it offers two
answers and charges differently for each — the careful line along the edge, and
the ramp that throws you across.

The rule has to be shown both ways round: that the sand really does hold a
marble that rolls into it, and that the way past really is passable.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Ramp
from openglcontext_marble_demo.mechanisms.sand import Sand

CELL = 4.0


def _built(variant='plain', seed=3):
    return fragments.build('sand_pit', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _cross(piece, speed, offset=0.0, seconds=14.0):
    """Roll a marble in ``offset`` cells off the centre line; how long to cross.

    Answers the seconds taken to reach the exit row, or None for a marble still
    in the pit when the time was up — which is the sand doing its job rather
    than a failure to measure.
    """
    level = piece.level(time_limit=900.0, goal=False) \
        if 'goal' in piece.level.__code__.co_varnames else piece.level(time_limit=900.0)
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    x, z = level.cell_center(piece.entry.cell)
    world.position[index] = (x + offset * CELL, piece.entry.height + 0.6, z)
    world.linear_velocity[index] = (0.0, 0.0, speed)
    world.wake(index)
    target = level.cell_center(piece.exit.cell)[1]
    for step in range(int(seconds / (1 / 120.0))):
        game.advance(1 / 120.0)
        if float(world.position[index][2]) >= target - CELL * 0.5:
            return step / 120.0
    return None


# -- it is a fragment like any other ---------------------------------------------

def test_the_library_holds_it():
    assert 'sand_pit' in fragments.library()


def test_it_offers_variants():
    assert len(fragments.library()['sand_pit'].variants) >= 3


def test_it_says_what_it_asks():
    assert fragments.library()['sand_pit'].rule


def test_it_actually_contains_sand():
    """The point of the fragment: a mechanism no board could reach before."""
    assert any(isinstance(f, Sand) for f in _built().features)


def test_the_sand_covers_cells_the_piece_laid():
    piece = _built()
    for feature in piece.features:
        if isinstance(feature, Sand):
            assert set(feature.cells) <= set(piece.cells)


# -- the rule ---------------------------------------------------------------------

@pytest.mark.slow
def test_rolling_into_the_sand_costs_you():
    """Measured on the variant with no ramp, which is the only way to ask.

    Where there is a ramp the sand is what a player is thrown *over*, so a run
    down the middle never touches it; ``sheer`` takes the ramp away and leaves
    the pit to be crossed.
    """
    piece = _built('sheer')
    through = _cross(piece, 8.0, offset=0.0)
    assert through is None or through > 4.0, \
        'the sand let a marble straight through in %.2f s' % (through or 0.0)


@pytest.mark.slow
def test_the_ramp_is_the_quick_way_and_the_shoulder_is_the_safe_one():
    """The two answers, and what each costs.

    The launch is quicker -- that is why it is worth lining up for -- and the
    shoulder is there for a player who would rather not gamble on the arc.  Both
    get across, which is what stops a muffed launch being the end of a run.
    """
    piece = _built()
    launched = _cross(piece, 8.0, offset=0.0)
    shoulder = _cross(piece, 8.0, offset=2.0)
    assert launched is not None, 'the ramp did not get a marble across at all'
    assert shoulder is not None, 'the careful line did not get across at all'
    assert launched < shoulder, \
        'the ramp was not worth taking: launched %.2f s, shoulder %.2f s' % (
            launched, shoulder)


def test_the_ramp_does_not_cover_the_shoulder():
    """A ramp across the whole mouth launches a marble whatever it does, and the
    pit becomes scenery under it."""
    piece = _built()
    ramps = {f.cell for f in piece.features if isinstance(f, Ramp)}
    sand = {tuple(c) for f in piece.features if isinstance(f, Sand) for c in f.cells}
    assert ramps
    ramp_columns = {col for col, _ in ramps}
    sand_columns = {col for col, _ in sand}
    assert ramp_columns <= sand_columns, \
        'the ramp reaches lanes the sand does not: %r' % sorted(
            ramp_columns - sand_columns)


@pytest.mark.slow
def test_a_marble_in_the_sand_still_gets_out_of_the_far_end():
    """Churning is slow; being stuck is a board nobody finishes.

    Given long enough the lean carries a marble out at a walking pace, which is
    what makes the slow way a cost rather than a dead end.
    """
    piece = _built()
    assert _cross(piece, 8.0, offset=0.0, seconds=45.0) is not None


def test_the_hard_edges_are_where_the_sand_stops():
    """Nothing tapers: a cell is sand or it is floor, so clipping the corner is
    a decision with a consequence rather than a gradient."""
    piece = _built()
    sand = {cell for f in piece.features if isinstance(f, Sand) for cell in f.cells}
    assert sand
    assert set(piece.cells) - sand, 'the whole piece is sand: there is no way round'
