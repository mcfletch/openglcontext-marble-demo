"""The broken slope, and the walls that keep what it knocks about on the board.

A rockfall's whole purpose is to knock a marble sideways, so the one thing the
piece around it must not be is open at the sides. That is the rule here, and it
is measured as geometry rather than by driving: a wall either stands between a
cell and the void beside it or it does not, and a test that drove the slope would
be measuring the pilot as much as the piece.

Every measurement is taken with the piece entered facing +Z, so a row of the grid
is a step down the board and a column is across it.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.level import Wall
from openglcontext_marble_demo.mechanisms.rockfall import Rockfall

VARIANTS = sorted(fragments.library()['scree'].variants)

#: Which way a wall on a cell faces, by the step to the cell beyond it.
SIDE = {(0, -1): 'N', (0, 1): 'S', (1, 0): 'E', (-1, 0): 'W'}


def _built(variant=None, seed=3):
    return fragments.build('scree', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _railed(piece):
    return {(wall.cell, wall.side) for wall in piece.features
            if isinstance(wall, Wall)}


def _descending_rows(piece):
    """The rows the slope falls over: below the entry and above the foot."""
    heights = set(piece.cells.values())
    floor = min(heights)
    return sorted({row for (_col, row), height in piece.cells.items()
                   if floor + 1e-9 < height < piece.entry.height - 1e-9})


def _open_sides(piece, rows):
    """The outermost cells of ``rows`` with void beside them and no wall."""
    railed = _railed(piece)
    found = []
    for row in rows:
        columns = sorted(col for (col, at) in piece.cells if at == row)
        if not columns:
            continue
        for cell, step in (((min(columns), row), (-1, 0)),
                           ((max(columns), row), (1, 0))):
            beside = (cell[0] + step[0], cell[1] + step[1])
            if beside not in piece.cells and (cell, SIDE[step]) not in railed:
                found.append(cell)
    return found


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_the_slope_is_walled_down_both_sides(variant):
    """`rails` skips a cell that is not on the board, so rails asked for at one
    width fall outside a run laid at another and leave it open.  Asked for at the
    landing's width alone, every descending row of every variant was open on both
    sides — eight of them on `plain`, fourteen on `long`."""
    piece = _built(variant)
    rows = _descending_rows(piece)
    assert rows, '%s: nothing descends' % variant
    open_sides = _open_sides(piece, rows)
    assert not open_sides, \
        '%s: %d of %d descending rows are open at the side, at %r' \
        % (variant, len(open_sides), len(rows), open_sides[:4])


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_landing_is_walled_too_and_it_is_wider_than_the_slope(variant):
    """Wide because the point is arriving somewhere you did not aim at, and
    walled for the same reason the slope is."""
    piece = _built(variant)
    floor = min(piece.cells.values())
    rows = sorted({row for (_col, row), height in piece.cells.items()
                   if abs(height - floor) < 1e-9})
    widest = max(len([col for (col, at) in piece.cells if at == row])
                 for row in rows)
    slope = max(len([col for (col, at) in piece.cells if at == row])
                for row in _descending_rows(piece))
    assert widest > slope, \
        '%s: the landing is %d cells across and the slope %d' % (variant, widest, slope)
    # The last row is the way out, so it is open at its end and not at its sides.
    assert not _open_sides(piece, rows), variant


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_rock_lies_on_the_slope_and_the_piece_runs_through(variant):
    piece = _built(variant)
    assert [f for f in piece.features if isinstance(f, Rockfall)], variant
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    assert piece.exit.height < piece.entry.height - 1.0, \
        '%s only falls %.2f m' % (variant, piece.entry.height - piece.exit.height)
