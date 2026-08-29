"""The fall, and the window of speeds that puts a marble on the landing.

The rule is about the speed the marble leaves the lip at, and that makes it a
rule a test can measure exactly rather than infer: fly the marble, and record
how far past the lip it is at the moment it comes down to the landing's height.
Short of the landing's near edge is the gap; past its far edge is the void; in
between is the pad. The number is in every assertion here, because "it failed"
is no use to anybody tuning the piece.

Every measurement is taken with the piece entered facing +Z, so a row of the
grid is a step down the board and a cell is
:data:`~openglcontext_marble_demo.level.CELL_SIZE` metres of it.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish, Wall

STEP = 1 / 120.0

#: How far above the landing the marble's centre is when it is resting on it,
#: plus a little: the height at which it counts as having come down.
TOUCHED = 0.7


def _entry():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('drop', random.Random(seed), _entry(),
                           variant=variant, **named)


def _lane(piece, height):
    """The rows of the lane's centre column that sit at ``height``."""
    return [row for (col, row), at in piece.cells.items()
            if col == piece.entry.cell[0] and abs(at - height) < 1e-6]


def _landing(piece):
    """``(lip_row, near, far)`` — where the landing is, in metres past the lip.

    Measured from the lip cell's far edge, which is where the marble leaves the
    ground, so the numbers are the flight rather than the geometry.
    """
    lip = max(_lane(piece, piece.entry.height))
    pad = _lane(piece, piece.exit.height)
    return lip, (min(pad) - lip) * CELL_SIZE - CELL_SIZE, (max(pad) - lip) * CELL_SIZE


def _came_down(piece, speed, seconds=6.0):
    """How far past the lip the marble first reaches the landing's height.

    ``None`` if it never does within ``seconds``.  The finish pad is taken out
    first: a marble that flies clean over the landing passes through the trigger
    that would end the run before it has come down anywhere.
    """
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    lip_edge = _landing(piece)[0] * CELL_SIZE + CELL_SIZE / 2.0
    plane = piece.exit.height + TOUCHED
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        at = world.position[index]
        if at[1] <= plane:
            return float(at[2]) - lip_edge
        if game.controller.fall_count:
            break
    return None


def _say(down):
    """Where the marble came down, for an assertion message."""
    return ('%.2f m past the lip' % down if down is not None
            else 'nowhere the run was long enough to see')


def _where(piece, speed):
    """``(distance, verdict)`` for one run at ``speed``."""
    _, near, far = _landing(piece)
    down = _came_down(piece, speed)
    if down is None:
        return down, 'never came down'
    return down, 'short' if down < near else ('on the landing' if down <= far
                                              else 'past the far edge')


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', sorted(fragments.library()['drop'].variants))
def test_a_drop_puts_a_gap_between_the_lip_and_the_landing(variant):
    piece = _built(variant)
    lip, near, far = _landing(piece)
    assert near >= CELL_SIZE - 1e-9, '%s leaves only %.1f m of gap' % (variant, near)
    assert far > near, variant
    assert piece.exit.height < piece.entry.height - 1.0, variant
    for row in range(lip + 1, lip + int(near / CELL_SIZE) + 1):
        assert (piece.entry.cell[0], row) not in piece.cells, \
            '%s has floor in the gap at row %d' % (variant, row)


@pytest.mark.parametrize('variant', sorted(fragments.library()['drop'].variants))
def test_a_ledge_steps_down_around_the_gap(variant):
    """A marble that will not commit still has to be able to get to the bottom,
    and every step of the way down has to be one a marble can roll."""
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    for (col, row), height in piece.cells.items():
        for step in ((1, 0), (0, 1)):
            beside = (col + step[0], row + step[1])
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                    '%s steps %.2f at %r' % (variant, piece.cells[beside] - height,
                                             (col, row))


def test_the_landing_is_walled_along_its_side_and_open_at_both_ends():
    """Open at the near end because that is where a marble arrives from, and at
    the far end because that is the way on."""
    piece = _built()
    _, _, far = _landing(piece)
    pad = set(_lane(piece, piece.exit.height))
    walls = [w for w in piece.features
             if isinstance(w, Wall) and w.cell[1] in pad]
    assert walls, 'the landing has no side to it at all'
    assert not [w for w in walls if w.side in ('N', 'S')], \
        'the landing is walled across an end'


# -- and the rule it imposes ---------------------------------------------------

def test_leaving_the_lip_too_slowly_lands_short_of_the_landing():
    piece = _built()
    _, near, _ = _landing(piece)
    down, verdict = _where(piece, 4.0)
    assert verdict == 'short', \
        'at 4 m/s the marble came down %s, and the landing starts at %.1f m: ' \
        '%s' % (_say(down), near, verdict)


def test_leaving_the_lip_at_the_right_speed_puts_the_marble_on_the_landing():
    piece = _built()
    _, near, far = _landing(piece)
    for speed in (7.0, 8.0, 10.0, 12.0):
        down, verdict = _where(piece, speed)
        assert verdict == 'on the landing', \
            'at %.0f m/s the marble came down %s, and the landing runs ' \
            '%.1f..%.1f m: %s' % (speed, _say(down), near, far, verdict)


def test_leaving_the_lip_too_fast_carries_past_the_landing():
    piece = _built()
    _, _, far = _landing(piece)
    down, verdict = _where(piece, 14.0)
    assert verdict == 'past the far edge', \
        'at 14 m/s the marble came down %s, and the landing ends at %.1f m: ' \
        '%s' % (_say(down), far, verdict)


@pytest.mark.parametrize('variant', sorted(fragments.library()['drop'].variants))
def test_every_variant_has_a_speed_that_lands_and_a_speed_that_does_not(variant):
    """A drop nothing can clear is a wall; one nothing can miss is a floor."""
    piece = _built(variant)
    _, near, far = _landing(piece)
    seen = {speed: _where(piece, speed) for speed in (4.0, 10.0, 18.0)}
    landed = [speed for speed, (_, verdict) in seen.items()
              if verdict == 'on the landing']
    missed = [speed for speed, (_, verdict) in seen.items()
              if verdict != 'on the landing']
    report = ', '.join('%.0f m/s: %s (%s)' % (speed, _say(down), verdict)
                       for speed, (down, verdict) in sorted(seen.items()))
    assert landed, '%s: nothing reached the landing at %.1f..%.1f m — %s' \
        % (variant, near, far, report)
    assert missed, '%s: every speed landed on %.1f..%.1f m — %s' \
        % (variant, near, far, report)
