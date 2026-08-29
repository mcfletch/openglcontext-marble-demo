"""Tests for the seesaw fragment: a mouth, a tilting plank, a landing beyond it.

Every measurement is taken with the piece entered facing +Z, so a row of the
grid is a step down the board and a cell is
:data:`~openglcontext_marble_demo.level.CELL_SIZE` metres of it. The rule is
compared against a bare corridor of the same shape, built from the same
:mod:`~openglcontext_marble_demo.pieces` primitives with no plank in it, which
is what turns "it took longer" into "the plank is what took longer."
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.fragments.seesaw import VARIANTS
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails

STEP = 1 / 120.0

#: How close to the exit row counts as having reached it.
_ARRIVED = 0.3


def _entry():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('seesaw', random.Random(seed), _entry(), variant=variant,
                           **named)


def _bare(length, width):
    """A flat corridor the same shape as the piece, with no plank in it."""
    entry = _entry()
    cells: dict = {}
    mouth = _lay(cells, entry, 2, width=width)
    start = mouth.ahead(1)
    end = _lay(cells, start, length, width=width)
    landing = _lay(cells, end.ahead(1), 2, width=width)
    walls = _rails(cells, entry, length + 4, width=width)
    return Piece(name='bare', cells=cells, entry=entry,
                exits={'ok': Port(cell=landing.cell, facing=entry.facing,
                                  height=entry.height, width=LANE)},
                features=walls, theme='stone')


def _time_to_exit(piece, speed, cap=16.0):
    """Seconds to first reach the exit row, or ``None`` if it fell first."""
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    exit_z = piece.exit.cell[1] * CELL_SIZE
    for step in range(int(cap / STEP)):
        game.advance(STEP)
        if world.position[index][2] >= exit_z - _ARRIVED:
            return step * STEP
        if game.controller.fall_count:
            return None
    return None


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_a_variant_builds_a_passable_piece(variant):
    piece = _built(variant)
    assert 'ok' in piece.exits, variant
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_every_variant_is_crossed_at_a_crawl_and_at_speed(variant):
    """Neither a crawl nor a charge is lost off the plank: both eventually
    reach the exit, which is what "always passable" means for a piece that
    has nothing pulling it through but the board's own lean."""
    piece = _built(variant)
    slow = _time_to_exit(piece, 1.5)
    fast = _time_to_exit(piece, 12.0)
    assert slow is not None, '%s: a 1.5 m/s entry never reached the exit' % variant
    assert fast is not None, '%s: a 12 m/s entry never reached the exit' % variant


# -- and the rule it imposes ---------------------------------------------------

def test_dawdling_costs_seconds_a_bare_corridor_would_not():
    piece = _built('plain')
    bare = _bare(length=VARIANTS['plain']['length'], width=VARIANTS['plain']['width'])

    slow_seesaw, slow_bare = _time_to_exit(piece, 1.5), _time_to_exit(bare, 1.5)
    fast_seesaw, fast_bare = _time_to_exit(piece, 12.0), _time_to_exit(bare, 12.0)
    assert None not in (slow_seesaw, slow_bare, fast_seesaw, fast_bare)
    slow_cost = slow_seesaw - slow_bare
    fast_cost = fast_seesaw - fast_bare

    assert slow_cost > 1.0, \
        'a 1.5 m/s crossing cost only %.2fs over a bare corridor (seesaw %.2fs, ' \
        'bare %.2fs)' % (slow_cost, slow_seesaw, slow_bare)
    assert fast_cost < 0.3, \
        'a 12 m/s crossing cost %.2fs over a bare corridor, not the near-nothing a ' \
        'fast crossing should (seesaw %.2fs, bare %.2fs)' % (fast_cost, fast_seesaw, fast_bare)
    assert slow_cost > fast_cost * 3, \
        'dawdling (%.2fs over bare) does not cost materially more than charging ' \
        '(%.2fs over bare)' % (slow_cost, fast_cost)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_every_variant_bites_harder_on_a_slow_crossing(variant):
    """The rule holds for every material/layout combination, not just the
    default: whatever the plank's own length, width or response, dawdling
    costs more over the same bare shape than charging does."""
    settings = VARIANTS[variant]
    piece = _built(variant)
    bare = _bare(length=settings['length'], width=settings['width'])

    slow_seesaw, slow_bare = _time_to_exit(piece, 1.5), _time_to_exit(bare, 1.5)
    fast_seesaw, fast_bare = _time_to_exit(piece, 12.0), _time_to_exit(bare, 12.0)
    assert None not in (slow_seesaw, slow_bare, fast_seesaw, fast_bare), variant
    slow_cost = slow_seesaw - slow_bare
    fast_cost = fast_seesaw - fast_bare

    assert slow_cost > fast_cost + 0.5, \
        '%s: dawdling cost %.2fs over bare, charging cost %.2fs over bare -- not ' \
        'enough of a gap between them' % (variant, slow_cost, fast_cost)
