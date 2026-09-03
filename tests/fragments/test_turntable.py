"""The turntable: several ways off a hub, and the arrival that decides which.

A rotating arm meets a marble that has rolled into the hub and carries it
whichever way it is currently pointing; nothing else about the piece changes
between one attempt and the next, so the only thing that can make two attempts
come out differently is when, in the bar's turn, the marble arrived. That is
measured the way ``tests/fragments/test_gauntlet.py`` measures its own rotating
arms: starting the bar part way round its turn, which is the same question as a
marble reaching the hub part way through it.

The board is given the lean the library's rules were measured against -- see
:data:`~openglcontext_marble_demo.fragments.DESIGN_TILT`.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.fragments import turntable as piece_module
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE
from openglcontext_marble_demo.mechanisms.turntable import Turntable
from openglcontext_marble_demo.pieces import Port

STEP = 1 / 120.0

#: The phases a run is tried at, in seconds of the bar's turn -- spread across
#: (rather than aligned with) a full period, so no one of them is special.
PHASES = (0.05, 0.6, 1.25, 1.9, 2.75)

#: How long a run is given before it counts as never having left the hub.
PATIENCE = 35.0


def _entry():
    return Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('turntable', random.Random(seed), _entry(),
                           variant=variant, **named)


def _cell_of(position):
    return (int(round(position[0] / CELL_SIZE)), int(round(position[2] / CELL_SIZE)))


def _arrival(piece, phase, seconds=PATIENCE):
    """Play ``piece`` alone, the bar already ``phase`` seconds into its turn.

    Answers ``(exit_name, seconds)`` for the mouth the marble left by, or
    ``(None, seconds)`` if it was still on the hub when ``seconds`` ran out.

    Undriven -- no autopilot, no keys -- because the whole claim is about what
    the bar does with a marble the board's own lean brings to it.  The board is
    given that lean explicitly (:data:`~openglcontext_marble_demo.fragments.DESIGN_TILT`)
    rather than inheriting one: the shipped board is level, and a marble nothing
    carries never reaches the hub for the bar to meet.
    """
    level = piece.level(time_limit=600.0)
    game = MarbleGame(level, base_tilt=fragments.DESIGN_TILT)
    for animator in game.build.animators:
        animator.time = phase
    world, index = game.scene.world, game.marble.index
    exit_cells = {name: set(port.cells()) for name, port in piece.exits.items()}
    for step in range(int(seconds / STEP)):
        state = game.advance(STEP)
        here = _cell_of(world.position[index])
        for name, cells in exit_cells.items():
            if here in cells:
                return name, round(step * STEP, 3)
        if state != 'playing':
            # The 'ok' exit doubles as the piece's own finish, so a win here
            # is the same event the exit-cell check above would have caught
            # a frame or two later.
            return 'ok', round(step * STEP, 3)
    return None, seconds


def _report(found):
    return ', '.join('%.2fs in: %s' % (phase, ('%s at %.2fs' % (name, took))
                                       if name else 'still on the hub')
                     for phase, (name, took) in sorted(found.items()))


@pytest.fixture(scope='module')
def runs():
    """``{phase: (exit_name, seconds)}`` for the ``plain`` variant."""
    piece = _built('plain')
    return {phase: _arrival(piece, phase) for phase in PHASES}


# -- what the piece is made of --------------------------------------------------

def test_a_turntable_offers_more_than_one_named_exit():
    piece = _built('plain')
    assert 'ok' in piece.exits, sorted(piece.exits)
    assert len(piece.exits) >= 2, \
        'only %d way(s) off the disc: %s' % (len(piece.exits), sorted(piece.exits))


@pytest.mark.parametrize('variant', sorted(fragments.library()['turntable'].variants))
def test_every_exit_is_reachable_from_the_entry(variant):
    piece = _built(variant)
    entry = _entry()
    for name, port in piece.exits.items():
        assert pieces.joined(piece.cells, entry.cell, port.cell), \
            '%s: %r is not reachable from the entry' % (variant, name)


@pytest.mark.parametrize('variant', sorted(fragments.library()['turntable'].variants))
def test_a_turntable_mechanism_sits_at_the_hub(variant):
    piece = _built(variant)
    discs = [f for f in piece.features if isinstance(f, Turntable)]
    assert len(discs) == 1, '%s: %d turntables in one piece' % (variant, len(discs))


# -- the rule: when you arrive decides where you leave --------------------------

def test_arriving_at_different_phases_leaves_by_different_exits(runs):
    taken = {name for name, _ in runs.values() if name is not None}
    assert len(taken) >= 2, \
        'every phase of the bar sent the marble out the same way: %s' % _report(runs)


def test_the_marble_never_leaves_by_a_mouth_the_bar_is_not_offering(runs):
    piece = _built('plain')
    for phase, (name, _took) in runs.items():
        assert name in piece.exits, \
            '%.2fs in: left by %r, which is not one of %s' \
            % (phase, name, sorted(piece.exits))


def test_reversing_the_turn_mirrors_left_and_right():
    """``reverse`` is ``plain`` with the sign of ``rpm`` flipped -- see the
    library table entry, "which way it turns" is the effect variants differ by.
    """
    plain = _built('plain')
    reverse = _built('reverse')
    mirror = {'left': 'right', 'right': 'left', 'ok': 'ok'}
    for phase in PHASES:
        forward_name, forward_took = _arrival(plain, phase)
        backward_name, backward_took = _arrival(reverse, phase)
        assert forward_name is not None and backward_name is not None, \
            '%.2fs in: plain=%r reverse=%r' % (phase, forward_name, backward_name)
        assert mirror[forward_name] == backward_name, \
            '%.2fs in: plain left by %r (%.2fs), reverse left by %r (%.2fs)' \
            % (phase, forward_name, forward_took, backward_name, backward_took)


# -- it is always passable, and bounded -----------------------------------------

def test_a_marble_always_leaves_the_disc_within_a_bounded_time(runs):
    for phase, (name, took) in runs.items():
        assert name is not None, \
            '%.2fs in: still on the hub after %.1fs' % (phase, PATIENCE)
        assert took < PATIENCE, \
            '%.2fs in: took %.2fs to leave by %r, against a %.1fs patience' \
            % (phase, took, name, PATIENCE)


@pytest.mark.parametrize('variant', sorted(fragments.library()['turntable'].variants))
def test_every_variant_lets_a_marble_go_at_every_phase_sampled(variant):
    piece = _built(variant)
    found = {phase: _arrival(piece, phase) for phase in PHASES}
    stuck = {phase: took for phase, (name, took) in found.items() if name is None}
    assert not stuck, '%s: still on the hub at %s' % (variant, sorted(stuck))


# -- the four variants, and the three axes they differ on -----------------------

def test_the_variants_differ_in_material_layout_and_effect():
    settings = piece_module._VARIANTS
    themes = {name: value['theme'] for name, value in settings.items()}
    assert len(set(themes.values())) >= 3, \
        'only %d distinct materials among the variants: %s' % (
            len(set(themes.values())), themes)

    exit_counts = {name: len(value['exits']) for name, value in settings.items()}
    assert len(set(exit_counts.values())) >= 2, \
        'every variant offers the same number of ways off: %s' % exit_counts

    rpms = {name: value['rpm'] for name, value in settings.items()}
    assert len(set(abs(rate) for rate in rpms.values())) >= 2, \
        'every variant turns at the same speed: %s' % rpms
    assert any(rate < 0 for rate in rpms.values()) and \
        any(rate > 0 for rate in rpms.values()), \
        'no variant turns the other way: %s' % rpms
