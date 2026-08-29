"""The gap, the shaft over it, and the two ways down beside it.

This chapter is answered by speed alone: the line across it is straight, so
there is nothing to steer at, and the only question is how fast the marble
was still going when the floor ran out. Three things are measured -- that the
gap really is too wide for the crossing to happen unaided, that carrying speed
into it gets a marble across, and that easing off sinks one in.
"""
import dataclasses
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.controller import ACTIVE
from openglcontext_marble_demo.fragments.updraft import LANDING, LANDING_DROP, RUN_UP
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE
from openglcontext_marble_demo.mechanisms.updraft import Updraft

STEP = 1 / 120.0

#: Speeds used throughout: comfortably either side of the threshold measured
#: by ``test_carrying_speed_across_the_gap_reaches_the_far_side`` and
#: ``test_easing_off_before_the_gap_sinks_you_into_it``.
FAST = 14.0
SLOW = 5.0


def _entry(width=3):
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=width)


def _built(variant='plain', seed=3, **named):
    return fragments.build('updraft', random.Random(seed), _entry(), variant=variant,
                           **named)


def _landing_start_z(piece):
    """World Z of the near edge of the landing -- the far side of the gap."""
    return (piece.exits['ok'].cell[1] - LANDING + 1) * CELL_SIZE - CELL_SIZE / 2.0


def _cross(piece, speed, seconds=10.0, dt=STEP, level=None):
    """Roll a marble onto the piece at ``speed`` and see what becomes of it.

    Answers ``(outcome, seconds, metres)``: ``outcome`` is ``'crossed'`` if it
    reaches the landing still active, ``'sunk'`` if the controller's own fall
    rule takes it first, or ``'timeout'`` if neither happens in time.
    ``seconds`` is how long that took and ``metres`` how far along Z the
    marble had got, so a failing assertion can say both.
    """
    level = piece.level(time_limit=600.0) if level is None else level
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    landing_z = _landing_start_z(piece)
    for step in range(int(seconds / dt)):
        game.advance(dt)
        pos = world.position[index]
        if game.controller.state != ACTIVE:
            return 'sunk', step * dt, float(pos[2])
        if float(pos[2]) >= landing_z:
            return 'crossed', step * dt, float(pos[2])
    return 'timeout', seconds, float(world.position[index][2])


def _without_the_draft(piece):
    """The same piece, with its shaft's lift switched off.

    What proves the gap is too wide to cross unaided: the same run, the same
    speed, the only thing removed is the mechanism.
    """
    level = piece.level(time_limit=600.0)
    level.features = [dataclasses.replace(f, strength=0.0) if isinstance(f, Updraft)
                      else f for f in level.features]
    return level


# -- the shape --------------------------------------------------------------

@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_there_is_no_floor_in_the_gap(variant):
    """The span between the lip and the landing carries no cells at all along
    the main lane -- a marble over it is off the track, exactly as it would be
    over any other fall too wide for it."""
    piece = _built(variant)
    lip_row = piece.entry.cell[1] + RUN_UP - 1
    landing_row = piece.exits['ok'].cell[1] - LANDING + 1
    for row in range(lip_row + 1, landing_row):
        assert (piece.entry.cell[0], row) not in piece.cells, \
            '%s has floor at the centre of the gap, row %d' % (variant, row)
    assert landing_row > lip_row + 1, '%s: no gap between lip and landing' % variant


@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_the_landing_sits_only_a_small_step_below_the_lip(variant):
    """The question this piece asks is the gap's width, not one more slope --
    the far side is close to the entry's own height."""
    piece = _built(variant)
    assert piece.entry.height - piece.exits['ok'].height == pytest.approx(LANDING_DROP)


@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_an_updraft_covers_the_gap_and_nothing_else(variant):
    piece = _built(variant)
    shafts = [f for f in piece.features if isinstance(f, Updraft)]
    assert len(shafts) == 1, variant
    shaft = shafts[0]
    lip_row = piece.entry.cell[1] + RUN_UP - 1
    landing_row = piece.exits['ok'].cell[1] - LANDING + 1
    gap_rows = set(range(lip_row + 1, landing_row))
    assert {row for _col, row in shaft.cells} == gap_rows, variant
    assert not (set(shaft.cells) & set(piece.cells)), \
        '%s: the shaft claims a cell the piece also built a floor for' % variant


# -- both exits are real, and reachable --------------------------------------

@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_every_variant_has_an_ok_exit_and_a_sunk_exit(variant):
    piece = _built(variant)
    assert 'ok' in piece.exits and 'sunk' in piece.exits, variant
    assert piece.exits['ok'].cell in piece.cells, variant
    assert piece.exits['sunk'].cell in piece.cells, variant
    assert piece.exits['sunk'].height < piece.exits['ok'].height, (
        '%s: "sunk" (%.2f m) should be well below "ok" (%.2f m)'
        % (variant, piece.exits['sunk'].height, piece.exits['ok'].height))


@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_the_bridge_and_the_stair_are_both_walkable_from_the_entry(variant):
    """A player who does not trust the jump can walk to either door: the quick
    bridge beside the gap, or the stair all the way down to the shaft's own
    floor. Neither route depends on the flight across the middle."""
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exits['ok'].cell), variant
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exits['sunk'].cell), variant


@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_every_variant_holds_the_slope_budget(variant):
    """Bridge, stair and underpass all respect the step a marble can climb --
    this fragment's own check, ahead of ``tests/test_registry.py``'s generic one."""
    piece = _built(variant)
    for (col, row), height in piece.cells.items():
        for step in ((1, 0), (0, 1)):
            beside = (col + step[0], row + step[1])
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                    '%s steps %.2f at %r' % (variant, piece.cells[beside] - height,
                                             (col, row))


# -- the rule: speed is what crosses the gap ---------------------------------

def test_the_gap_is_too_wide_to_cross_without_the_draft():
    """Switch the shaft's lift off and even the fast run sinks: the gap is
    genuinely too wide for a jump on its own, and the mechanism is what
    changes that."""
    piece = _built('plain')
    outcome, seconds, metres = _cross(piece, FAST, level=_without_the_draft(piece))
    assert outcome == 'sunk', (
        'at %.0f m/s with the draft switched off the marble %s after %.2f s at '
        '%.1f m along -- the gap should be uncrossable without the shaft'
        % (FAST, outcome, seconds, metres))


def test_carrying_speed_across_the_gap_reaches_the_far_side():
    piece = _built('plain')
    outcome, seconds, metres = _cross(piece, FAST)
    assert outcome == 'crossed', (
        'at %.0f m/s the marble %s after %.2f s, %.1f m along a gap %.1f m wide'
        % (FAST, outcome, seconds, metres, _gap_width(piece)))


def test_easing_off_before_the_gap_sinks_you_into_it():
    piece = _built('plain')
    outcome, seconds, metres = _cross(piece, SLOW)
    assert outcome == 'sunk', (
        'at %.0f m/s the marble %s after %.2f s, only %.1f m along a gap %.1f m '
        'wide -- coming in slowly should sink it before it reaches the far side'
        % (SLOW, outcome, seconds, metres, _gap_width(piece)))


def _gap_width(piece):
    lip_row = piece.entry.cell[1] + RUN_UP - 1
    landing_row = piece.exits['ok'].cell[1] - LANDING + 1
    return (landing_row - lip_row - 1) * CELL_SIZE


@pytest.mark.parametrize('variant', sorted(fragments.library()['updraft'].variants))
def test_every_variant_can_be_crossed_at_a_generous_speed(variant):
    """Whatever the layout and the draft's own strength, some speed gets a
    marble across -- a gap nothing can ever cross is a wall wearing a shaft's
    clothes."""
    piece = _built(variant)
    outcome, seconds, metres = _cross(piece, 18.0)
    assert outcome == 'crossed', (
        '%s: at 18 m/s the marble %s after %.2f s, %.1f m along' %
        (variant, outcome, seconds, metres))
