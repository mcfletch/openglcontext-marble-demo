"""The sweeping arms, and what arriving at the wrong moment costs.

A rotating arm is a revolving door rather than a gate: it never shuts a way for
good, so the piece cannot be held to "the wrong moment does not get through".
What it can be held to is the thing a clock cares about — that a run made at the
right moment costs almost nothing over an empty lane of the same length, and one
made at the wrong moment costs several times that.

The moment is set by starting the arms part way round, which is the same
question as a marble reaching the mouth part way through their turn, and the
runs are driven by the autopilot so what is measured is a board being played.
"""
import math
import random

import pytest

from openglcontext_marble_demo import fragments, pilot
from openglcontext_marble_demo.fragments import gauntlet as piece_module
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, RotatingArm
from openglcontext_marble_demo.pieces import Port

STEP = 1 / 120.0

#: The arm phases a run is tried at, in seconds of turn.
PHASES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
#: How long a run is given before it counts as having been turned back.
PATIENCE = 20.0


def _entry():
    return Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('gauntlet', random.Random(seed), _entry(),
                           variant=variant, **named)


def _crossing(level, phase=0.0, seconds=PATIENCE):
    """Play ``level`` with the arms already ``phase`` seconds into their turn.

    Answers the seconds the run took, or ``None`` if it was still going when
    :data:`PATIENCE` ran out.
    """
    game = MarbleGame(level)
    for animator in game.build.animators:
        animator.time = phase
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
    """``(clear_lane, {phase: seconds})`` for one variant of the piece."""
    piece = _built('brisk')
    clear = piece.level(time_limit=600.0)
    clear.features = [f for f in clear.features
                      if not isinstance(f, RotatingArm)]
    level = piece.level(time_limit=600.0)
    return (_crossing(clear),
            {phase: _crossing(level, phase) for phase in PHASES})


def _report(found):
    return ', '.join('%.1f s in: %s' % (phase,
                                        '%.2f s' % took if took else 'still going')
                     for phase, took in sorted(found.items()))


# -- how a row is put together -------------------------------------------------

@pytest.mark.parametrize('variant',
                         sorted(fragments.library()['gauntlet'].variants))
def test_a_row_of_arms_spans_the_whole_lane(variant):
    """One arm per cell of the lane, so a row is a door rather than a post."""
    piece = _built(variant)
    arms = [f for f in piece.features if isinstance(f, RotatingArm)]
    assert arms, variant
    rows: dict = {}
    for arm in arms:
        rows.setdefault(arm.cell[1], []).append(arm)
    for row, abreast in rows.items():
        assert len(abreast) == piece.entry.width, \
            '%s: row %d has %d arms across a lane %d cells wide' \
            % (variant, row, len(abreast), piece.entry.width)
        assert len({arm.rpm for arm in abreast}) == 1, \
            '%s: row %d is not one row but several' % (variant, row)


def _between_the_bars(variant):
    """Metres between the tips of two neighbouring bars of a row."""
    piece = _built(variant)
    return CELL_SIZE - next(f.length for f in piece.features
                            if isinstance(f, RotatingArm))


@pytest.mark.parametrize('variant', ('plain', 'foundry', 'brisk'))
def test_a_row_of_arms_closes_the_lane_when_it_lies_across_it(variant):
    """A door rather than a set of posts: with a marble a metre across, the
    bars have to leave less than that between their tips."""
    between = _between_the_bars(variant)
    assert between < 1.0, \
        '%s leaves %.2f m between the tips of neighbouring bars' % (variant,
                                                                    between)


def test_the_sparse_row_leaves_a_way_between_the_bars():
    """The variant a player can steer through whatever the timing."""
    between = _between_the_bars('sparse')
    assert between > 1.0, \
        'sparse leaves only %.2f m between the tips of neighbouring bars' % between


@pytest.mark.parametrize('variant',
                         sorted(fragments.library()['gauntlet'].variants))
def test_every_row_turns_fast_enough_to_be_worth_timing(variant):
    settings = piece_module._VARIANTS[variant]
    for rpm in piece_module.rates(settings['rows'], settings['pace'], (0, 1)):
        assert rpm * math.pi / 30.0 >= piece_module.MIN_RATE - 1e-9, \
            '%s has a row at %.1f rpm' % (variant, rpm)


def test_a_row_lies_along_the_lane_when_a_marble_at_the_design_pace_reaches_it():
    """Which is the phasing: the rate is chosen for the moment of arrival."""
    pace = piece_module._VARIANTS['plain']['pace']
    rows = piece_module._VARIANTS['plain']['rows']
    for facing, open_at in (((0, 1), math.pi / 2.0), ((1, 0), 0.0)):
        for row, rpm in enumerate(piece_module.rates(rows, pace, facing)):
            arrive = ((piece_module.LEAD + row * piece_module.SPACING)
                      * CELL_SIZE / pace)
            turned = (rpm * math.pi / 30.0) * arrive
            # A bar is symmetric, so it lies along the lane every half turn.
            over = (turned - open_at) % math.pi
            assert min(over, math.pi - over) == pytest.approx(0.0, abs=1e-6), \
                'facing %r, row %d is %.3f rad round when the marble arrives' \
                % (facing, row, turned)


# -- and the rule it imposes ---------------------------------------------------

def test_a_run_made_at_the_right_moment_gets_through(runs):
    clear, found = runs
    made = [took for took in found.values() if took is not None]
    assert made, 'no phase of the arms let a marble by at all: %s' % _report(found)
    assert min(made) <= clear * 1.3, \
        'the best of the phases took %.2f s against %.2f s down the same lane ' \
        'with the arms taken out (%s)' % (min(made), clear, _report(found))


def test_a_run_made_at_the_wrong_moment_pays_for_it(runs):
    """A rule is only a rule if it can be failed, and here failing costs time."""
    clear, found = runs
    made = [took for took in found.values() if took is not None]
    worst = max([took for took in found.values() if took is not None]
                + ([PATIENCE] if None in found.values() else []))
    assert worst >= min(made) * 2.0, \
        'the arms cost between %.2f s and %.2f s against %.2f s down the clear ' \
        'lane, which is not a piece anybody has to time (%s)' \
        % (min(made), worst, clear, _report(found))
