"""The riding platforms, and what waiting for one costs -- or does not.

An elevator is a revolving door that runs vertically: a marble that boards too
early or too late is not turned away, it simply cannot climb the gap yet and
sits at the mouth until the platform comes home. That is measured the same
way ``gauntlet`` measures its arms -- entering at several phases of the
platform's cycle and comparing what each one costs -- except here the two
things held to different standards are the two exits: the flat lane costs
about the same whatever the phase, and the lane that must be ridden does not.

The other half of the piece's claim is that neither lane can lose a marble.
Every run below is also checked against the controller's own count of marbles
destroyed, which must stay at zero: a mistimed ride costs seconds, never the
marble.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.level import Elevator, Finish
from openglcontext_marble_demo.pieces import Port

STEP = 1 / 120.0

#: The platform's phases a run is tried at, in seconds of travel, spanning a
#: full cycle of the ``foundry`` variant's four-second period.
PHASES = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6)
#: How long a run is given before it counts as having been turned back.
PATIENCE = 30.0


def _entry():
    return Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('lift', random.Random(seed), _entry(), variant=variant,
                           **named)


def _aimed_at(piece, exit_name, time_limit=600.0):
    """The piece as a level whose finish is one particular exit."""
    level = piece.level(time_limit=time_limit)
    target = piece.exits[exit_name].cell
    level.finish_cell = target
    level.features = [Finish(target) if isinstance(feature, Finish) else feature
                      for feature in level.features]
    return level


def _phased(game, phase):
    """Put every platform in ``game`` ``phase`` seconds into its travel.

    Setting :attr:`~omi_physics.kinematic.KinematicAnimator.time` alone only
    changes what the *next* frame asks the platform to move towards -- the
    body itself is still sitting wherever it was built, at the bottom of its
    travel. Left alone, the first :meth:`~KinematicAnimator.update` reads that
    whole gap in one frame and turns it into a velocity large enough to launch
    a marble resting on top, which is a test artifact rather than the piece:
    a platform that has been running since the level loaded is already where
    its own travel puts it. So the body is put there directly, the same place
    :meth:`~KinematicAnimator.update` would have carried it to over ``phase``
    seconds of real running.
    """
    for animator in game.build.animators:
        animator.time = phase
        pose = animator.pose_fn(phase)
        if len(pose) == 2 and np.ndim(pose[0]) == 1:
            position, orientation = pose
        else:
            position, orientation = pose, (0.0, 0.0, 0.0, 1.0)
        game.scene.world.position[animator.index] = position
        game.scene.world.orientation[animator.index] = orientation


def _crossing(level, phase=0.0, seconds=PATIENCE):
    """Play ``level`` with a platform ``phase`` seconds into its travel.

    Answers ``(seconds taken or None, marbles the controller destroyed)``.
    """
    game = MarbleGame(level)
    _phased(game, phase)
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
    took = played if game.state == WON else None
    return took, game.controller.loss_count


def _report(found):
    return ', '.join('%.1f s in: %s' % (phase,
                                        '%.2f s' % took if took else 'still going')
                     for phase, took in sorted(found.items()))


@pytest.fixture(scope='module')
def runs():
    """``{'early': {phase: seconds}, 'ok': {phase: (seconds, losses)}}`` for
    the ``foundry`` variant, whose four-second period is long enough that the
    phases in :data:`PHASES` span the whole of it."""
    piece = _built('foundry')
    found = {}
    for name in ('early', 'ok'):
        level = _aimed_at(piece, name)
        found[name] = {phase: _crossing(level, phase) for phase in PHASES}
    return found


# -- what every variant is built from -------------------------------------------

def test_a_lift_offers_a_bail_out_and_a_way_worth_riding_for():
    piece = _built('plain')
    assert set(piece.exits) == {'early', 'ok'}


def test_the_tower_variant_offers_a_stop_on_the_way_too():
    """The layout axis: more landings, not just a different one."""
    piece = _built('tower')
    assert set(piece.exits) == {'early', 'mid', 'ok'}


@pytest.mark.parametrize('variant', sorted(fragments.library()['lift'].variants))
def test_every_exit_is_reachable_and_the_ridden_ones_are_higher(variant):
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exits['ok'].cell)
    for name, port in piece.exits.items():
        assert pieces.joined(piece.cells, piece.entry.cell, port.cell), name
        if name != 'early':
            assert port.height > piece.exits['early'].height, \
                '%s: %s exit is not above the bail-out' % (variant, name)


def test_the_tower_variant_s_stop_is_lower_than_its_ridden_end():
    piece = _built('tower')
    assert piece.exits['mid'].height < piece.exits['ok'].height


@pytest.mark.parametrize('variant', sorted(fragments.library()['lift'].variants))
def test_every_lane_carries_its_own_platform(variant):
    """One elevator per riding lane -- the bail-out lane carries none."""
    piece = _built(variant)
    elevators = [f for f in piece.features if isinstance(f, Elevator)]
    assert len(elevators) == len(piece.exits) - 1, variant
    for elevator in elevators:
        assert elevator.travel > 0.0, variant


# -- the rule: staying aboard costs waiting, and the wait depends on phase -----

def test_the_bail_out_costs_about_the_same_whatever_the_phase(runs):
    """The lane nobody has to time: it never asks the platform for anything."""
    early = runs['early']
    made = [took for took, _losses in early.values() if took is not None]
    assert len(made) == len(early), \
        'the bail-out failed at some phase: %s' % _report(
            {phase: took for phase, (took, _losses) in early.items()})
    assert max(made) - min(made) < 0.5, \
        'the bail-out ranged from %.2f s to %.2f s across phases (%s)' \
        % (min(made), max(made), _report(
            {phase: took for phase, (took, _losses) in early.items()}))


def test_riding_to_ok_always_gets_there(runs):
    ok = runs['ok']
    report = {phase: took for phase, (took, _losses) in ok.items()}
    made = [took for took in report.values() if took is not None]
    assert len(made) == len(ok), \
        'some phase never got a ride to ok: %s' % _report(report)


def test_when_you_boarded_decides_how_long_the_ride_to_ok_takes(runs):
    """The rule biting: the same exit, the same board, and a real spread in
    how long it takes depending on where the platform was when the marble
    arrived."""
    ok = runs['ok']
    report = {phase: took for phase, (took, _losses) in ok.items()}
    made = [took for took in report.values() if took is not None]
    assert max(made) >= min(made) * 1.3, \
        'riding to ok took between %.2f s and %.2f s across phases, which is ' \
        'not something a phase can be blamed for (%s)' \
        % (min(made), max(made), _report(report))


def test_ok_costs_no_more_than_riding_the_platform_should(runs):
    """However late a marble boards, the wait is at most a lap of the
    platform's own period -- riding it is never a worse plan than the clock
    a board without one would have set."""
    piece = _built('foundry')
    period = next(f.period for f in piece.features if isinstance(f, Elevator))
    clear = _crossing(_aimed_at(piece, 'early'), 0.0)[0]
    ok = runs['ok']
    report = {phase: took for phase, (took, _losses) in ok.items()}
    made = [took for took in report.values() if took is not None]
    assert max(made) <= clear + period + 2.0, \
        'the worst phase took %.2f s against a %.2f s bail-out and a %.1f s ' \
        'period (%s)' % (max(made), clear, period, _report(report))


# -- the rule: nothing here can lose a marble ----------------------------------

def test_a_mistimed_ride_never_loses_the_marble(runs):
    for name, found in runs.items():
        for phase, (_took, losses) in found.items():
            assert losses == 0, \
                '%s at phase %.1f s destroyed %d marble(s) instead of making ' \
                'it wait' % (name, phase, losses)


@pytest.mark.parametrize('variant', sorted(fragments.library()['lift'].variants))
def test_every_variant_is_passable_at_a_hostile_phase(variant):
    """Boarding just as the platform leaves is the worst moment to arrive --
    a full period from the next chance to board -- and it still only costs
    time."""
    piece = _built(variant)
    elevator = next(f for f in piece.features if isinstance(f, Elevator))
    phase = 1e-3
    for name in piece.exits:
        took, losses = _crossing(_aimed_at(piece, name), phase,
                                 seconds=elevator.period + PATIENCE)
        assert took is not None, \
            '%s/%s never got through boarding at the worst phase' % (variant, name)
        assert losses == 0, \
            '%s/%s lost a marble waiting to board' % (variant, name)
