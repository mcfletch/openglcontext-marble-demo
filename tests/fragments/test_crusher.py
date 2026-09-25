"""The crusher, and what arriving under it costs against reading its rhythm.

A rotating arm (see ``tests/fragments/test_gauntlet.py``) never shuts a lane for
good, so it can only be held to costing time.  A press is worse: held under a
blow long enough, the marble is gone.  So the question this file answers is the
one the brief sets -- not "can the wrong moment be survived" but "is the wrong
moment rare and the right one easy to find, and does *stopping* under the press,
rather than merely being there, is what a marble cannot survive."

Both are answered by driving the real piece: the game's own
:class:`~openglcontext_marble_demo.pilot.Autopilot` crosses it with the press
started at several points in its cycle, the same question as a marble reaching
the press at various moments in a real run.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pilot
from openglcontext_marble_demo.controller import CRUSHED, DESTROYED
from openglcontext_marble_demo.fragments import crusher as piece_module
from openglcontext_marble_demo.game import WON, MarbleGame
from openglcontext_marble_demo.mechanisms.crusher import Crusher
from openglcontext_marble_demo.pieces import Port

STEP = 1 / 120.0

#: The press's phases a run is tried at, in seconds of its cycle -- the same
#: question as a marble reaching it at various moments in a real run.
PHASES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
#: How long a run is given before it counts as never getting through.
PATIENCE = 20.0


def _entry():
    return Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('crusher', random.Random(seed), _entry(),
                           variant=variant, **named)


def _presses(piece):
    return [f for f in piece.features if isinstance(f, Crusher)]


def _crossing(level, phase=0.0, seconds=PATIENCE):
    """Play ``level`` with every press already ``phase`` seconds into its cycle.

    Answers the seconds the run took, ``'crushed'`` if the press took the
    marble, or ``None`` if :data:`PATIENCE` ran out with neither.
    """
    game = MarbleGame(level)
    for animator in game.build.animators:
        animator.time = phase
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis)
    world, index = game.scene.world, game.marble.index
    played = 0.0
    for _ in range(int(seconds / STEP)):
        game.lean(*driver.lean(world.position[index], world.linear_velocity[index]))
        state = game.advance(STEP)
        played += STEP
        if state != 'playing':
            break
    if game.state == WON:
        return played
    if game.controller.state == DESTROYED:
        return 'crushed'
    return None


@pytest.fixture(scope='module')
def runs():
    """``(clear_lane, {phase: outcome})`` for the ``plain`` variant."""
    piece = _built('plain')
    clear = piece.level(time_limit=600.0)
    clear.features = [f for f in clear.features if not isinstance(f, Crusher)]
    level = piece.level(time_limit=600.0)
    return (_crossing(clear), {phase: _crossing(level, phase) for phase in PHASES})


def _report(found):
    return ', '.join(
        '%.1f s in: %s' % (phase, '%.2f s' % took if isinstance(took, float)
                           else (took or 'still going'))
        for phase, took in sorted(found.items()))


# -- the whole point: reading the rhythm gets through, stopping does not -------

def test_a_run_that_reads_the_rhythm_gets_through(runs):
    clear, found = runs
    made = [took for took in found.values() if isinstance(took, float)]
    assert made, 'no phase of the press let a marble by at all: %s' % _report(found)
    assert len(made) >= len(found) - 1, (
        'only %d of %d phases got through untouched, against a %.2f s clear '
        'lane: %s' % (len(made), len(found), clear, _report(found)))
    assert min(made) <= clear * 1.4, (
        'the best of the phases took %.2f s against %.2f s down the same lane '
        'with the press taken out (%s)' % (min(made), clear, _report(found)))


def test_a_run_made_at_the_wrong_moment_can_be_lost_to_the_press(runs):
    """A rule that nothing can fail is scenery; this one can be got wrong."""
    _clear, found = runs
    caught = [phase for phase, took in found.items() if took == 'crushed']
    assert caught, (
        'no phase of the press caught a crossing marble at all, so the rule '
        'cannot be got wrong: %s' % _report(found))


def test_a_marble_left_under_the_press_is_lost_to_crushed():
    """The other half: not being there at the wrong moment, but staying there."""
    piece = _built('plain')
    level = piece.level(time_limit=600.0)
    press = _presses(piece)[0]
    game = MarbleGame(level)
    world, marble = game.scene.world, game.marble.index
    x, z = level.cell_center(press.cells[0])
    world.place_body(marble, position=(x, game.radius + 0.001, z))
    world.linear_velocity[marble] = (0.0, 0.0, 0.0)
    world.wake(marble)
    limit = press.period + press.strike * 2.0 + press.dwell
    crushed_at = None
    for i in range(int(limit / STEP)):
        game.advance(STEP)
        if game.controller.state == DESTROYED:
            crushed_at = i * STEP
            break
    assert crushed_at is not None, (
        'a marble parked under the press was never crushed in %.2f s, more '
        'than a full cycle (period %.2f s)' % (limit, press.period))
    assert game.controller.last_loss == CRUSHED, (
        'lost to %r rather than being crushed' % (game.controller.last_loss,))
    assert crushed_at < press.strike + press.dwell + 0.05, (
        'took %.3f s to crush a stationary marble, longer than one blow '
        '(strike %.2f s + dwell %.2f s)' % (crushed_at, press.strike, press.dwell))


# -- the shape: no way round, and the variants are different pieces ------------

def test_every_variant_has_at_least_one_press():
    for variant in fragments.library()['crusher'].variants:
        piece = _built(variant)
        assert _presses(piece), variant


def test_the_press_spans_the_whole_width_of_its_lane():
    """Every press covers the lane's own width -- a gate, not a post with a
    way round it on either side."""
    for variant, settings in piece_module.VARIANTS.items():
        piece = _built(variant)
        for press in _presses(piece):
            assert len(press.cells) == settings['width'], \
                '%s: press covers %d cells, lane is %d wide' \
                % (variant, len(press.cells), settings['width'])


def test_the_foundry_variant_puts_two_presses_in_the_lane_where_plain_puts_one():
    """``presses`` is a layout axis a story can ask for by name."""
    assert len(_presses(_built('plain'))) == 1
    assert len(_presses(_built('foundry'))) == 2


def test_the_wide_variant_is_wider_than_plain():
    """``width`` is a layout axis distinct from how many presses stand in it."""
    plain = piece_module.VARIANTS['plain']['width']
    wide = piece_module.VARIANTS['wide']['width']
    assert wide > plain, 'wide (%d) is not wider than plain (%d)' % (wide, plain)


def test_the_brisk_variant_strikes_on_a_shorter_period_than_plain():
    """``period`` is the third axis: the same rule, read faster."""
    plain = piece_module.VARIANTS['plain']['period']
    brisk = piece_module.VARIANTS['brisk']['period']
    assert brisk < plain, 'brisk (%.1f s) is not faster than plain (%.1f s)' \
        % (brisk, plain)


def test_the_two_presses_of_the_foundry_variant_are_out_of_step():
    """Two presses that always struck together would be one press twice as
    long, not a second question."""
    presses = _presses(_built('foundry'))
    phases = {press.phase % press.period for press in presses}
    assert len(phases) == len(presses), \
        'the presses share a phase: %s' % sorted(phases)
