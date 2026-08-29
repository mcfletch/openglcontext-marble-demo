"""The cannon: which lane you enter decides where you land.

Every barrel fires unconditionally the moment a marble crosses it -- unlike
:mod:`~openglcontext_marble_demo.fragments.drop`, where the rule is the speed a
marble carries to a lip, here the aim is made *before* the launcher, at the
mouth, and the launcher does not care how fast a marble arrived.  The rule is
demonstrated the same way :mod:`~openglcontext_marble_demo.fragments.drop` is:
fly the marble down a lane and record where it comes down, with the distance in
every assertion.
"""
import random

import pytest

from openglcontext_marble_demo import fragments, pieces
from openglcontext_marble_demo.fragments.cannon import VARIANTS
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import CELL_SIZE, Finish, Ramp

STEP = 1 / 120.0

#: How far behind a lane's own ramp cell the marble starts, so it arrives
#: rolling rather than materialising on top of the trigger.
APPROACH = CELL_SIZE * 0.4

#: The three entry speeds every measurement is taken at.  Chosen wide enough to
#: show a lane's reach does not depend on how fast a marble arrived -- that is
#: what "the aim happens before the launch" has to mean.
SPEEDS = (3.0, 6.0, 9.0)


def _entry():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _built(variant=None, seed=3, **named):
    return fragments.build('cannon', random.Random(seed), _entry(),
                           variant=variant, **named)


def _ramps(piece):
    """Every lane's launch ramp, weakest (shortest reach) first."""
    return sorted((f for f in piece.features if isinstance(f, Ramp)),
                  key=lambda ramp: ramp.launch_up)


def _came_down(piece, ramp, speed, seconds=14.0):
    """How far past ``ramp``'s own cell the marble first comes back to the
    entry's floor height, entering that lane at ``speed``.

    ``None`` if it neither lands nor falls within ``seconds``, which would
    itself be a defect: every lane is sized to bring a marble down well inside
    its own walled corridor.
    """
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features if not isinstance(f, Finish)]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    x, z = level.cell_center(ramp.cell)
    facing = piece.entry.facing
    world.position[index] = (x - facing[0] * APPROACH, piece.entry.height + 0.6,
                             z - facing[1] * APPROACH)
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    start = float(world.position[index][2] if facing[1] else world.position[index][0])
    axis = 2 if facing[1] else 0
    plane = piece.entry.height + 0.65
    airborne = False
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        at = world.position[index]
        if at[1] > plane + 0.1:
            airborne = True
        if airborne and at[1] <= plane:
            return float(at[axis]) - start
        if game.controller.fall_count:
            return None
    return None


def _say(down):
    return '%.1f m past the ramp' % down if down is not None else 'never came down'


# -- it is a fragment like any other -----------------------------------------

def test_the_library_holds_it():
    assert 'cannon' in fragments.library()


def test_it_offers_four_variants():
    assert sorted(fragments.library()['cannon'].variants) == sorted(VARIANTS)


def test_it_says_what_it_asks():
    assert fragments.library()['cannon'].rule


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_every_variant_offers_ok_and_a_named_short_exit(variant):
    piece = _built(variant)
    assert 'ok' in piece.exits, variant
    assert 'short' in piece.exits, \
        '%s has no exit named "short" for the weakest lane' % variant
    assert piece.exits['ok'].cell in piece.cells, variant


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_every_exit_is_reachable_from_the_entry(variant):
    """Every lane -- named or not -- is somewhere a marble can continue from,
    which is what "no lane throws the marble off the board" means in cells."""
    piece = _built(variant)
    for name, port in piece.exits.items():
        assert pieces.joined(piece.cells, piece.entry.cell, port.cell), \
            '%s/%s is not reachable from the entry' % (variant, name)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_every_lane_holds_the_slope_budget(variant):
    piece = _built(variant)
    for (col, row), height in piece.cells.items():
        for step in ((1, 0), (0, 1)):
            beside = (col + step[0], row + step[1])
            if beside in piece.cells:
                assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                    '%s steps at %r' % (variant, (col, row))


# -- the rule: the lane picks the landing ------------------------------------

@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_the_lane_changes_where_the_marble_comes_down(variant):
    """Fly each lane at three entry speeds and show the landings differ --
    which is the whole of the rule, since nothing about *when* to launch is
    left for a player to get right."""
    piece = _built(variant)
    ramps = _ramps(piece)
    assert len(ramps) >= 2, '%s has only one lane' % variant

    by_lane = [[_came_down(piece, ramp, speed) for speed in SPEEDS]
               for ramp in ramps]
    report = ', '.join(
        'launch_up=%.2f: %s' % (ramp.launch_up,
                                ', '.join(_say(d) for d in dists))
        for ramp, dists in zip(ramps, by_lane, strict=False))

    for distances in by_lane:
        assert all(d is not None for d in distances), \
            '%s: a lane never came down -- %s' % (variant, report)

    weakest, strongest = by_lane[0], by_lane[-1]
    for speed, short_down, ok_down in zip(SPEEDS, weakest, strongest, strict=False):
        assert ok_down > short_down, \
            '%s at %.0f m/s: ok landed %.1f m and short landed %.1f m -- ' \
            'the lane made no difference: %s' % (
                variant, speed, ok_down, short_down, report)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_ok_lands_well_clear_of_short(variant):
    """"Much better" means a real margin, not a coin flip's worth."""
    piece = _built(variant)
    ramps = _ramps(piece)
    short_ramp, ok_ramp = ramps[0], ramps[-1]
    speed = 6.0
    short_down = _came_down(piece, short_ramp, speed)
    ok_down = _came_down(piece, ok_ramp, speed)
    assert short_down is not None and ok_down is not None, \
        '%s: %s / %s' % (variant, _say(short_down), _say(ok_down))
    margin = ok_down - short_down
    assert margin >= 4.0, \
        '%s: ok landed %.1f m and short %.1f m, only %.1f m clear of it' % (
            variant, ok_down, short_down, margin)


def test_a_middle_lane_lands_between_short_and_ok():
    """The variants with more than two lanes offer more than a binary choice."""
    for variant in ('triple', 'quad', 'far'):
        piece = _built(variant)
        ramps = _ramps(piece)
        assert len(ramps) >= 3, variant
        speed = 6.0
        distances = [_came_down(piece, ramp, speed) for ramp in ramps]
        report = ', '.join('%.2f -> %s' % (ramp.launch_up, _say(d))
                           for ramp, d in zip(ramps, distances, strict=False))
        assert distances == sorted(distances), \
            '%s: the lanes are not in increasing order of reach -- %s' % (
                variant, report)
