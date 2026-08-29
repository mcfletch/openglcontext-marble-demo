"""The magnets, and what a marble left alone does between them.

The claim is about the *line*, so the measurement is the line: roll a marble
straight down the middle of the corridor with nothing steering it, and record how
far off the middle it ends up.  Down a plain corridor of the same shape that
number is nothing, and the difference between the two is the whole of the piece.

Every measurement is taken with the piece entered facing +Z, so a row of the grid
is a step down the board and X is across the lane.
"""
import random

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, pieces, pilot
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish
from openglcontext_marble_demo.mechanisms.magnet import Magnet, MagnetPull

#: How far along the piece counts as through it, as a fraction of entry to exit.
THROUGH = 0.95

STEP = 1 / 120.0

VARIANTS = sorted(fragments.library()['lodestone'].variants)


def _built(variant=None, seed=3):
    return fragments.build('lodestone', random.Random(seed),
                           pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0,
                                       width=3), variant=variant)


def _posts(piece):
    return [f for f in piece.features if isinstance(f, Magnet)]


def _roll(piece, speed=8.0, magnets=True, seconds=8.0):
    """``(furthest_off_the_middle, reached_the_end)`` rolling straight down it.

    Nothing steers: what is being measured is what the posts do to a marble that
    does nothing about them.
    """
    level = piece.level(time_limit=600.0)
    level.features = [f for f in level.features
                      if not isinstance(f, Finish)
                      and (magnets or not isinstance(f, Magnet))]
    game = MarbleGame(level)
    world, index = game.scene.world, game.marble.index
    world.linear_velocity[index] = (0.0, 0.0, speed)
    world.wake(index)
    past = (max(row for _, row in piece.cells) - 0.5) * pieces.CELL_SIZE
    off, reached = 0.0, False
    for _ in range(int(seconds / STEP)):
        game.advance(STEP)
        where = world.position[index]
        off = max(off, abs(float(where[0])))
        if float(where[2]) >= past:
            reached = True
            break
        if game.controller.fall_count:
            break
    return off, reached


# -- what the piece is made of -------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_the_posts_stand_alternately_down_either_side(variant):
    """A run of posts all one side is a constant lean, which a player sets once
    and then forgets about."""
    piece = _built(variant)
    across = piece.entry.across()
    sides = [(post.cell[0] - piece.entry.cell[0]) * across[0]
             + (post.cell[1] - piece.entry.cell[1]) * across[1]
             for post in _posts(piece)]
    assert len(sides) >= 3, variant
    for before, after in zip(sides, sides[1:], strict=False):
        assert before * after < 0, \
            '%s: two posts running on the same side — %r' % (variant, sides)


@pytest.mark.parametrize('variant', VARIANTS)
def test_a_post_is_scenery_the_marble_can_pass_through(variant):
    """The piece is about the line.  A post that could be hit would make it a
    slalom with a force painted on it."""
    piece = _built(variant)
    for post in _posts(piece):
        assert post.owned_cells() == set(), \
            '%s: a post owns floor, so the piece has a hole in it' % variant
        assert post.cell in piece.cells, \
            '%s: a post stands off the board at %r' % (variant, post.cell)


@pytest.mark.parametrize('variant', VARIANTS)
def test_the_corridor_is_wider_than_a_lane_and_runs_through(variant):
    """There has to be somewhere to be pulled to."""
    piece = _built(variant)
    assert pieces.joined(piece.cells, piece.entry.cell, piece.exit.cell), variant
    across = piece.entry.across()
    lanes = {(col - piece.entry.cell[0]) * across[0]
             + (row - piece.entry.cell[1]) * across[1] for col, row in piece.cells}
    assert len(lanes) > pieces.LANE, \
        '%s: the corridor is %d lanes wide' % (variant, len(lanes))


def test_the_pull_reaches_its_rim_and_stops():
    """A field with no edge is one a player cannot learn."""
    post = _posts(_built())[0]
    pull = MagnetPull(None, [(0.0, 0.0, post.strength, post.radius, post.axis)])
    inside = abs(pull.pull_at((post.radius - 0.5, 0.5, 0.0))[0])
    outside = abs(pull.pull_at((post.radius + 0.5, 0.5, 0.0))[0])
    assert inside > 0.0, 'the pull is nothing just inside its own radius'
    assert outside == 0.0, 'the pull carries %.2f past its radius' % outside


def _drive(piece, speed=8.0, magnets=True, seconds=25.0, dt=1 / 120.0):
    """Drive the corridor with the autopilot, as the rest of the library is driven.

    Answers how far down it got, as a fraction of entry to exit, and how long it
    took to get through.
    """
    level = piece.level(time_limit=900.0)
    if not magnets:
        level.features = [f for f in level.features if not isinstance(f, Magnet)]
    game = MarbleGame(level)
    driver = pilot.Autopilot(level, forward_axis=game.tilt.forward_axis,
                             right_axis=game.tilt.right_axis)
    world, index = game.scene.world, game.marble.index
    facing = piece.entry.facing
    world.linear_velocity[index] = (facing[0] * speed, 0.0, facing[1] * speed)
    world.wake(index)
    begin = np.array(level.cell_center(piece.entry.cell))
    span = np.array(level.cell_center(piece.exit.cell)) - begin
    whole = float(np.dot(span, span)) or 1.0
    across = np.array([-span[1], span[0]])
    across = across / (np.linalg.norm(across) or 1.0)
    furthest, took, off = 0.0, None, 0.0
    for step in range(int(seconds / dt)):
        game.lean(*driver.lean(world.position[index], world.linear_velocity[index]))
        game.advance(dt)
        at = world.position[index]
        here = np.array([at[0], at[2]]) - begin
        furthest = max(furthest, float(np.dot(here, span) / whole))
        off = max(off, abs(float(np.dot(here, across))))
        if furthest >= THROUGH:
            took = step * dt
            break
    return dict(furthest=furthest, took=took, off=off)


# -- and the rule it imposes ---------------------------------------------------

@pytest.mark.parametrize('variant', VARIANTS)
def test_a_marble_left_alone_is_walked_off_the_middle(variant):
    """Which is what there is to lean against."""
    piece = _built(variant)
    pulled, _ = _roll(piece)
    plain, _ = _roll(piece, magnets=False)
    assert pulled > plain + 0.5, \
        '%s: the posts moved the marble %.2f m off the middle, against %.2f m ' \
        'down the same corridor with them taken off' % (variant, pulled, plain)


def test_a_stronger_post_moves_the_marble_further():
    """So the strength is the dial it looks like."""
    weak, _ = _roll(_built('plain'))
    strong, _ = _roll(_built('strong'))
    assert strong > weak, \
        'at 9 m/s/s the marble went %.2f m off the middle and at 14 it went ' \
        '%.2f' % (weak, strong)


@pytest.mark.parametrize('variant', VARIANTS)
def test_a_driven_marble_gets_down_the_corridor(variant):
    """A corridor nothing can get down is a wall.

    Driven rather than rolled straight, which is the standard the rest of the
    library is held to: what this piece costs is the *line*, and a marble that
    does nothing about the posts is walked into the wall and held there.  That
    is the piece working, not a piece that cannot be crossed.
    """
    piece = _built(variant)
    result = _drive(piece, 8.0)
    assert result['furthest'] >= THROUGH, \
        '%s: driven in at 8 m/s the marble got %.2f of the way down' \
        % (variant, result['furthest'])


def test_the_posts_push_a_driven_marble_off_the_line_it_was_holding():
    """What the piece costs a player is the line, so the line is what to measure.

    Not the time: a driven marble gets down the corridor in the same 12.89 s
    whether the posts are there or not, because the autopilot is a controller
    with no reaction time and it cancels a steady sideways force as fast as the
    force arrives.  What it cannot cancel is having been moved, and that is the
    number here.
    """
    piece = _built('strong')
    with_posts = _drive(piece)
    bare = _drive(piece, magnets=False)
    assert with_posts['off'] > bare['off'] + 0.5, \
        'driven down it strayed %.2f m off the line with the posts and %.2f m ' \
        'without' % (with_posts['off'], bare['off'])
