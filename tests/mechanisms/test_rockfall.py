"""The rockfall: a slope you can bounce down, that picks the way you leave it.

What a rockfall claims is a claim about variance, so most of what is here is
statistical: a marble is sent down four fields from thirteen entry points each,
all within a cell of one another, and three questions are asked of the answers.

Does it scatter?  The spread of exit headings is read against the same slope
with the rock taken off it, which sends every entry the same way to a fraction
of a degree.  A rockfall whose spread were that small would be a ramp.

Can it be got through?  A hazard nobody survives is a wall, so most descents
have to reach the bottom rather than jam between two slabs or be thrown back up.

Does it cost the speed it saves?  The marble is measured crossing the top of the
slope and again crossing the bottom; a rockfall that stopped it dead would never
be worth committing to.

The fields and the entries are fixed and the physics is deterministic, so each
of these is one number rather than a sample, and every assertion carries the
number it measured in its message -- a change in the physics then says what it
did rather than only that something moved.
"""
import json
import math

import numpy as np
import pytest

from openglcontext_marble_demo import fragments, levelfile
from openglcontext_marble_demo.game import MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.rockfall import Rockfall

CELL = 4.0
RUN_IN, FALL, RUN_OUT = 3, 6, 4          # rows of flat, rockfall, and flat again
DROP = 0.6                               # how far each rockfall row falls
LAST = RUN_IN + FALL + RUN_OUT - 1

#: Where the slope begins and ends, as Z in world metres (a cell's height is the
#: height of its uphill edge, so the lines sit on the boundaries between rows).
TOP_Z = (RUN_IN - 0.5) * CELL
BOTTOM_Z = (RUN_IN + FALL - 0.5) * CELL

#: The fields every statistical measurement is made over.  Four seeds rather
#: than one: a single seed measures the field that seed happened to lay, and
#: what is claimed is about rockfalls.
FIELDS = (3, 7, 12, 19)

#: Where each field is entered from: 3.6 metres of centre line at 30 cm steps,
#: which is less than a cell across.  Close enough together that a slope would
#: send them all the same way, and that is the point of them.
ENTRIES = tuple(round(-1.8 + step * 0.3, 2) for step in range(13))

#: How long a descent is given before it counts as having jammed.  Five seconds
#: is the free-rolling crossing; the rest is room for one that is thrown about
#: on the way.
PATIENCE = 14.0


def _cells():
    """A wide lane: flat, then falling for :data:`FALL` rows, then flat."""
    cells = {}
    for row in range(RUN_IN + FALL + RUN_OUT):
        if row < RUN_IN:
            height = 0.0
        elif row < RUN_IN + FALL:
            height = -(row - RUN_IN) * DROP
        else:
            height = -FALL * DROP
        for col in range(-3, 4):
            cells[(col, row)] = height
    return cells


def _level(*features):
    return Level(name='rockfall-under-test', cells=_cells(), start_cell=(0, 0),
                 finish_cell=(0, LAST), time_limit=600.0,
                 features=[*features, Finish((0, LAST))])


def _rockfall(**named):
    named.setdefault('cell', (0, RUN_IN))
    named.setdefault('length', FALL)
    named.setdefault('width', 5)
    named.setdefault('seed', 7)
    return Rockfall(**named)


def _descend(feature, offset, speed=6.0):
    """Send a marble down the slope from ``offset`` metres off the centre line.

    Answers what it was doing as it crossed the top and the bottom of the slope:
    the speed at each, the heading it left with, and whether it got there at all.
    """
    game = MarbleGame(_level(feature) if feature is not None else _level(),
                      debug_flags=0, base_tilt=fragments.DESIGN_TILT)
    world, marble = game.scene.world, game.marble.index

    # The crash rule is the one thing on the way down that could turn a scatter
    # into a stop, so count the times it fires rather than infer it from speed.
    crashes = [0]
    scrub = game.controller._scale_horizontal_speed

    def counted(retain):
        crashes[0] += 1
        return scrub(retain)
    game.controller._scale_horizontal_speed = counted

    world.position[marble] = (offset, 0.6, 0.0)
    world.linear_velocity[marble] = (0.0, 0.0, speed)
    world.wake(marble)

    entered = None
    for _ in range(int(PATIENCE * 120)):
        game.advance(1 / 120.0)
        where, moving = world.position[marble], world.linear_velocity[marble]
        speed_now = float(np.hypot(moving[0], moving[2]))
        if entered is None and where[2] >= TOP_Z:
            entered = speed_now
        elif entered is not None and where[2] >= BOTTOM_Z:
            return {'entry': entered, 'exit': speed_now, 'through': True,
                    'heading': math.degrees(math.atan2(moving[0], moving[2])),
                    'crashes': crashes[0]}
    return {'entry': entered, 'exit': 0.0, 'through': False, 'heading': None,
            'crashes': crashes[0]}


def _headings(results):
    return [run['heading'] for run in results if run['through']]


@pytest.fixture(scope='module')
def over_rock():
    """Every entry down every field, keyed by the seed that laid the field."""
    return {seed: [_descend(_rockfall(seed=seed), offset) for offset in ENTRIES]
            for seed in FIELDS}


@pytest.fixture(scope='module')
def every_descent(over_rock):
    return [run for field in over_rock.values() for run in field]


@pytest.fixture(scope='module')
def over_bare_slope():
    """The same slope with nothing on it: what a rockfall's spread is read against."""
    return [_descend(None, offset) for offset in ENTRIES]


# -- what it is made of --------------------------------------------------------

def test_a_rockfall_owns_the_cells_it_covers():
    fall = _rockfall(length=4, width=3)
    assert len(fall.owned_cells()) == 12
    assert (0, RUN_IN) in fall.owned_cells()
    assert (1, RUN_IN + 3) in fall.owned_cells()


def test_a_rockfall_lays_rock_only_on_its_own_cells():
    level = _level()
    fall = _rockfall()
    owned = fall.owned_cells()
    for rock in fall.rocks(level):
        col = int(round(rock.position[0] / CELL))
        row = int(round(rock.position[2] / CELL))
        assert (col, row) in owned, rock


def test_the_rock_stands_proud_of_the_slope_it_lies_on():
    """Buried rock is a floor; the slabs have to be in the way."""
    level = _level()
    for rock in _rockfall().rocks(level):
        assert rock.stand > 0.0, rock
        # Low enough that a marble of the game's radius rides over rather than
        # meets a face; higher than this and the field is a set of walls.
        assert rock.stand < 0.6, rock


def test_the_rock_lies_at_angles_rather_than_square_to_the_slope():
    """Square boxes are a grid; what makes a rockfall chaotic is that it is not one."""
    angles = [rock.angle for rock in _rockfall().rocks(_level())]
    assert min(angles) > 0.0
    assert max(angles) - min(angles) > 0.05


def test_the_rock_is_hard_enough_that_the_crash_rule_lets_it_alone():
    """Hard rock returns a marble's speed, and the crash rule exempts surfaces
    that do -- which is what keeps a field of rock a scatter rather than a stop."""
    from openglcontext_marble_demo.mechanisms import rockfall
    exempt_above = MarbleGame(_level(), debug_flags=0).controller.elastic_restitution
    assert rockfall.ROCK.restitution >= exempt_above


# -- the same seed is the same rock --------------------------------------------

def test_the_same_seed_builds_the_same_rock():
    level = _level()
    first = _rockfall(seed=4).rocks(level)
    again = _rockfall(seed=4).rocks(level)
    assert first == again


def test_a_different_seed_builds_different_rock():
    level = _level()
    first = _rockfall(seed=4).rocks(level)
    other = _rockfall(seed=5).rocks(level)
    assert len(first) == len(other)
    assert first != other


def test_the_same_seed_puts_the_same_rock_in_a_built_scene():
    """Reproducible as far as the physics, not only as far as the data."""
    level = _level(_rockfall(seed=4))
    first = MarbleGame(level, debug_flags=0)
    again = MarbleGame(_level(_rockfall(seed=4)), debug_flags=0)
    assert np.allclose(first.scene.world.position, again.scene.world.position)


# -- the file format -----------------------------------------------------------

def test_a_rockfall_survives_the_round_trip():
    before = _level(_rockfall(cell=(2, 3), direction=(1, 0), length=5, width=3,
                              seed=19, density=1.4))
    after = levelfile.from_json(json.loads(json.dumps(levelfile.to_json(before))))
    saved = after.features[0]
    assert isinstance(saved, Rockfall)
    assert saved == before.features[0]
    assert saved.cell == (2, 3)
    assert saved.direction == (1, 0)


def test_a_rockfall_is_saved_under_its_own_name():
    document = levelfile.to_json(_level(_rockfall()))
    assert document['features'][0]['kind'] == 'rockfall'


# -- what it does to a marble --------------------------------------------------

@pytest.mark.slow
def test_a_bare_slope_sends_every_run_the_same_way(over_bare_slope):
    """The reading the rockfall's spread is measured against."""
    headings = _headings(over_bare_slope)
    assert len(headings) == len(over_bare_slope), 'the bare slope did not pass a marble'
    spread = float(np.std(headings))
    assert spread < 1.0, 'a bare slope already scatters by %.2f degrees' % spread


@pytest.mark.slow
def test_two_runs_down_one_rockfall_are_two_different_runs(over_rock, over_bare_slope):
    """Entry points within a cell of each other become exits nothing alike.

    Asked of each field on its own: mixing the seeds would measure how much two
    rockfalls differ, and what is claimed is that *one* rockfall answers the
    same question two ways.
    """
    bare = float(np.std(_headings(over_bare_slope)))
    spreads = {seed: float(np.std(_headings(runs))) for seed, runs in over_rock.items()}
    report = ', '.join('seed %d: %.2f' % (seed, spread)
                       for seed, spread in sorted(spreads.items()))
    assert min(spreads.values()) > 4.0, \
        'exit headings spread by (degrees) %s, against %.2f on the bare slope' \
        % (report, bare)
    assert min(spreads.values()) > bare * 8.0, \
        'the rock spreads exits by %s against the bare slope %.2f' % (report, bare)


@pytest.mark.slow
def test_a_rockfall_sends_the_widest_two_exits_far_apart(over_rock):
    """A spread that is all small deflections is a rough surface, not a rockfall."""
    spans = {seed: max(_headings(runs)) - min(_headings(runs))
             for seed, runs in over_rock.items() if len(_headings(runs)) > 1}
    report = ', '.join('seed %d: %.2f' % (seed, span) for seed, span in sorted(spans.items()))
    assert min(spans.values()) > 15.0, \
        'the widest two exits of each field are (degrees) %s apart' % report


@pytest.mark.slow
def test_most_descents_of_a_rockfall_get_through(every_descent):
    """A hazard nobody survives is a wall."""
    got = sum(run['through'] for run in every_descent)
    assert got >= 0.75 * len(every_descent), \
        '%d of %d descents reached the bottom' % (got, len(every_descent))


@pytest.mark.slow
def test_a_rockfall_leaves_the_marble_moving(every_descent, over_bare_slope):
    """It is a way through rather than a stop: what it mainly changes is the
    direction.

    Measured against the bare slope rather than against one, because the same
    six cells of descent hand a marble 1.29 times the speed it entered with and
    the rock is what that is spent on.  Half of it goes: 0.64 of the entry speed
    out of a possible 1.29, over every descent that reaches the bottom.  The
    crash rule -- which scrubs a marble to a tenth — fires on none of them, and
    that is the difference this is drawing: a rockfall costs a descent's worth
    of speed, and a wall costs all of it.
    """
    kept = [run['exit'] / run['entry'] for run in every_descent if run['through']]
    bare = float(np.mean([run['exit'] / run['entry']
                          for run in over_bare_slope if run['through']]))
    average = float(np.mean(kept))
    report = ('%.2f of the entry speed, against %.2f down the bare slope '
              '(worst %.2f)' % (average, bare, min(kept)))
    assert average > 0.5, 'the marble leaves at ' + report
    assert not sum(run['crashes'] for run in every_descent), \
        'the crash rule fired on a rockfall, which is what ROCK exists to ' \
        'prevent: ' + report
    # The floor is well under the average because the descents at the bottom of
    # the range are the ones that were thoroughly turned on the way down, which
    # is the piece working: what they are not is stopped, and a tenth is what
    # the crash rule would have left.
    assert min(kept) > 0.15, \
        'the worst descent kept only %.2f of its speed (average %.2f)' \
        % (min(kept), average)


@pytest.mark.slow
def test_the_crash_rule_does_not_fire_on_the_rock(every_descent):
    """The measurement behind :data:`~openglcontext_marble_demo.mechanisms.rockfall.ROCK`.

    A marble the rule takes arrives at the bottom with a tenth of its speed
    rather than with a heading it did not choose.  The same field laid in a
    surface the rule does not exempt takes it eight times in fourteen descents.
    """
    crashes = sum(run['crashes'] for run in every_descent)
    assert crashes == 0, 'the crash rule fired %d times over %d descents' \
        % (crashes, len(every_descent))
