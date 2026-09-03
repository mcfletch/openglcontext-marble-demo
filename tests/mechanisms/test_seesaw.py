"""Tests for the tilting plank mechanism.

The crossing tests run the real physics on a corridor leaning the way the
game's board leans, exactly as
:mod:`~openglcontext_marble_demo.mechanisms.sand`'s tests do, so the seconds
they report are the seconds a player actually waits.
"""
import math

import numpy as np
from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from openglcontext_marble_demo import fragments, levelfile, materials
from openglcontext_marble_demo.game import GRAVITY, ROLL_DAMPING
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.mechanisms.seesaw import Seesaw

CELL = 4.0
RADIUS = 0.5
STEP = 1 / 60.0
#: The plank runs cells 3..7 along +X (see ``_seesaw``); the near edge of that
#: span and a point well clear of its hinge, at the far face of cell 7.
ENTRY, EXIT = 2.5 * CELL, 9.0 * CELL


def _level(features):
    cells = {(col, 0): 0.0 for col in range(-2, 12)}
    return Level(name='seesaw', cells=cells, start_cell=(-2, 0), finish_cell=(11, 0),
                time_limit=120.0, features=[*features, Finish((11, 0))])


def _seesaw(**named):
    settings = dict(max_tilt=7.0, response=0.5)
    settings.update(named)
    return Seesaw(cell=(3, 0), facing=(1, 0), length=5, width=1, **settings)


def _build(level):
    """Build ``level`` into a world leaning downhill along +X.

    The lean is the one this mechanism's rule was measured against
    (:data:`~openglcontext_marble_demo.fragments.DESIGN_TILT`) rather than the
    game's own, which is level: what the rule says happens to a marble being
    carried through the piece, and a world that carries nothing says nothing.
    """
    direction = (math.sin(fragments.DESIGN_TILT), -math.cos(fragments.DESIGN_TILT), 0.0)
    scene = DemoScene(gravity=model.Gravity(gravity=GRAVITY, direction=direction),
                      debug_flags=0,
                      default_linear_damping=ROLL_DAMPING[0],
                      default_angular_damping=ROLL_DAMPING[1])
    index = materials.register_materials(scene.world)
    materials.apply_pair_frictions(scene.world, index)
    return scene, index, level.build_into(scene, index)


def _marble(scene, index, x, velocity=(0.0, 0.0, 0.0), material='steel'):
    ball = scene.world.add_shape(model.Shape.sphere(RADIUS))
    return scene.world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=(x, RADIUS + 0.001, 0.0))


def _crossing(scene, result, marble, limit=40.0):
    """Seconds from the plank's near edge to well clear of its hinge, or ``None``."""
    entered = None
    elapsed = 0.0
    while elapsed < limit:
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)
        elapsed += STEP
        x = scene.world.position[marble][0]
        if entered is None and x >= ENTRY:
            entered = elapsed
        if entered is not None and x >= EXIT:
            return elapsed - entered
    return None


def _run(scene, result, seconds):
    """Drive the world for ``seconds``, as the game drives it."""
    for _ in range(int(seconds / STEP)):
        for animator in result.animators:
            animator.update(STEP)
        scene.world.step(STEP)


def _plank_of(result):
    """The live ``_Plank`` behind the mechanism's ``KinematicAnimator``."""
    return result.animators[0].pose_fn.__self__


# -- what it costs ---------------------------------------------------------

def test_dawdling_costs_seconds_a_bare_corridor_would_not():
    """The rule: cross with speed before the plank finds you, or it digs a
    climb behind you that costs seconds a bare floor never would."""
    cost = {}
    for speed in (1.0, 14.0):
        scene, index, result = _build(_level([_seesaw()]))
        seesaw_t = _crossing(scene, result,
                             _marble(scene, index, ENTRY - 1.0, (speed, 0, 0)))
        scene, index, result = _build(_level([]))
        bare_t = _crossing(scene, result,
                           _marble(scene, index, ENTRY - 1.0, (speed, 0, 0)))
        assert seesaw_t is not None and bare_t is not None, \
            '%.0f m/s never crossed either the seesaw or the bare corridor' % speed
        cost[speed] = seesaw_t - bare_t

    slow_cost, fast_cost = cost[1.0], cost[14.0]
    assert slow_cost > 2.0, \
        'a 1 m/s entry cost only %.2fs over a bare corridor' % slow_cost
    assert fast_cost < 0.5, \
        'a 14 m/s entry cost %.2fs over a bare corridor, not the near-nothing a ' \
        'fast crossing should' % fast_cost
    assert slow_cost > fast_cost * 4, \
        'dawdling (%.2fs over bare) does not cost materially more than charging ' \
        '(%.2fs over bare)' % (slow_cost, fast_cost)


# -- the hinge and the reset -------------------------------------------------

def test_the_hinge_stays_flush_however_far_the_tilt_develops():
    """Whatever the tilt, the seam with the floor beyond the plank never
    moves -- which is what keeps the way on flush at any tilt."""
    scene, index, result = _build(_level([_seesaw(max_tilt=14.0, response=0.2)]))
    plank = _plank_of(result)
    hinge_before = plank.hinge.copy()
    _marble(scene, index, ENTRY - 1.0, (0.5, 0, 0))
    _run(scene, result, 3.0)
    assert plank.tilt > math.radians(5.0), \
        'a 0.5 m/s marble only tipped the plank %.1f degrees in 3s' % math.degrees(plank.tilt)
    assert np.allclose(plank.hinge, hinge_before), \
        'the hinge moved from %s to %s' % (hinge_before, plank.hinge)


def test_an_empty_plank_settles_level():
    scene, index, result = _build(_level([_seesaw(max_tilt=10.0, response=0.3)]))
    plank = _plank_of(result)
    marble = _marble(scene, index, ENTRY - 1.0, (0.5, 0, 0))
    _run(scene, result, 2.0)
    developed = plank.tilt
    assert developed > math.radians(2.0), \
        'the plank only reached %.1f degrees with a marble on it' % math.degrees(developed)
    scene.world.place_body(marble, position=(-4.0, RADIUS, 0.0))   # lifted clear of it
    _run(scene, result, 3.0)
    assert plank.tilt < developed * 0.1, \
        'the plank stayed at %.1f degrees (of a developed %.1f) with nothing on it' \
        % (math.degrees(plank.tilt), math.degrees(developed))


def test_a_restarted_run_finds_the_plank_level_again():
    scene, index, result = _build(_level([_seesaw(max_tilt=10.0, response=0.3)]))
    plank = _plank_of(result)
    assert plank in result.resettable
    _marble(scene, index, ENTRY - 1.0, (0.5, 0, 0))
    _run(scene, result, 2.0)
    assert plank.tilt > 0.0, 'the plank never tipped, so the reset is not being tested'
    plank.reset()
    assert plank.tilt == 0.0


# -- shape and the file format -----------------------------------------------

def test_owned_cells_is_the_whole_footprint():
    plank = Seesaw(cell=(1, 0), facing=(1, 0), length=3, width=3)
    assert plank.owned_cells() == {(1, -1), (1, 0), (1, 1),
                                   (2, -1), (2, 0), (2, 1),
                                   (3, -1), (3, 0), (3, 1)}


def test_the_registry_knows_it():
    from openglcontext_marble_demo import mechanisms
    assert mechanisms.registry()['seesaw'] is Seesaw


def test_a_seesaw_round_trips_through_a_file():
    level = _level([_seesaw(max_tilt=12.0, response=0.4)])
    again = levelfile.from_json(levelfile.to_json(level))
    plank = again.features[0]
    assert isinstance(plank, Seesaw)
    assert plank.cell == (3, 0) and plank.facing == (1, 0)
    assert plank.length == 5 and plank.width == 1
    assert plank.max_tilt == 12.0 and plank.response == 0.4
