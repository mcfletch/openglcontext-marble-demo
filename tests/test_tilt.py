"""The board's lean: what the player asks for, and what the board is doing.

:class:`~openglcontext_marble_demo.tilt.TiltRig` is the whole control model with
no marble, no physics and no window in it: held input goes in, and a gravity
direction and a board rotation come out.  That is what makes the feel testable —
every question about how the controls respond is a question about this object.
"""
import math

import numpy as np
import pytest

from openglcontext_marble_demo.tilt import TiltRig

# The camera-relative ground axes run.py computes; the plain world axes here.
FORWARD = (0.0, 0.0, -1.0)      # up-screen, away from the viewer
RIGHT = (1.0, 0.0, 0.0)
DOWNHILL = (0.0, 0.0, 1.0)      # the board's constant lean, toward the finish


def _rig(**named):
    named.setdefault('base', math.radians(12))
    named.setdefault('limit', math.radians(25))
    named.setdefault('forward_axis', FORWARD)
    named.setdefault('right_axis', RIGHT)
    named.setdefault('downhill_axis', DOWNHILL)
    return TiltRig(**named)


def _hold(rig, forward, right, seconds, dt=1 / 120.0):
    """Hold an input for ``seconds`` and return the rig."""
    for _ in range(int(round(seconds / dt))):
        rig.update(dt, forward, right)
    return rig


# -- what the board does with no input -----------------------------------------

def test_a_rig_starts_level():
    assert _rig().pitch == 0.0
    assert _rig().roll == 0.0


def test_an_untouched_board_still_leans_toward_the_finish():
    """The base lean is what makes the marble roll on its own."""
    direction = _rig().gravity_direction()
    assert direction[1] < 0                       # still mostly downward
    assert direction[2] > 0                       # and leaning downhill (+Z)
    assert direction[0] == pytest.approx(0.0)     # with no sideways component


def test_the_base_lean_is_the_angle_it_says_it_is():
    for degrees in (0, 6, 12, 20):
        direction = _rig(base=math.radians(degrees)).gravity_direction()
        from_vertical = math.degrees(math.acos(-direction[1]))
        assert from_vertical == pytest.approx(degrees, abs=1e-6)


def test_gravity_direction_is_a_unit_vector():
    rig = _hold(_rig(), 1.0, 1.0, 1.0)
    assert np.linalg.norm(rig.gravity_direction()) == pytest.approx(1.0)


# -- what the board does with input --------------------------------------------

def test_holding_right_leans_the_board_right():
    rig = _hold(_rig(), 0.0, 1.0, 1.0)
    assert rig.roll > 0
    assert rig.gravity_direction()[0] > 0          # gravity now pulls +X


def test_holding_left_leans_the_board_left():
    rig = _hold(_rig(), 0.0, -1.0, 1.0)
    assert rig.gravity_direction()[0] < 0


def test_holding_forward_leans_the_board_up_the_slope():
    """Forward is up-screen, which is -Z: it fights the downhill lean."""
    level = _rig().gravity_direction()[2]
    leaning = _hold(_rig(), 1.0, 0.0, 1.0).gravity_direction()[2]
    assert leaning < level


def test_a_hard_enough_forward_lean_beats_the_base_lean():
    """Full authority is enough to hold station against the board, and no more."""
    rig = _hold(_rig(base=math.radians(12), limit=math.radians(25)), 1.0, 0.0, 3.0)
    assert rig.gravity_direction()[2] < 0          # now pulling back up the slope


# -- the lean is proportional and bounded --------------------------------------

def test_a_brief_tap_leans_the_board_a_little_and_a_hold_leans_it_fully():
    """Every input size is available: this is what a kick-per-keypress cannot do."""
    tap = _hold(_rig(), 0.0, 1.0, 0.03).roll
    hold = _hold(_rig(), 0.0, 1.0, 2.0).roll
    assert 0 < tap < hold / 4


def test_the_lean_never_passes_the_limit():
    rig = _hold(_rig(limit=math.radians(25)), 0.0, 1.0, 10.0)
    assert rig.roll == pytest.approx(math.radians(25), abs=1e-9)


def test_a_diagonal_hold_does_not_exceed_the_limit_on_either_axis():
    rig = _hold(_rig(limit=math.radians(25)), 1.0, 1.0, 10.0)
    assert abs(rig.pitch) <= math.radians(25) + 1e-9
    assert abs(rig.roll) <= math.radians(25) + 1e-9


def test_a_partial_input_asks_for_that_fraction_of_the_pull():
    """An analog stick at half deflection asks for half the sideways gravity.

    Half the *pull* rather than half the angle: the pull is what the marble
    answers to, and it is what makes a half-and-full diagonal fall in the
    direction it was asked to.
    """
    half = _hold(_rig(), 0.0, 0.5, 10.0).roll
    full = _hold(_rig(), 0.0, 1.0, 10.0).roll
    assert math.tan(half) == pytest.approx(math.tan(full) / 2.0, rel=1e-6)
    assert half < full / 2.0 * 1.1


# -- letting go ----------------------------------------------------------------

def test_releasing_returns_the_board_to_level():
    """Asymptotically: a damped board approaches level rather than arriving at
    it, so what is asserted is that nothing worth seeing is left."""
    rig = _hold(_rig(), 0.0, 1.0, 2.0)
    _hold(rig, 0.0, 0.0, 4.0)
    assert abs(math.degrees(rig.roll)) < 0.01


def test_the_board_does_not_overshoot_level_on_the_way_back():
    """A spring that crossed zero would make the board twitch the other way."""
    rig = _hold(_rig(), 0.0, 1.0, 2.0)
    for _ in range(600):
        rig.update(1 / 120.0, 0.0, 0.0)
        assert rig.roll >= -1e-12


def test_settling_can_be_slower_than_the_push():
    """Separate rates: a board that snaps flat the instant a key lifts feels twitchy."""
    quick = _rig(stiffness=9.0, settle=9.0)
    slow = _rig(stiffness=9.0, settle=2.0)
    _hold(quick, 0.0, 1.0, 2.0)
    _hold(slow, 0.0, 1.0, 2.0)
    _hold(quick, 0.0, 0.0, 0.1)
    _hold(slow, 0.0, 0.0, 0.1)
    assert slow.roll > quick.roll


def test_reversing_the_input_crosses_level_at_the_push_rate():
    """Pushing the other way is a push, not a settling."""
    rig = _rig(stiffness=9.0, settle=1.0)
    _hold(rig, 0.0, 1.0, 2.0)
    _hold(rig, 0.0, -1.0, 0.8)
    assert rig.roll < 0


# -- frame-rate independence ---------------------------------------------------

def test_the_lean_does_not_depend_on_the_frame_rate():
    """The same held second leans the same board on a fast machine and a slow one.

    To within what an integrator gives: the lean is a differential equation
    stepped once a frame, so eight times the step size is not the same arithmetic
    -- only the same answer.
    """
    slow = _hold(_rig(), 0.0, 1.0, 0.5, dt=1 / 30.0)
    fast = _hold(_rig(), 0.0, 1.0, 0.5, dt=1 / 240.0)
    assert slow.roll == pytest.approx(fast.roll, rel=0.05)


# -- the board rotation the renderer draws -------------------------------------

def test_a_level_board_has_no_visible_rotation():
    axis_x, axis_y, axis_z, angle = _rig().board_rotation()
    assert angle == pytest.approx(0.0)


def _drawn_normal(rig):
    """Where the board's up-vector ends up once the renderer applies the lean.

    Through ``vrml.vrml97.transformmatrix`` -- the same call
    ``Transform.localMatrix`` makes -- rather than an assumed axis-angle
    convention, because what matters is which way the board tips on screen.
    """
    from vrml.vrml97 import transformmatrix
    matrix = transformmatrix.transformMatrix(rotation=rig.board_rotation())
    return (np.array([0.0, 1.0, 0.0, 0.0]) @ matrix)[:3]


def test_leaning_right_drops_the_right_edge_of_the_drawn_board():
    """The board must tip the way it pulls.

    Lean right: gravity gains +X, so the board's +X edge is the low one and its
    normal leans to -X.  Drawn the other way round, the board would appear to
    climb into the direction the marble accelerates.
    """
    rig = _hold(_rig(), 0.0, 1.0, 2.0)
    assert rig.gravity_direction()[0] > 0        # pulling right
    assert _drawn_normal(rig)[0] < 0             # and tipped down to the right


def test_leaning_left_drops_the_left_edge_of_the_drawn_board():
    rig = _hold(_rig(), 0.0, -1.0, 2.0)
    assert rig.gravity_direction()[0] < 0
    assert _drawn_normal(rig)[0] > 0


def test_the_drawn_board_tips_the_way_the_player_leaned_it():
    """For any held direction, the drawn normal opposes the player's own pull."""
    for forward, right in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)):
        rig = _hold(_rig(base=0.0), forward, right, 2.0)
        pull = rig.gravity_direction()[[0, 2]]
        normal = _drawn_normal(rig)[[0, 2]]
        assert np.dot(pull, normal) < 0, (forward, right)


def test_the_drawn_board_stays_a_rotation():
    """The lean tips the board; it must not stretch or shrink it."""
    rig = _hold(_rig(), 1.0, 1.0, 2.0)
    assert np.linalg.norm(_drawn_normal(rig)) == pytest.approx(1.0)


def test_the_visible_lean_is_the_physical_one_through_the_gain():
    rig = _hold(_rig(limit=math.radians(25), visual_gain=0.5), 0.0, 1.0, 5.0)
    *_, angle = rig.board_rotation()
    assert math.degrees(angle) == pytest.approx(12.5, abs=0.05)


def test_the_visible_lean_can_be_scaled_without_changing_the_physics():
    """A gain on the drawn lean is a look, not a rule."""
    subtle = _rig(limit=math.radians(25), visual_gain=0.25)
    plain = _rig(limit=math.radians(25), visual_gain=0.5)
    _hold(subtle, 0.0, 1.0, 5.0)
    _hold(plain, 0.0, 1.0, 5.0)
    assert subtle.board_rotation()[3] == pytest.approx(plain.board_rotation()[3] / 2.0)
    assert np.allclose(subtle.gravity_direction(), plain.gravity_direction())


def test_the_drawn_board_shows_the_player_lean_and_not_the_base_lean():
    """The base lean is a constant, and a constant needs no feedback; the drawn
    lean is the channel that says how hard the board is being pushed."""
    *_, angle = _rig(base=math.radians(20)).board_rotation()
    assert angle == pytest.approx(0.0)


# -- steering frame ------------------------------------------------------------

def test_the_lean_follows_the_camera_axes_it_was_given():
    """Under the yawed isometric view, "right" is screen-right, not world +X."""
    diagonal = (math.sqrt(0.5), 0.0, math.sqrt(0.5))
    rig = _rig(right_axis=diagonal, forward_axis=(math.sqrt(0.5), 0.0, -math.sqrt(0.5)))
    _hold(rig, 0.0, 1.0, 5.0)
    direction = rig.gravity_direction()
    assert direction[0] > 0 and direction[2] > 0    # leaned along the screen axis


# -- the board as something with mass ------------------------------------------
#
# A board thrown to full deflection in a sixth of a second reads as a pinball
# flipper. These are the properties that make it read as a table instead.

def test_the_combined_lean_is_bounded_not_each_axis_separately():
    """A full diagonal must not exceed the limit.

    Bounding pitch and roll apart lets the two together reach the limit times
    root two -- 26 degrees each drawing as 35 -- so the limit did not limit.
    """
    rig = _hold(_rig(limit=math.radians(20)), 1.0, 1.0, 10.0)
    combined = math.degrees(math.atan(math.hypot(math.tan(rig.pitch),
                                                 math.tan(rig.roll))))
    assert combined <= 20.0 + 1e-6


def test_a_diagonal_lean_keeps_its_direction_while_it_is_bounded():
    """Bounding the pair must not turn a diagonal into something else.

    The direction is the *gradient's* -- what gravity is given and what the
    board is drawn at -- so it is the tangents that keep their ratio rather than
    the angles.
    """
    rig = _hold(_rig(limit=math.radians(20)), 1.0, 0.5, 10.0)
    assert math.tan(rig.pitch) / math.tan(rig.roll) == pytest.approx(2.0, rel=1e-6)


def test_the_board_takes_a_moment_to_reach_a_lean():
    """Hands do not snap a table to full deflection."""
    rig = _rig()
    _hold(rig, 0.0, 1.0, 0.08)
    assert rig.roll < rig.limit * 0.5


def test_the_board_gets_there_in_the_end():
    rig = _hold(_rig(), 0.0, 1.0, 3.0)
    assert rig.roll == pytest.approx(rig.limit, rel=1e-3)


def test_the_board_eases_in_rather_than_moving_at_one_rate():
    """Second order: it accelerates into the lean, so there is weight to feel.

    A constant rate covers the same ground in each equal slice of time; a board
    with mass covers less at first and more once it is moving.
    """
    # Measured over the first tenth of the rise rather than over fixed tenths
    # of a second: the board answers in about a quarter of a second now, so two
    # 0.1 s windows straddle the whole movement and the second is the tail of
    # it rather than the middle.
    rig = _rig()
    first = _hold(rig, 0.0, 1.0, 0.04).roll
    before = rig.roll
    second = _hold(rig, 0.0, 1.0, 0.04).roll - before
    assert second > first * 1.2, \
        'first 40 ms moved %.4f rad, second %.4f' % (first, second)


def test_the_board_eases_out_rather_than_stopping_dead():
    rig = _hold(_rig(), 0.0, 1.0, 2.0)
    _hold(rig, 0.0, 0.0, 0.08)
    assert rig.roll > rig.limit * 0.5


def test_the_board_still_never_passes_the_limit_however_long_it_is_held():
    """Something with momentum must not overshoot the stop."""
    rig = _rig(limit=math.radians(18))
    for _ in range(2000):
        rig.update(1 / 120.0, 0.0, 1.0)
        assert rig.roll <= math.radians(18) + 1e-9


def test_the_board_does_not_ring_after_a_release():
    """A spring that oscillated would be a table nobody could aim."""
    rig = _hold(_rig(), 0.0, 1.0, 2.0)
    crossings = 0
    was = rig.roll
    for _ in range(600):
        rig.update(1 / 120.0, 0.0, 0.0)
        if (rig.roll < 0) != (was < 0) and abs(rig.roll) > 1e-6:
            crossings += 1
        was = rig.roll
    assert crossings == 0


def test_the_weight_does_not_depend_on_the_frame_rate():
    slow = _hold(_rig(), 0.0, 1.0, 0.3, dt=1 / 30.0)
    fast = _hold(_rig(), 0.0, 1.0, 0.3, dt=1 / 240.0)
    assert slow.roll == pytest.approx(fast.roll, rel=0.05)


# -- what a viewer sees --------------------------------------------------------

def test_the_drawn_lean_is_a_hint_of_the_physical_one():
    """A small tilt across a large surface already reads as a large tilt."""
    rig = _hold(_rig(limit=math.radians(20)), 0.0, 1.0, 5.0)
    assert math.degrees(rig.board_rotation()[3]) < 10.0


def test_a_far_corner_does_not_swing_across_the_screen():
    """The world heaving twenty metres about a stationary ball is what made the
    board read as a flipper."""
    rig = _hold(_rig(), 1.0, 1.0, 5.0)
    span = 36.0                                  # a typical board, corner to corner
    assert span * math.sin(rig.board_rotation()[3]) < 2 * 4.0     # under two cells
