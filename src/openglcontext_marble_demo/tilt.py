"""The board's lean — the control model, with no marble and no window in it.

The player does not push the marble; the player **tilts the board**, and gravity
does the rest.  A :class:`TiltRig` is the whole of that: held input goes in, and a
gravity direction and a board rotation come out.

Three things make the lean what a player feels:

**It is an angle, not an impulse.**  Holding a key leans the board toward what
that key asks for, and it stops at ``limit``.  A tap of thirty milliseconds leans
it a little; a hold leans it fully; letting go returns it to level.  Every size of
input in between exists, which is the difference between steering and nudging.

**It has weight.**  The lean is a *critically damped* second-order system rather
than something that moves at a constant rate: it accelerates into a lean and
eases out of one, which is what a heavy table tilted by hand does and what a
constant rate cannot imitate.  Critically damped because a table that rang after
being let go is a table nobody could aim, and because such a system never
overshoots — so the limit holds without being clamped to.

**The drawn lean is a hint of it.**  A small tilt across a large surface already
reads as a large tilt: drawn degree for degree, a board thirty-six metres across
swings a far corner twenty metres and the world heaves around a ball that never
moves.  ``visual_gain`` is what separates what the physics uses from what the
eye is given.

**It is expressed as a gradient.**  A tilted plane is described by its slope in
two horizontal directions, and gravity's pull along that plane is exactly that
gradient.  So a lean of ``theta`` about one axis is ``tan(theta)`` of horizontal
pull per unit of down, two axes add as vectors, and the direction that comes out
is a true unit gravity vector for a board leaning that far.

**It is measured in the camera's frame.**  ``forward_axis`` and ``right_axis``
are ground-plane directions the caller computes from where the camera sits, so
"right" is right on the screen under a yawed view rather than world +X.

On top of the player's lean sits ``base``, a constant lean along
``downhill_axis``, which is what makes the marble roll toward the finish when
nobody touches anything.

Nothing here reads a clock, opens a window or touches the physics: it is handed
its own time step, so how the controls respond is a question a test can ask.
"""
import math

import numpy as np

__all__ = ['TiltRig']

_UP = np.array([0.0, 1.0, 0.0])

#: Default lean toward the finish, in radians -- what makes the marble roll.
BASE = math.radians(12)
#: Default limit on the player's own lean, on either axis.
LIMIT = math.radians(25)
#: How quickly the board answers, as the natural frequency of its lean, in
#: radians a second.  A critically damped system reaches about 95% of a step in
#: ``4.75 / frequency`` seconds, so 8 is a little over half a second to full
#: deflection: an appreciable moment, which is what makes it feel like something
#: being moved rather than something being switched.
#:
#: Settling back to level is the slower of the two, because a board that snapped
#: flat the instant a key lifted would feel twitchy to let go of.
STIFFNESS = 8.0
SETTLE = 5.5

#: How much of the physical lean is drawn.  The whole of it heaves a large board
#: about; a third of it reads as a tilt without the world swinging past the ball.
VISUAL_GAIN = 0.32


def _clip(value):
    return max(-1.0, min(1.0, float(value)))


def _demand(amount, limit):
    """The lean ``amount`` in [-1, 1] asks for, as an angle.

    ``amount`` scales the board's **pull** rather than its angle: half a stick is
    half the sideways gravity, which is the quantity the marble answers to and
    the one a player is really asking about.  Scaling the angle instead would
    make a half-and-full diagonal fall in a direction that was neither.
    """
    return math.atan(math.tan(limit) * _clip(amount))


def _step(value, rate, wanted, frequency, dt):
    """One semi-implicit step of a critically damped spring toward ``wanted``.

    Semi-implicit -- the rate is moved first and the value follows it -- because
    it stays stable at the step sizes a slow frame gives, where the explicit
    form of the same spring grows instead of settling.

    Critical damping is ``2 * frequency``: the quickest approach that does not
    overshoot, which is what keeps the limit a limit without clamping to it and
    what keeps a released board from ringing.
    """
    rate += (frequency * frequency * (wanted - value)
             - 2.0 * frequency * rate) * dt
    value += rate * dt
    return value, rate


def _unit(vector):
    vector = np.asarray(vector, dtype='d')
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else vector


def _bounded(pitch, roll, limit):
    """``(pitch, roll)`` scaled down together so their combined lean fits.

    The lean a board is *drawn* at, and the one gravity gets, is the two axes
    together — ``atan`` of the gradient's length.  Bounding each axis on its own
    lets a full diagonal reach the limit times root two, which is a limit that
    does not limit: 26 degrees on each axis drew as 35.  Scaling the pair keeps
    the direction the player asked for and only takes away the excess.
    """
    combined = math.hypot(math.tan(pitch), math.tan(roll))
    ceiling = math.tan(limit)
    if combined <= ceiling or combined < 1e-12:
        return pitch, roll
    scale = ceiling / combined
    return (math.atan(math.tan(pitch) * scale),
            math.atan(math.tan(roll) * scale))


class TiltRig:
    """A board that leans, and the gravity that leaning produces.

    ``base``, ``limit``, ``rate`` and ``recover`` are radians (and radians per
    second); ``forward_axis``, ``right_axis`` and ``downhill_axis`` are
    ground-plane directions in world space.

    ``visual_gain`` scales only :meth:`board_rotation` — how far the board is
    *drawn* leaning — leaving the physics alone, so the lean can be shown subtly
    or exaggerated without changing how the game plays.
    """

    def __init__(self, base=BASE, limit=LIMIT, stiffness=STIFFNESS,
                 settle=SETTLE, forward_axis=(0.0, 0.0, -1.0),
                 right_axis=(1.0, 0.0, 0.0), downhill_axis=(0.0, 0.0, 1.0),
                 visual_gain=VISUAL_GAIN):
        self.base = float(base)
        self.limit = float(limit)
        self.stiffness = float(stiffness)
        self.settle = float(settle)
        self.forward_axis = _unit(forward_axis)
        self.right_axis = _unit(right_axis)
        self.downhill_axis = _unit(downhill_axis)
        self.visual_gain = float(visual_gain)

        #: How far the board is leaning along ``forward_axis``, in radians.
        self.pitch = 0.0
        #: How far the board is leaning along ``right_axis``, in radians.
        self.roll = 0.0
        #: How fast each is moving, which is where the weight lives.
        self.pitch_rate = 0.0
        self.roll_rate = 0.0

    # -- per-frame ------------------------------------------------------
    def update(self, dt, forward=0.0, right=0.0):
        """Lean the board toward what ``forward``/``right`` in [-1, 1] ask for.

        A critically damped step toward the demanded lean: the board accelerates
        into it and eases out, and never passes it.  Pushing toward a lean uses
        ``stiffness`` and letting go uses ``settle``, so a board can be leaned
        smartly and come back at its own pace.  Pushing the *other* way is a
        push, not a settling — reversing is as quick as leaning was.
        """
        if dt <= 0.0:
            return
        wanted_pitch = _demand(forward, self.limit)
        wanted_roll = _demand(right, self.limit)
        wanted_pitch, wanted_roll = _bounded(wanted_pitch, wanted_roll, self.limit)
        moving = bool(wanted_pitch or wanted_roll)
        frequency = self.stiffness if moving else self.settle
        self.pitch, self.pitch_rate = _step(
            self.pitch, self.pitch_rate, wanted_pitch, frequency, dt)
        self.roll, self.roll_rate = _step(
            self.roll, self.roll_rate, wanted_roll, frequency, dt)

    def level(self):
        """Return the board to level at once (a respawn, a new board)."""
        self.pitch = self.roll = 0.0
        self.pitch_rate = self.roll_rate = 0.0

    # -- what the board leans into --------------------------------------
    def gradient(self):
        """The board's slope as a horizontal vector: base lean plus player lean.

        Its length is the tangent of the total lean, and its direction is the way
        the board falls — which is the direction gravity pulls along the board.
        """
        return (self.downhill_axis * math.tan(self.base)
                + self.forward_axis * math.tan(self.pitch)
                + self.right_axis * math.tan(self.roll))

    def gravity_direction(self):
        """The unit gravity direction for the current lean.

        Written straight onto ``world.gravity.direction``; the physics re-reads it
        every step, so this is the whole of how the control reaches the marble.
        """
        return _unit(self.gradient() - _UP)

    # -- what the renderer draws ----------------------------------------
    def board_rotation(self):
        """The board's visible lean as a VRML ``(x, y, z, angle)`` axis-angle.

        The **player's** lean only, scaled by ``visual_gain``: ``base`` is a
        constant, and a constant is not feedback.  What the player needs to see is
        how hard they are pushing, which is what this shows.

        Applied to a Transform centred on the marble, so the world leans around
        the ball rather than swinging about the world origin.
        """
        gradient = (self.forward_axis * math.tan(self.pitch)
                    + self.right_axis * math.tan(self.roll))
        angle = math.atan(float(np.linalg.norm(gradient))) * self.visual_gain
        if angle <= 1e-9:
            return (0.0, 1.0, 0.0, 0.0)
        # The rotation that takes world-up to the leaning board's normal. The
        # board falls along ``gradient``, so its normal leans the *other* way,
        # and the axis is ``gradient x up`` rather than ``up x gradient`` --
        # the opposite sense draws the board leaning into the climb.
        axis = _unit(np.cross(_unit(gradient), _UP))
        return (float(axis[0]), float(axis[1]), float(axis[2]), float(angle))
