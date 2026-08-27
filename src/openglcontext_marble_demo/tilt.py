"""The board's lean — the control model, with no marble and no window in it.

The player does not push the marble; the player **tilts the board**, and gravity
does the rest.  A :class:`TiltRig` is the whole of that: held input goes in, and a
gravity direction and a board rotation come out.

Three things make the lean what a player feels:

**It is an angle, not an impulse.**  Holding a key moves the board toward the
lean that key asks for at ``rate`` radians a second, and it stops at ``limit``.
A tap of thirty milliseconds leans it a little; a hold leans it fully; letting go
returns it to level at ``recover``.  Every size of input in between exists, which
is the difference between steering and nudging.

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
#: Default radians a second the lean moves toward what is held, and back to
#: level when nothing is.  Leaning in is the quicker of the two: a board that
#: snapped flat the instant a key lifted would feel twitchy to let go of.
RATE = math.radians(160)
RECOVER = math.radians(110)


def _unit(vector):
    vector = np.asarray(vector, dtype='d')
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else vector


def _approach(value, wanted, step):
    """Move ``value`` toward ``wanted`` by at most ``step``, without overshooting."""
    gap = wanted - value
    if abs(gap) <= step:
        return wanted
    return value + math.copysign(step, gap)


class TiltRig:
    """A board that leans, and the gravity that leaning produces.

    ``base``, ``limit``, ``rate`` and ``recover`` are radians (and radians per
    second); ``forward_axis``, ``right_axis`` and ``downhill_axis`` are
    ground-plane directions in world space.

    ``visual_gain`` scales only :meth:`board_rotation` — how far the board is
    *drawn* leaning — leaving the physics alone, so the lean can be shown subtly
    or exaggerated without changing how the game plays.
    """

    def __init__(self, base=BASE, limit=LIMIT, rate=RATE, recover=RECOVER,
                 forward_axis=(0.0, 0.0, -1.0), right_axis=(1.0, 0.0, 0.0),
                 downhill_axis=(0.0, 0.0, 1.0), visual_gain=1.0):
        self.base = float(base)
        self.limit = float(limit)
        self.rate = float(rate)
        self.recover = float(recover)
        self.forward_axis = _unit(forward_axis)
        self.right_axis = _unit(right_axis)
        self.downhill_axis = _unit(downhill_axis)
        self.visual_gain = float(visual_gain)

        #: How far the board is leaning along ``forward_axis``, in radians.
        self.pitch = 0.0
        #: How far the board is leaning along ``right_axis``, in radians.
        self.roll = 0.0

    # -- per-frame ------------------------------------------------------
    def update(self, dt, forward=0.0, right=0.0):
        """Move the lean toward what ``forward``/``right`` in [-1, 1] ask for.

        Moving toward a lean uses ``rate``; returning to level uses ``recover``,
        so a board can be pushed quickly and settle slowly.  Pushing the *other*
        way is a push, not a recovery — the whole move is at ``rate``, so
        reversing is as sharp as leaning was.
        """
        if dt <= 0.0:
            return
        self.pitch = self._axis(self.pitch, forward, dt)
        self.roll = self._axis(self.roll, right, dt)

    def _axis(self, current, demand, dt):
        wanted = self.limit * max(-1.0, min(1.0, float(demand)))
        rate = self.recover if wanted == 0.0 else self.rate
        return _approach(current, wanted, rate * dt)

    def level(self):
        """Return the board to level at once (a respawn, a new board)."""
        self.pitch = self.roll = 0.0

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
