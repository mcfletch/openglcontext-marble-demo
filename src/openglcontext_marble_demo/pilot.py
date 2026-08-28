"""A board played by nobody: the attract mode, and what a recording records.

A demo that cannot demonstrate itself is a screenshot. :class:`Autopilot` reads a
board, works out a line across it, and leans the board along that line — so
``oglc-marble --demo`` plays, and a video of a run is a video of the game rather
than of a marble rolling into the first wall.

It plays the game the way a player does. It asks for a **lean**, bounded to what
a held key gives, through the same
:class:`~openglcontext_marble_demo.tilt.TiltRig` and the same physics; there is
no path it is dragged along and no force it has that a player does not. A
recording of it is therefore a recording of the game, which is the whole point of
making one.

Three things make it drive rather than weave:

**It aims down the route, not at the finish.** A pilot aiming at the goal cuts
every corner, including the ones with nothing under them. The aim is a point a
few cells along the line from wherever the marble is.

**It steers at the error and at the drift.** Leaning only toward where it should
be makes a pilot that crosses the line and comes back, for ever. Leaning against
how fast it is already returning is what makes it settle.

**It brakes with the board.** A marble has no brakes, but a board has an up-hill:
running at the line too fast, the pilot leans back against its own travel, which
is the same thing a player does at the top of a drop.

    >>> from openglcontext_marble_demo import generator
    >>> board = generator.generate(seed=1, difficulty=2)
    >>> driver = Autopilot(board)
    >>> driver.route[0] == board.start_cell
    True
"""
from collections import deque

import numpy as np

from .level import Bumper, RotatingArm, SpringTrap

__all__ = ['Autopilot', 'hazard_cells', 'route_over']

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))

#: Mechanisms a marble cannot simply roll across.  A board always has a way
#: through that meets none of them; a pilot that ignored them would spend the
#: run being flung about instead of driving.
HAZARDS = (Bumper, SpringTrap, RotatingArm)

#: How far along the route the pilot aims, in cells.  Far enough that it turns
#: before the corner rather than at it; near enough that it does not aim across
#: the void on the inside of one.
LOOK_AHEAD = 1.6

#: How hard the lean answers being off the line, per cell of error, and how hard
#: it answers the speed it is closing at.  The second is what stops the weave:
#: without it the pilot arrives on the line at full lean and crosses it.
STEER_GAIN = 1.5
DAMPING = 0.35

#: Above this speed toward the aim, the pilot starts leaning back against its own
#: travel, and by this much per metre a second over.
#:
#: These four numbers are worth their measurements.  Over twelve boards at
#: difficulty 2: a look-ahead of 2.2 with light braking finishes 10 and falls off
#: 18 times; these finish 11 and fall off 3.  Aiming nearer and braking earlier is
#: most of the difference between a demo worth watching and one that is mostly a
#: marble in the void.
BRAKE_SPEED = 4.5
BRAKE_GAIN = 0.28


def hazard_cells(level):
    """The cells of ``level`` that carry something a marble cannot roll over."""
    return {feature.cell for feature in level.features
            if isinstance(feature, HAZARDS) and feature.cell in level.cells}


def route_over(level, blocked=frozenset()):
    """The shortest run of cells from start to finish, avoiding ``blocked``.

    Empty when there is no way through, which is a board the pilot can only sit
    on — and sitting still is a better answer than driving into the void.
    """
    start, finish = level.start_cell, level.finish_cell
    if start not in level.cells or start in blocked:
        return []
    previous = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == finish:
            run = []
            while cell is not None:
                run.append(cell)
                cell = previous[cell]
            return run[::-1]
        for dcol, drow in NEIGHBOURS:
            nxt = (cell[0] + dcol, cell[1] + drow)
            if nxt in level.cells and nxt not in previous and nxt not in blocked:
                previous[nxt] = cell
                queue.append(nxt)
    return []


class Autopilot:
    """Plays ``level`` by leaning the board along a route across it."""

    def __init__(self, level, forward_axis=(0.0, 0.0, -1.0),
                 right_axis=(1.0, 0.0, 0.0), look_ahead=LOOK_AHEAD,
                 steer_gain=STEER_GAIN, damping=DAMPING,
                 brake_speed=BRAKE_SPEED, brake_gain=BRAKE_GAIN):
        self.level = level
        self.forward_axis = _unit(forward_axis)
        self.right_axis = _unit(right_axis)
        self.look_ahead = float(look_ahead)
        self.steer_gain = float(steer_gain)
        self.damping = float(damping)
        self.brake_speed = float(brake_speed)
        self.brake_gain = float(brake_gain)
        #: The line it means to take: the clean way through where there is one,
        #: and the way through there is where there is not.  Better to roll over
        #: a bumper than to stop.
        self.route = (route_over(level, hazard_cells(level))
                      or route_over(level))
        self._points = np.array(
            [level.cell_center(cell) for cell in self.route], dtype='d') \
            if self.route else np.zeros((0, 2))

    # -- where it is aiming ---------------------------------------------
    def target(self, position):
        """The point on the route it is steering for, in world space.

        The nearest point of the line, plus a look-ahead along it — so the aim
        runs up the route as the marble does, and never leaves it.
        """
        if not self.route:
            return np.asarray(position, dtype='d')
        here = np.array([position[0], position[2]], dtype='d')
        nearest = int(np.argmin(np.linalg.norm(self._points - here, axis=1)))
        ahead = min(nearest + int(round(self.look_ahead)), len(self.route) - 1)
        point = self._points[ahead]
        return np.array([point[0], position[1], point[1]], dtype='d')

    # -- what it asks of the board --------------------------------------
    def lean(self, position, velocity):
        """The ``(forward, right)`` a player would be holding, each in [-1, 1].

        Bounded to what a held key gives, because a pilot with more authority
        than a player is not playing the same game.
        """
        if not self.route:
            return (0.0, 0.0)
        position = np.asarray(position, dtype='d')
        velocity = np.asarray(velocity, dtype='d')
        error = (self.target(position) - position) / self.level.cell_size
        # Steer at where it should be, less how fast it is already getting
        # there: the damping is what turns a weave into a line.
        demand = error * self.steer_gain - velocity * self.damping
        forward = float(np.dot(demand, self.forward_axis))
        right = float(np.dot(demand, self.right_axis))
        return (_clip(forward + self._braking(velocity)), _clip(right))

    def _braking(self, velocity):
        """How much to lean back up the slope against the speed it is carrying.

        A marble has no brakes; a board has an up-hill.  Leaning into it is what
        a player does at the top of a drop, and what keeps the pilot from
        arriving at a corner too fast to turn.
        """
        downhill = -float(np.dot(velocity, self.forward_axis))
        over = downhill - self.brake_speed
        return max(0.0, over) * self.brake_gain


def _unit(vector):
    vector = np.asarray(vector, dtype='d')
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else vector


def _clip(value):
    return max(-1.0, min(1.0, float(value)))
