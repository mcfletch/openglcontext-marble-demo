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
few cells along the line from wherever the marble is — and only as far along as
it can be steered to in a straight line, because a cell *on* the route is not the
same as a cell the marble can reach from here. Round a right angle in a one-cell
corridor the cell two along is diagonally past a corner, and a pilot that aims at
it leans the marble into the wall and holds it there until the clock runs out.

**It drives and it steers, and they are two different things.** The lean is built
from two parts and added: a *throttle* along the route, which holds a cruising
speed, and a *steering* term across it, which closes the gap to the line. The
board is level until somebody leans it, so nothing else is going to move the
marble — a pilot that only corrected sideways would sit where it was put.

**It steers at the error and at the drift.** Leaning only toward where it should
be makes a pilot that crosses the line and comes back, for ever. Leaning against
how fast it is already returning is what makes it settle.

**It slows for corners.** The throttle asks for less where the route turns inside
the next few cells, because arriving at a right angle at full speed is how a
marble ends up in the wall on the outside of it. Braking is the same term with
its sign turned round: over the cruise it leans back, under it leans on.

    >>> from openglcontext_marble_demo import generator
    >>> board = generator.generate(seed=1, difficulty=2)
    >>> driver = Autopilot(board)
    >>> driver.route[0] == board.start_cell
    True
"""
import math
from collections import deque

import numpy as np

from .level import Bumper, RotatingArm, SpringTrap
from .pieces import MAX_STEP

__all__ = ['Autopilot', 'hazard_cells', 'route_over', 'steerable']

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
STEER_GAIN = 0.75
DAMPING = 0.45

#: The speed the pilot holds along its route, and how hard it leans to hold it.
#:
#: The board does not lean by itself, so this is where the marble's motion comes
#: from: below the cruise the pilot leans on, above it the same term leans back,
#: and braking is not a separate rule.
#:
#: Six metres a second because that is about what a marble settles at down a
#: built slope, so the pilot is asking the board for the pace the board already
#: has rather than fighting it on the descents and crawling on the flats.
CRUISE_SPEED = 6.0
THROTTLE_GAIN = 0.35

#: How much of the cruise is left where the route turns a right angle inside the
#: look-ahead, and how far ahead to look for the turn, in cells.
#:
#: Arriving at a corner at full speed is how a marble ends up in the wall on the
#: outside of it.  The scale is linear in the heading change, so a gentle bend
#: costs a little and a hairpin costs the most of it.
CORNER_SPEED = 0.45
CORNER_LOOK = 3

#: The steering gain is set by how often the pilot ends up asking for everything
#: the board has.  At 1.5 it is at the stop for well over half the run, which is
#: a demo of the extremes rather than of the game; at 0.75 it is there for 3% of
#: it, and finishes the same boards with fewer falls.  A pilot should reach the
#: stop when it is in trouble, not as a matter of course.


def hazard_cells(level):
    """The cells of ``level`` that carry something a marble cannot roll over."""
    return {feature.cell for feature in level.features
            if isinstance(feature, HAZARDS) and feature.cell in level.cells}


def steerable(level, here, there, step=0.05):
    """Can a marble be steered in a straight line from ``here`` to ``there``?

    Both are ``(x, z)`` in world metres.  Every cell the line crosses has to be
    floor, and each has to share a *face* with the one before it: a line that
    leaves a cell by its corner passes between two voids, which is a fall rather
    than a route.

    Sampled rather than solved.  A grid traversal would answer the same question
    exactly, and this is called a handful of times a frame on a line two cells
    long, where twenty samples cost less than the setup would.
    """
    cells = level.cells
    size = level.cell_size
    span = (there[0] - here[0], there[1] - here[1])
    previous = None
    at = 0.0
    while at <= 1.0 + 1e-9:
        cell = (int(round((here[0] + span[0] * at) / size)),
                int(round((here[1] + span[1] * at) / size)))
        if cell != previous:
            if cell not in cells:
                return False
            if previous is not None and (abs(cell[0] - previous[0])
                                         + abs(cell[1] - previous[1])) != 1:
                return False
            previous = cell
        at += step
    return True


def route_over(level, blocked=frozenset()):
    """The shortest run of cells from start to finish, avoiding ``blocked``.

    A step up of more than :data:`~openglcontext_marble_demo.pieces.MAX_STEP` is
    not a step at all: a marble rolls down one and cannot roll up one, so a
    route that climbed it would be a line the pilot leans at for the rest of the
    run.  Down is unbounded, because falling is something a marble does well.

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
            if nxt in level.cells and nxt not in previous and nxt not in blocked \
                    and level.cells[nxt] - level.cells[cell] <= MAX_STEP + 1e-9:
                previous[nxt] = cell
                queue.append(nxt)
    return []


class Autopilot:
    """Plays ``level`` by leaning the board along a route across it."""

    def __init__(self, level, forward_axis=(0.0, 0.0, -1.0),
                 right_axis=(1.0, 0.0, 0.0), look_ahead=LOOK_AHEAD,
                 steer_gain=STEER_GAIN, damping=DAMPING,
                 cruise_speed=CRUISE_SPEED, throttle_gain=THROTTLE_GAIN,
                 corner_speed=CORNER_SPEED, corner_look=CORNER_LOOK):
        self.level = level
        self.forward_axis = _unit(forward_axis)
        self.right_axis = _unit(right_axis)
        self.look_ahead = float(look_ahead)
        self.steer_gain = float(steer_gain)
        self.damping = float(damping)
        self.cruise_speed = float(cruise_speed)
        self.throttle_gain = float(throttle_gain)
        self.corner_speed = float(corner_speed)
        self.corner_look = int(corner_look)
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

        The look-ahead is shortened where the board will not carry it.  Aiming
        at a cell the marble cannot reach in a straight line is how a pilot ends
        a run pressed into the inside of a corner: the demand never changes,
        because the thing it is steering at never gets closer.
        """
        if not self.route:
            return np.asarray(position, dtype='d')
        here = np.array([position[0], position[2]], dtype='d')
        nearest = int(np.argmin(np.linalg.norm(self._points - here, axis=1)))
        ahead = min(nearest + int(round(self.look_ahead)), len(self.route) - 1)
        while ahead > nearest and not steerable(self.level, here,
                                                self._points[ahead]):
            ahead -= 1
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
        to_aim = self.target(position) - position
        along = _flat_unit(to_aim)
        if along is None:                       # standing on the aim itself
            return (0.0, 0.0)
        across = np.array([-along[2], 0.0, along[0]])

        # The throttle is clipped before the two are added, so that a marble well
        # under the cruise cannot ask for more lean than the board has and have
        # the steering scaled away with it: falling off costs more than being
        # slow, so the throttle is what gives way.
        #
        # A guard rather than a measured gain.  It engages on 12% of frames and
        # changes no outcome at all -- the same boards, the same falls to the
        # digit -- because a marble far below the cruise is usually one that has
        # just been put back on the line, so the two conditions it arbitrates
        # between rarely happen at once.  It is here for when they do.
        drive = max(-1.0, min(1.0, self.throttle(position, velocity)))
        demand = along * drive + across * self.steering(to_aim, velocity, across)
        return _bounded(float(np.dot(demand, self.forward_axis)),
                        float(np.dot(demand, self.right_axis)))

    def throttle(self, position, velocity):
        """How hard to lean along the route: positive leans on, negative brakes.

        One term for both, because they are one thing -- the board is level, so
        going faster and slowing down are the same lever pushed either way.
        """
        along = _flat_unit(self.target(position) - position)
        if along is None:
            return 0.0
        speed = float(np.dot(velocity, along))
        return (self.cruise_for(position) - speed) * self.throttle_gain

    def steering(self, to_aim, velocity, across):
        """How hard to lean across it: at the offset, less how fast it is closing.

        The damping is what turns a weave into a line.
        """
        offset = float(np.dot(to_aim, across)) / self.level.cell_size
        drift = float(np.dot(velocity, across))
        return offset * self.steer_gain - drift * self.damping

    def cruise_for(self, position):
        """The speed to hold here: less of it where the route turns ahead.

        Measured over the next :attr:`corner_look` cells of route, as the angle
        between where it is going now and where it goes then.  A straight run
        keeps the whole cruise; a right angle keeps :attr:`corner_speed` of it.
        """
        if len(self.route) < 3:
            return self.cruise_speed
        here = np.array([position[0], position[2]], dtype='d')
        nearest = int(np.argmin(np.linalg.norm(self._points - here, axis=1)))
        turn = self._turn_after(nearest)
        eased = 1.0 - (1.0 - self.corner_speed) * min(turn / (math.pi / 2.0), 1.0)
        return self.cruise_speed * eased

    def _turn_after(self, nearest):
        """How far the route turns between here and ``corner_look`` cells on."""
        last = len(self._points) - 1
        first, mid, far = (min(nearest, last - 2),
                           min(nearest + 1, last - 1),
                           min(nearest + self.corner_look, last))
        before = _flat2(self._points[mid] - self._points[first])
        after = _flat2(self._points[far] - self._points[mid])
        if before is None or after is None:
            return 0.0
        return float(math.acos(max(-1.0, min(1.0, float(np.dot(before, after))))))


def _flat_unit(vector):
    """``vector`` flattened onto the ground and normalised, or None if it is nothing."""
    flat = np.array([vector[0], 0.0, vector[2]], dtype='d')
    length = float(np.linalg.norm(flat))
    return flat / length if length > 1e-9 else None


def _flat2(vector):
    """A 2D route step normalised, or None if the two points coincide."""
    length = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype='d') / length if length > 1e-9 else None


def _unit(vector):
    vector = np.asarray(vector, dtype='d')
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else vector


def _bounded(forward, right):
    """The pair scaled to fit in [-1, 1], keeping the direction it asked for.

    Clipping each axis on its own turns a demand of two-forward-one-right into
    one-forward-one-right -- a different direction, chosen by the clip rather
    than by the pilot, and the reason it spent well over half a run at the stop.
    """
    size = math.hypot(forward, right)
    if size <= 1.0:
        return (float(forward), float(right))
    return (float(forward / size), float(right / size))
