"""A lever thrown by a hard enough blow, and the door it opens.

A :class:`Lever` is a paddle standing on a cell.  A marble that reaches it
slowly bounces off something that does not move; one that arrives above the
lever's ``hardness`` puts it over, and the paddle stays over.  That is the whole
of the mechanism: a place on the board that asks for speed rather than for aim,
so the way past it is to go back and take a run at it.

A :class:`Door` is a wall across one side of a cell that sinks into the board
when its lever is thrown.  Shut it is a barrier like any other; open it has
dropped a cell's thickness below the surface, which is both how it stops
colliding and how a player sees it go -- the leaf slides down out of the way.
Nothing is removed from the world, so the door keeps its body and its place in
the scene. A run started over with
:meth:`~openglcontext_marble_demo.game.MarbleGame.reset` shuts every door and
stands every paddle up again.

The two find each other by **channel**: a string both carry, so a board says
which lever opens which door and neither holds a reference to the other.  One
channel takes as many levers and as many doors as a board wants -- three doors
on ``'vault'`` all go down together, and any lever on ``'vault'`` opens them.

    >>> Lever(cell=(2, 0), channel='vault').hardness
    4.0

**How hard is hard enough.**  ``hardness`` is a closing speed in metres a
second: how fast the marble and the paddle were coming together along the
contact normal at the moment they met. Only the marble's blows count: anything
else on the board bounces off the paddle whatever its speed.  The world reports it on the step the
blow lands, as the ``approach`` of a contact event on the paddle, so the
number a designer writes is the speed the marble has to arrive at, and it can
be compared with the marble's own speed on the HUD.  The default, 4 m/s, is
above a marble that has trickled down onto the cell and inside what a free roll
down a couple of squares builds up.
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from omi_physics import mathutil

from .. import render
from . import mechanism

__all__ = ['Channel', 'Door', 'Lever', 'channels']

#: Which way a named side of a cell lies, as a ``(dcol, drow)`` step.
_SIDES = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}

_UP = np.array([0.0, 1.0, 0.0])

class Channel:
    """The wire between levers and the doors they open, named by a string.

    A door registers what to do when the channel is thrown; a lever throws it.
    Both reach the channel by name as they are built, so which of them a level
    lists first is not something either of them has to know.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        #: Whether any lever on this channel has gone over.
        self.thrown = False
        #: Body indices of the doors listening, in the order they were built.
        self.doors: list[Any] = []
        self._opening: list[Any] = []
        self._shutting: list[Any] = []

    def add_door(self, body_index: Any, opening: Any, shutting: Any=None) -> None:
        """Register a door: ``opening()`` runs when the channel is thrown.

        ``shutting()`` puts it back where it was, which is what a restarted run
        needs: a door left open would make the second attempt a different board.
        """
        self.doors.append(body_index)
        self._opening.append(opening)
        self._shutting.append(shutting)

    def reset(self, world: Any=None) -> None:
        """Shut every door and let the levers be thrown again."""
        for shutting in self._shutting:
            if shutting is not None:
                shutting()
        self.thrown = False

    def throw(self) -> Any:
        """Open every door on the channel; True the first time, False after."""
        if self.thrown:
            return False
        self.thrown = True
        for opening in self._opening:
            opening()
        return True

    def __repr__(self) -> Any:
        return '<Channel %r %s, %d door(s)>' % (
            self.name, 'thrown' if self.thrown else 'set', len(self.doors))


def channels(result: Any) -> Any:
    """Every channel in one build, by name.

    ``result`` is what :meth:`~openglcontext_marble_demo.level.Level.build_into`
    returns, and :attr:`~openglcontext_marble_demo.level.BuildResult.channels`
    is where mechanisms that have to find each other do it.  This is how a game
    (or a test) asks whether a board's levers have been thrown and which bodies
    its doors are.
    """
    return result.channels


def _channel(result: Any, name: str) -> Any:
    """The channel called ``name`` in ``result``, made if it is new."""
    table = channels(result)
    if name not in table:
        table[name] = Channel(name)
    return table[name]


def _facing(step: Any) -> Any:
    """A grid ``(dcol, drow)`` step as a unit world vector in the XZ plane."""
    vector = np.array([step[0], 0.0, step[1]], dtype='d')
    length = np.linalg.norm(vector)
    return vector / length if length else vector


@mechanism('lever')
@dataclass
class Lever:
    """A paddle that goes over only for a blow of at least ``hardness``.

    ``hardness`` is the closing speed of the blow in metres a second (see the
    module docstring).  Under it the paddle is a post: the marble bounces and
    the door stays shut.

    ``facing`` is the axis step the paddle faces and the way it goes over, so a
    lever on ``(1, 0)`` stands across the +X approach and lies down toward +X.
    ``throw`` is how far over it ends up, in degrees from upright; the default
    leaves it low enough to roll across.
    """
    cell: tuple[int, int]
    channel: str = 'gate'
    hardness: float = 4.0
    facing: tuple[int, int] = (1, 0)
    height: float = 1.2
    width: float = 1.6
    thickness: float = 0.25
    throw: float = 80.0

    #: Brass, so an upright lever reads as the one thing on the cell to hit.
    COLOUR = (0.85, 0.62, 0.15)

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        facing = _facing(self.facing)
        # Broad side to the approach: a paddle facing along X is thin in X.
        size = ((self.thickness, self.height, self.width) if self.facing[0]
                else (self.width, self.height, self.thickness))
        body = scene.add_box(size=size, position=(x, base + self.height / 2.0, z),
                             color=self.COLOUR, dynamic=False, material=index['metal'])
        body.transform.children[0].appearance = render.color_appearance(
            self.COLOUR, metallic=0.8, roughness=0.3)
        result.feature_bodies.append(body)
        world = scene.world
        world.report_contacts(body.index)
        if world.contact_reporting == 'off':
            world.contact_reporting = 'flagged'
        paddle = _Paddle(world, body.index, self.hardness, result,
                         _channel(result, self.channel),
                         self._going_over((x, base, z), facing))
        world.add_contact_listener(paddle.struck)
        result.resettable.append(paddle)

    def _going_over(self, foot: Any, facing: Any) -> Any:
        """A call that lays the paddle over about its foot, toward ``facing``."""
        angle = math.radians(self.throw)
        axis = np.cross(_UP, facing)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        quaternion = tuple(mathutil.quat_from_axis_angle(tuple(axis), angle))
        # It turns about its foot rather than its middle, so the middle swings
        # out along the arc with it.
        middle = tuple(np.asarray(foot, dtype='d')
                       + (math.cos(angle) * _UP + math.sin(angle) * facing)
                       * (self.height / 2.0))

        def over(world: Any, index: Any) -> None:
            world.place_body(index, position=middle, orientation=quaternion)
        return over


class _Paddle:
    """One lever's paddle: goes over for the marble's hard blow, stands up on reset.

    :meth:`struck` is a contact listener, so it runs inside the physics step
    the blow lands on and the paddle is laid over before the next step: the
    marble carries on past it.
    """

    def __init__(self, world: Any, index: int, hardness: float, result: Any,
                 channel: Any, going_over: Any) -> None:
        self.world = world
        self.index = index
        self.hardness = hardness
        self.result = result
        self.channel = channel
        self.going_over = going_over
        self.over = False
        self._upright = (tuple(float(v) for v in world.position[index]),
                         tuple(float(v) for v in world.orientation[index]))

    def struck(self, event: Any) -> None:
        """Weigh one contact on the paddle, and go over for a hard one from the marble."""
        if self.over or event.phase == 'end' or event.approach <= self.hardness:
            return
        ends = (event.a.index, event.b.index)
        if self.index not in ends or self.result.marble not in ends:
            return
        self.over = True
        self.going_over(self.world, self.index)
        self.channel.throw()

    def reset(self, world: Any = None) -> None:
        """Stand the paddle up where it was built, ready to be thrown again."""
        position, orientation = self._upright
        self.world.place_body(self.index, position=position, orientation=orientation)
        self.over = False


@mechanism('door')
@dataclass
class Door:
    """A wall across one side of a cell that sinks away when its channel is thrown.

    ``side`` is which edge of ``cell`` it stands on -- ``'N'`` (-Z), ``'S'``
    (+Z), ``'E'`` (+X) or ``'W'`` (-X) -- exactly as
    :class:`~openglcontext_marble_demo.level.Wall` reads it, so a door can be
    swapped for a rail and back.  A side the mechanism does not know is a
    :exc:`KeyError` as the board is built, rather than a barrier that turns out
    at play time to be somewhere else.
    """
    cell: tuple[int, int]
    side: str = 'E'
    channel: str = 'gate'
    height: float = 2.0
    thickness: float = 0.4
    material: str = 'metal'

    #: Cold blue against the lever's brass: the thing that opens and the thing
    #: that is opened do not look alike.
    COLOUR = (0.30, 0.45, 0.58)

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        cs = level.cell_size
        dcol, drow = _SIDES[self.side]
        px, pz = x + dcol * cs / 2.0, z + drow * cs / 2.0
        size = ((self.thickness, self.height, cs) if dcol
                else (cs, self.height, self.thickness))
        body = scene.add_box(size=size, position=(px, base + self.height / 2.0, pz),
                             color=self.COLOUR, dynamic=False,
                             material=index[self.material])
        body.transform.children[0].appearance = render.color_appearance(
            self.COLOUR, metallic=0.7, roughness=0.4)
        result.feature_bodies.append(body)

        world = scene.world
        leaf = body.index
        # A cell's tile is a unit thick, so a leaf whose top sits that far under
        # the surface is below everything the marble can reach.
        sunk = (px, base - 1.0 - self.height / 2.0, pz)
        shut = tuple(float(value) for value in world.position[leaf])

        def open_the_door() -> None:
            world.place_body(leaf, position=sunk)

        def shut_the_door() -> None:
            world.place_body(leaf, position=shut)

        channel = _channel(result, self.channel)
        channel.add_door(leaf, open_the_door, shut_the_door)
        if channel not in result.resettable:
            result.resettable.append(channel)
