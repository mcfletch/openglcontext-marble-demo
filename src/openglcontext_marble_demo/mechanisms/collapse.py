"""A patch of floor that drops away once something has stood on it too long.

A marble that crosses at pace is off the far side before the floor has taken
enough of its weight to matter; one that dawdles -- steered in and left, tipped
in and stopped, sitting in the middle working out where to go -- has put enough
time on it that the tile gives way under it.  The measure is a clock, not a
speed: two quick taps across never add up the way one long stand does, because
the reading drains at :attr:`Collapse.recover_rate` the moment nothing is on it.

**The floor is a trapdoor, not a switch.** It hinges on one edge and swings
clear over :attr:`Collapse.drop_time` once it lets go, exactly like
:class:`~openglcontext_marble_demo.mechanisms.water.Water`'s plug -- a body
resting on it is carried down and out of the way rather than left to clip
through a tile that vanished.  It swings once and stays open: the piece that
built it puts a graduated way down immediately beyond it, so what was a floor
becomes a hole with a slower route past it rather than the end of the run.

    >>> Collapse(cells=((3, 0), (4, 0))).owned_cells() == {(3, 0), (4, 0)}
    True
"""
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from .. import render
from ..level import UP, _world_direction
from . import mechanism

__all__ = ['Collapse', 'CollapseWatch']

PANEL_COLOR = (0.44, 0.32, 0.22)


@mechanism('collapse')
@dataclass
class Collapse:
    """A rigid floor, ``cells`` wide, that gives way under sustained weight.

    ``direction`` is the way it hinges open, a grid step like
    :class:`~openglcontext_marble_demo.level.Ramp`'s: the edge behind that
    direction stays put and the floor swings down and clear across it.

    ``hold_time`` is how many seconds of continuous weight the floor bears
    before it gives, and ``recover_rate`` is how fast that reading falls once
    the weight is gone, in seconds per second -- a floor that drained no
    faster than it filled would fail a marble that crossed it twice in short
    order for no reason it could feel.  ``drop_time`` is how long the swing
    itself takes once the floor lets go, and ``catch_height`` is how far above
    the floor, in metres, a body still counts as standing on it -- roughly a
    marble's diameter, so an arc that clears the tile is a real way past it.

    ``carry_speed`` is what a body over the floor is moving at, along
    ``direction``, the moment it gives -- a body that triggered the floor by
    dawdling is by definition not carrying it on its own, and a trapdoor a
    body is standing near the hinge of barely tips it at all: the way down is
    built beyond the floor's own far edge, and something has to put the body
    there rather than straight through the hole it fell into.  A cap, exactly
    like :attr:`~openglcontext_marble_demo.level.Ramp.boost_speed`, never a
    brake on a body already carrying more.  ``launch_speed`` is a matching lift,
    so the carry clears the near corner of what is on the other side of the
    hole rather than driving into it.
    """
    #: The cells this covers, as `(col, row)` pairs. Taken as any sequence
    #: of them, because a level gives tuples and a file gives back lists;
    #: `__post_init__` settles it to tuples so the two are one board.
    cells: Sequence[Sequence[int]]
    direction: tuple[int, int] = (1, 0)
    hold_time: Any = 0.9
    recover_rate: float = 1.5
    drop_time: float = 0.35
    catch_height: float = 1.0
    carry_speed: float = 10.0
    launch_speed: float = 4.0
    thickness: float = 0.4

    def __post_init__(self) -> None:
        # A file gives back lists where a level gave tuples; one spelling here
        # means a board that has been through the file format is the same board.
        self.cells = tuple((int(col), int(row)) for col, row in self.cells)
        self.direction = (int(self.direction[0]), int(self.direction[1]))

    def owned_cells(self) -> Any:
        return set(self.cells)

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        material = index[level.surface]
        cs = level.cell_size
        surface = level.cells[self.cells[0]]
        wdir = _world_direction(self.direction)
        axis = np.cross(UP, wdir)
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm else np.array([1.0, 0.0, 0.0])

        cols = [col for col, _row in self.cells]
        rows = [row for _col, row in self.cells]
        # The box's top, not its centre, has to sit flush with the surface the
        # neighbouring tiles present, exactly as a Ramp's does.
        center = np.array([(min(cols) + max(cols)) * cs / 2.0,
                           surface - self.thickness / 2.0,
                           (min(rows) + max(rows)) * cs / 2.0])
        span_cells = ((max(cols) - min(cols) + 1) if self.direction[0]
                     else (max(rows) - min(rows) + 1))
        across_cells = ((max(rows) - min(rows) + 1) if self.direction[0]
                        else (max(cols) - min(cols) + 1))
        reach = span_cells * cs / 2.0
        hinge = center - wdir * reach

        panel = _Panel(hinge=hinge, axis=axis, wdir=wdir, reach=reach,
                       duration=self.drop_time, carry_speed=self.carry_speed,
                       launch_speed=self.launch_speed)
        size = ((span_cells * cs, self.thickness, across_cells * cs)
               if self.direction[0] else (across_cells * cs, self.thickness, span_cells * cs))
        position, _shut = panel.pose(0.0)
        body = _kinematic_box(scene, size=size, position=position, color=PANEL_COLOR,
                              material_index=material)
        result.feature_bodies.append(body)
        result.animators.append(KinematicAnimator(scene.world, body.index, panel.pose))
        result.resettable.append(panel)

        watch = CollapseWatch(scene.world, self.cells, surface, cs,
                              self.hold_time, self.recover_rate, self.catch_height, panel)
        result.animators.append(watch)
        result.resettable.append(watch)


class CollapseWatch:
    """How long a body has stood on the floor, and when it gives way.

    Modelled on :class:`~openglcontext_marble_demo.mechanisms.burner.BurnerHeat`:
    a bounding-box test over the floor's own cells, a per-body reading that
    rises while something is on it and falls once it is not.  Reaching
    ``hold_time`` releases the :class:`_Panel` it watches instead of destroying
    what tripped it -- the fall through the floor is the trap here, not the
    marble.
    """

    def __init__(self, world: Any, cells: Any, surface: Any, cell_size: float, hold_time: Any, recover_rate: Any,
                catch_height: Any, panel: Any) -> None:
        self.world = world
        self.surface = float(surface)
        self.hold_time = float(hold_time)
        self.recover_rate = float(recover_rate)
        self.catch_height = float(catch_height)
        self.panel = panel
        #: Body index -> seconds of continuous weight taken so far.
        self.held: dict[int, float] = {}
        cs = float(cell_size)
        half = cs / 2.0
        cols = [col for col, _row in cells]
        rows = [row for _col, row in cells]
        self._west = min(cols) * cs - half
        self._east = max(cols) * cs + half
        self._north = min(rows) * cs - half
        self._south = max(rows) * cs + half

    def holds(self, point: Any) -> Any:
        """Is world ``point`` down on the floor, within reach of its weight?"""
        x, y, z = point[0], point[1], point[2]
        return (self._west <= x <= self._east and self._north <= z <= self._south
               and y < self.surface + self.catch_height)

    def bodies_inside(self) -> Any:
        """The indices of the dynamic bodies currently pressing on the floor."""
        world = self.world
        position = world.position
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        near = ((world.inv_mass > 0.0) & (y < self.surface + self.catch_height)
               & (x >= self._west) & (x <= self._east)
               & (z >= self._north) & (z <= self._south))
        return [int(i) for i in np.flatnonzero(near)]

    def update(self, dt: float=0.0) -> None:
        """Take ``dt`` seconds of weight from everything on the floor, give it
        back from everything that has left -- and let the floor go once
        anything has taken enough of it."""
        if self.panel.released_at is not None:
            return
        inside = set(self.bodies_inside())
        for index in inside:
            self.held[index] = min(self.hold_time, self.held.get(index, 0.0) + dt)
            if self.held[index] >= self.hold_time:
                self.panel.release(self.world, index)
                return
        for index in list(self.held):
            if index in inside:
                continue
            eased = self.held[index] - self.recover_rate * dt
            if eased <= 0.0:
                del self.held[index]
            else:
                self.held[index] = eased

    def reset(self, world: Any=None) -> None:
        """Nothing has stood on the floor yet: a restarted run starts clean."""
        self.held.clear()


class _Panel:
    """The collapsing floor itself: shut, then a quarter turn about its hinge.

    Mirrors :class:`~openglcontext_marble_demo.mechanisms.water.Plug`: the
    pose is a function of the animator's own clock, and :meth:`release` marks
    the moment on that clock, so the swing lasts ``duration`` seconds of the
    time the animator is actually given rather than of the game's.
    """

    def __init__(self, hinge: Any, axis: Any, wdir: Any, reach: Any, duration: Any, carry_speed: Any=0.0,
                launch_speed: Any=0.0) -> None:
        self.hinge = np.asarray(hinge, dtype='d')
        self.axis = np.asarray(axis, dtype='d')
        self.wdir = np.asarray(wdir, dtype='d')
        self.reach = float(reach)
        self.duration = float(duration)
        self.carry_speed = float(carry_speed)
        self.launch_speed = float(launch_speed)
        #: When the tile last let go, or None while it is holding.
        self.released_at: float | None = None
        self._now = 0.0

    def reset(self, world: Any=None) -> None:
        """Shut the floor again, so a restarted run meets the same surprise."""
        self.released_at = None

    def release(self, world: Any, body: Any) -> None:
        """Let the floor go: carry the body toward the way down, and wake it.

        The carry alone would drive the body straight at the near edge of
        whatever the way down starts with, low enough to catch that edge
        rather than the surface past it -- a body that dawdled into the hold
        has fallen further than it has travelled by the time it gets there.
        The matching lift puts it over that edge instead of into it.
        """
        if self.released_at is not None:
            return
        self.released_at = self._now
        along = float(np.dot(world.linear_velocity[body], self.wdir))
        boost = self.wdir * max(0.0, self.carry_speed - along) + UP * self.launch_speed
        if np.any(boost):
            world.apply_impulse(body, boost * world.mass[body])
        world.wake(body)

    @property
    def angle(self) -> Any:
        """How far open the floor is, in radians, from shut to a quarter turn."""
        if self.released_at is None:
            return 0.0
        share = (self._now - self.released_at) / self.duration
        return math.pi / 2.0 * min(1.0, max(0.0, share))

    def pose(self, t: float) -> Any:
        """Where the floor is at animator time ``t``: ``(position, quaternion)``."""
        self._now = t
        angle = self.angle
        position = (self.hinge + self.wdir * self.reach * math.cos(angle)
                   - UP * self.reach * math.sin(angle))
        quat = mathutil.quat_from_axis_angle(self.axis, angle)
        return tuple(position), tuple(quat)


def _kinematic_box(scene: Any, size: Any, position: Any, color: Any, material_index: Any) -> Any:
    """Add a kinematic box to a ``DemoScene``: render Transform plus physics body."""
    shape = scene.world.add_shape(model.Shape.box(size))
    transform = Transform(translation=tuple(position),
                          children=[basenodes.Shape(
                              geometry=basenodes.Box(size=size),
                              appearance=render.color_appearance(color, metallic=0.2,
                                                                 roughness=0.6))])
    body = PhysicsBody(transform, model.Motion(type=model.KINEMATIC),
                       model.Collider(shape=shape, physicsMaterial=material_index))
    scene.manager.add(body)
    scene.children.append(transform)
    return body
