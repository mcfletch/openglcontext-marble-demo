"""A plank that dips toward whatever weight is standing on it.

A :class:`Seesaw` is a single kinematic plank, ``length`` rows long, hinged at
the row where it meets whatever comes after it. Its tilt is not on a
schedule: every frame it looks at how far the dynamic bodies over it still
have to go to reach the hinge, and dips the near end down under them, up to
``max_tilt``. Empty, or with everything on it already at the hinge, it settles
flat.

The hinge is what a crossing plays against. It never moves — the plank's
*far* end is always exactly flush with whatever is beyond it — so nothing a
plank does can ever leave a lip to climb at the way on. What it can do is dig
a hole behind a body that lingers: the tilt chases its target with a lag
(:attr:`Seesaw.response`), so a marble that crosses quickly is at the hinge
before the plank has caught up to where it started, and barely feels it. One
that dawdles gives the near end time to sink fully under its own weight, which
turns the whole plank into a climb it has to carry itself out of — a hole its
own hesitation dug, that eases the closer it gets to the hinge.

**The plank pivots about its far edge, not its middle.** A point at signed
distance ``d`` *before* the hinge (``d`` negative, since the hinge is the
origin) sits ``|d| * sin(tilt)`` below the hinge's height for a positive tilt.
The body itself has to move to keep pivoting about a fixed edge rather than
its own centre — turning about your own foot swings your middle out along an
arc, which is the same geometry :class:`~openglcontext_marble_demo.mechanisms.
lever.Lever` uses to lay its paddle over its foot rather than its waist.

Driving it is the technique every kinematic mechanism in this game uses:
:class:`~omi_physics.kinematic.KinematicAnimator` is handed a ``pose(t)``
function and sets the body's velocity to reach that pose, rather than
teleporting it there, so a marble on the plank feels it move rather than
being shoved by a body that jumped. What is unusual here is that ``pose``
does not describe a schedule: it reads the world's own body positions afresh
on every call, which is what lets one mechanism answer to as many marbles as
are on it at once.
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from .. import render
from . import mechanism

__all__ = ['Seesaw']

_UP = np.array([0.0, 1.0, 0.0])

#: A varnished-timber colour, distinct from the stone/ice/foundry floors it
#: sits among.
_COLOUR = (0.62, 0.46, 0.28)


def _footprint(cell: Any, facing: Any, length: Any, width: Any) -> Any:
    """Every grid cell the plank covers: ``length`` rows by ``width`` across.

    ``cell`` is the row nearest the entry, matching
    :class:`~openglcontext_marble_demo.mechanisms.pegs.PegBoard`'s convention,
    so a plank is described the same way a peg board is.
    """
    left = (facing[1], -facing[0])
    half = width // 2
    return {(cell[0] + facing[0] * row + left[0] * offset,
            cell[1] + facing[1] * row + left[1] * offset)
           for row in range(length) for offset in range(-half, width - half)}


def _kinematic_box(scene: Any, size: Any, position: Any, color: Any, material_index: Any) -> Any:
    """Add a kinematic box to a ``DemoScene``: render ``Transform`` plus body.

    ``DemoScene`` offers only static and dynamic bodies; a plank the mechanism
    turns itself is neither, so it is assembled here from the same node types
    the scene uses and registered with the scene's physics manager and render
    children — exactly as
    :mod:`~openglcontext_marble_demo.mechanisms.water` builds the flap in a
    pool's floor.
    """
    shape = scene.world.add_shape(model.Shape.box(size))
    transform = Transform(translation=tuple(position),
                          children=[basenodes.Shape(
                              geometry=basenodes.Box(size=size),
                              appearance=render.color_appearance(
                                  color, metallic=0.15, roughness=0.6))])
    body = PhysicsBody(transform, model.Motion(type=model.KINEMATIC),
                       model.Collider(shape=shape, physicsMaterial=material_index))
    scene.manager.add(body)
    scene.children.append(transform)
    return body


class _Plank:
    """The reactive half of a :class:`Seesaw`: which way it leans, and how fast.

    :meth:`pose` is what a :class:`~omi_physics.kinematic.KinematicAnimator`
    drives the plank's body with. ``hinge`` is fixed for the plank's life;
    the body's own position swings on an arc around it as :attr:`tilt`
    changes, the way a paddle turning over its foot carries its middle out
    with it.
    """

    def __init__(self, world: Any, hinge: Any, along: Any, length: Any, half_width: Any,
                max_tilt: Any, response: Any) -> None:
        self.world = world
        self.hinge = np.asarray(hinge, dtype='d')
        self.along = along
        #: The horizontal axis the plank rocks about: across the direction of
        #: travel, so rotating about it changes height along ``along`` and
        #: leaves height unchanged across it.
        self.perp = np.cross(_UP, along)
        self.length = length
        self.half_width = half_width
        self.max_tilt = max_tilt
        self.response = response
        #: The body's centre, before the hinge, at zero tilt.
        self._rest_offset = -(length / 2.0) * along
        #: How far the near (pre-hinge) end has dipped, in radians; always
        #: zero or positive. The rotation this drives is about ``-tilt``
        #: (see :meth:`pose`): a point *behind* the hinge, which is where the
        #: whole plank lives, is carried down by a rotation in the opposite
        #: sense to one that would carry a point *ahead* of it down.
        self.tilt = 0.0
        self._t = 0.0

    def reset(self, world: Any=None) -> None:  # noqa: ARG002 the resettable protocol passes the world to reset()
        """Level the plank, so a restarted run meets it flat again."""
        self.tilt = 0.0
        self._t = 0.0

    def _target(self) -> Any:
        """How far the plank should dip, from how far its load still has to
        go to reach the hinge; always zero or positive."""
        world = self.world
        onboard = world.inv_mass > 0.0
        if not np.any(onboard):
            return 0.0
        rel = world.position[onboard] - self.hinge
        along = rel @ self.along
        across = np.abs(rel @ self.perp)
        # A half-metre of slack on both sides, so a body still arriving
        # through the mouth or already over the hinge is already felt — a
        # plank that only reacts once a marble is fully aboard would jerk.
        held = (along <= 0.5) & (along >= -self.length - 0.5) & (across <= self.half_width + 0.5)
        if not np.any(held):
            return 0.0
        # ``along`` is zero or negative everywhere on the plank, most negative
        # at the far (entry) end, so flipping its sign turns "how far from
        # the hinge" into "how far dipped", already in [0, 1] once divided by
        # the plank's own length.
        share = -float(np.mean(along[held])) / self.length
        return max(0.0, min(1.0, share)) * self.max_tilt

    def pose(self, t: float) -> Any:
        """Where the plank is at animator time ``t``: ``(position, quaternion)``."""
        dt = t - self._t
        self._t = t
        if dt > 0:
            rate = 1.0 if self.response <= 0 else min(1.0, dt / self.response)
            self.tilt += (self._target() - self.tilt) * rate
        quaternion = mathutil.quat_from_axis_angle(tuple(self.perp), -self.tilt)
        offset = mathutil.quat_rotate(np.asarray(quaternion), self._rest_offset)
        position = self.hinge + offset
        return tuple(position), tuple(quaternion)


@mechanism('seesaw')
@dataclass
class Seesaw:
    """A plank ``length`` rows long, hinged where it meets what comes after it.

    ``cell`` is the row nearest the entry; ``facing`` the direction of travel
    along it. ``max_tilt`` is how far over the plank dips, in degrees, when a
    body is at the far (entry) end of it; ``response`` is roughly how many
    seconds the actual tilt takes to close most of the way to that — see the
    module docstring for what the two of them do together.

    Every cell the plank tiles has to already be in ``level.cells`` at one
    height before this is built: the piece (or the hand) that placed the
    plank lays the floor, and the plank only claims it, exactly as
    :class:`~openglcontext_marble_demo.level.Elevator` claims the one cell of
    its own platform.
    """
    cell: tuple[int, int]
    facing: tuple[int, int] = (0, 1)
    length: int = 5
    width: int = 3
    max_tilt: Any = 14.0
    response: float = 0.5
    thickness: float = 0.5

    def owned_cells(self) -> Any:
        return _footprint(self.cell, self.facing, self.length, self.width)

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        cs = level.cell_size
        base = level.cells[self.cell]
        along = np.array([self.facing[0], 0.0, self.facing[1]], dtype='d')
        last_cell = (self.cell[0] + self.facing[0] * (self.length - 1),
                    self.cell[1] + self.facing[1] * (self.length - 1))
        lx, lz = level.cell_center(last_cell)
        # The hinge sits at the plank's far face -- half a cell beyond the
        # last row's centre -- which is exactly the seam with whatever the
        # piece lays next.  Pivoting there is what keeps that seam flush.
        hinge = np.array([lx, base - self.thickness / 2.0, lz], dtype='d') \
            + along * (cs / 2.0)
        full_length = self.length * cs

        dcol, _drow = self.facing
        size = ((full_length, self.thickness, self.width * cs) if dcol
               else (self.width * cs, self.thickness, full_length))
        rest_position = hinge - (full_length / 2.0) * along
        body = _kinematic_box(scene, size, tuple(rest_position), _COLOUR,
                              index[level.surface])

        plank = _Plank(scene.world, hinge=hinge, along=along,
                       length=full_length, half_width=self.width * cs / 2.0,
                       max_tilt=math.radians(self.max_tilt), response=self.response)
        result.animators.append(KinematicAnimator(scene.world, body.index, plank.pose))
        result.feature_bodies.append(body)
        result.resettable.append(plank)
