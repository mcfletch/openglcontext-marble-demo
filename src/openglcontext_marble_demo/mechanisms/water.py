"""A pool with a plug in the bottom of it.

A marble that rolls in sinks: gravity is most of the way off inside the water and
the drag is high, so it goes down at a walking pace with the board still playing
around it.  At the bottom is a flap, and the flap gives when the marble arrives.
It swings down on its hinge, the marble drops through, and below the pool the
water lets go of it — full gravity, no drag, rolling as it always did.

Nothing on the surface says the bottom will open, which is what the pool is for:
the first crossing is a discovery and every later one is a choice, because the
way down is quick and the way round is not.

**A board wants somewhere under the plug to land.**  The marble leaves through
the floor and keeps falling, so a pool put over nothing drops the player past
:attr:`Level.kill_y` and back to their checkpoint.  Placed over a lower run it is
a shortcut; placed over the void it is a hole.

The cell the pool occupies is its own: :meth:`Water.owned_cells` claims it, so no
flat tile is built across the top.  What is built is a rim around the four edges
(the tops flush with the surrounding floor, so the marble rolls in), the water
column, the flap, and two trigger volumes -- one filling the pool, one just above
the flap.

    >>> pool = Water(cell=(2, 1))
    >>> pool.owned_cells()
    {(2, 1)}
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from .. import render
from . import mechanism

__all__ = ['Water']

#: How much of the world's gravity still reaches a body in the water.
SINK_GRAVITY = 0.25
#: How hard the water holds a body back, per second, on top of the world's own
#: linear damping.  Together with :data:`SINK_GRAVITY` it settles a marble at
#: about 1.4 m/s, where two metres of open air have it doing 6.
SINK_DAMPING = 1.5
#: How close to the flap a marble must come for it to give, in metres: the depth
#: of the trigger volume lying on top of it, so the flap goes as the marble
#: settles onto it rather than while it is still on its way down.
PLUG_REACH = 0.25
#: How far outside the pool a body has to be before the water gives back what it
#: took, in metres.  A marble across, so one bobbing at the surface stays wet and
#: one that has dropped out of the bottom is dry.
DRY_MARGIN = 1.0

WATER_COLOR = (0.15, 0.45, 0.62)
RIM_COLOR = (0.30, 0.33, 0.38)
PLUG_COLOR = (0.42, 0.38, 0.30)


@mechanism('water')
@dataclass
class Water:
    """A pool one cell across, ``depth`` metres deep, with a hinged flap for a floor.

    ``sink_gravity`` and ``sink_damping`` are what the water does to a body in
    it: the fraction of gravity it still feels, and the extra linear damping in
    per-second units.  Lower gravity with higher damping is a slow, even descent;
    raising either makes the pool shallower to cross in time.

    ``open_time`` is how long the flap takes to swing its quarter turn once the
    marble reaches it, in seconds.  It swings once and stays open for the rest of
    the run.

    ``rim`` is the width of the lip around the pool, in metres, and
    ``plug_thickness`` the thickness of the flap.  The clear opening is the cell
    less two rims.
    """
    cell: tuple[int, int]
    depth: float = 3.0
    sink_gravity: float = SINK_GRAVITY
    sink_damping: float = SINK_DAMPING
    open_time: float = 0.8
    rim: float = 0.3
    plug_thickness: float = 0.25

    def owned_cells(self) -> Any:
        return {self.cell}

    def site(self, level: Any) -> Any:
        """Where this pool is in world units, given the level it belongs to."""
        x, z = level.cell_center(self.cell)
        surface = level.cells[self.cell]
        return _Site(x=x, z=z, surface=surface, floor=surface - self.depth,
                     span=level.cell_size - 2.0 * self.rim,
                     cell_size=level.cell_size)

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        site = self.site(level)
        material = index[level.surface]
        self._build_rim(scene, site, material, result)
        self._build_water(scene, site)
        self._build_pool(scene, site, result)
        self._build_plug(scene, site, material, result)

    # -- the parts ------------------------------------------------------
    def _build_rim(self, scene: Any, site: Any, material: Any, result: Any) -> None:
        """Four static walls inside the cell edges, tops flush with the floor around.

        They are what keeps the marble in the pool rather than dropping out of the
        side of it: the tiles of the neighbouring cells are a metre thick and the
        pool is deeper than that, so below them the sides are open.
        """
        cs, rim, span = site.cell_size, self.rim, site.span
        offset = (cs - rim) / 2.0
        walls = ((site.x + offset, site.z, (rim, self.depth, cs)),
                 (site.x - offset, site.z, (rim, self.depth, cs)),
                 (site.x, site.z + offset, (span, self.depth, rim)),
                 (site.x, site.z - offset, (span, self.depth, rim)))
        for wx, wz, size in walls:
            body = scene.add_box(size=size, position=(wx, site.middle, wz),
                                 color=RIM_COLOR, dynamic=False, material=material)
            body.transform.children[0].appearance = render.color_appearance(
                RIM_COLOR, metallic=0.1, roughness=0.7)
            result.feature_bodies.append(body)

    def _build_water(self, scene: Any, site: Any) -> None:
        """The water itself: a translucent column, drawn and not collided with."""
        column = basenodes.Box(size=(site.span, self.depth, site.span))
        appearance = basenodes.Appearance(material=PBRMaterial(
            baseColor=WATER_COLOR, metallic=0.0, roughness=0.08,
            transparency=0.55, alphaMode='BLEND', doubleSided=True))
        scene.children.append(Transform(
            translation=(site.x, site.middle, site.z),
            children=[basenodes.Shape(geometry=column, appearance=appearance)]))

    def _build_pool(self, scene: Any, site: Any, result: Any) -> None:
        """The volume that does the sinking, and the :class:`_Pool` behind it."""
        half = site.span / 2.0
        pool = _Pool(scene.world,
                     low=(site.x - half, site.floor, site.z - half),
                     high=(site.x + half, site.surface, site.z + half),
                     gravity_factor=self.sink_gravity, damping=self.sink_damping)
        trigger = scene.add_trigger_box(
            size=(site.span, self.depth, site.span),
            position=(site.x, site.middle, site.z), color=WATER_COLOR)
        result.effects[trigger.index] = pool.take
        result.feature_bodies.append(trigger)
        result.animators.append(pool)

    def _build_plug(self, scene: Any, site: Any, material: Any, result: Any) -> None:
        """The flap, its hinge animation, and the trigger that lets it go."""
        plug = Plug(hinge=(site.x - site.span / 2.0,
                           site.floor - self.plug_thickness / 2.0, site.z),
                    reach=site.span / 2.0, duration=self.open_time)
        shut, _ = plug.pose(0.0)
        body = _kinematic_box(scene, size=(site.span, self.plug_thickness, site.span),
                              position=shut, color=PLUG_COLOR, material_index=material)
        trigger = scene.add_trigger_box(
            size=(site.span, PLUG_REACH, site.span),
            position=(site.x, site.floor + PLUG_REACH / 2.0, site.z), color=PLUG_COLOR)
        result.effects[trigger.index] = plug.release
        result.animators.append(KinematicAnimator(scene.world, body.index, plug.pose))
        result.resettable.append(plug)
        result.feature_bodies.append(trigger)
        result.feature_bodies.append(body)


@dataclass(frozen=True)
class _Site:
    """Where one pool sits, in world units, so the parts are built from one measure."""
    x: float
    z: float
    #: Top of the water, level with the floor of the cells around it.
    surface: float
    #: Top of the flap: the bottom of the pool while it is shut.
    floor: float
    #: The clear opening across, which is the cell less a rim on each side.
    span: float
    cell_size: float

    @property
    def middle(self) -> Any:
        """Half way down, where anything spanning the whole depth is centred."""
        return (self.surface + self.floor) / 2.0


class _Pool:
    """What the water does to a body inside it, and undoes when it leaves.

    The change is made from the pool's trigger effect and taken back from
    :meth:`update`, which the game calls once a frame alongside its animators.
    The restore is by *position* rather than by a trigger's exit event, so a body
    that leaves the pool any way at all -- out of the bottom, back out of the
    top, respawned at a checkpoint half a board away -- gets its gravity and its
    drag back.
    """

    def __init__(self, world: Any, low: Any, high: Any, gravity_factor: Any, damping: Any,
                 margin: Any=DRY_MARGIN) -> None:
        self.world = world
        self.low = np.asarray(low, dtype='d') - margin
        self.high = np.asarray(high, dtype='d') + margin
        self.gravity_factor = gravity_factor
        self.damping = damping
        #: body index -> the gravity factor and damping it arrived with.
        self.held: dict[int, tuple[float, float]] = {}

    def take(self, world: Any, body: Any) -> None:
        """Slow ``body`` down, remembering what it was before the water had it."""
        if body in self.held:
            return
        self.held[body] = (float(world.gravity_factor[body]),
                           float(world.linear_damping[body]))
        world.gravity_factor[body] = self.gravity_factor
        world.linear_damping[body] = self.damping

    def holds(self, position: Any) -> Any:
        """True while ``position`` is inside the pool, plus the dry margin."""
        return bool(np.all(position >= self.low) and np.all(position <= self.high))

    def update(self, dt: float) -> None:  # noqa: ARG002 the animator protocol passes the frame step to update()
        """Give back what the water took from anything that has left it."""
        for body in [b for b in self.held if not self.holds(self.world.position[b])]:
            gravity_factor, damping = self.held.pop(body)
            self.world.gravity_factor[body] = gravity_factor
            self.world.linear_damping[body] = damping


class Plug:
    """The flap in the floor of the pool: shut, then a quarter turn about its hinge.

    ``hinge`` is the centre of the hinged edge and ``reach`` how far the flap
    extends from it, so the flap's centre is ``reach`` along +X while shut and
    swings down and back as the angle opens.  The pose is a function of the
    animator's clock, and :meth:`release` marks the moment on that clock rather
    than on the world's, so the swing lasts ``duration`` seconds of the time the
    animator is actually given.
    """

    def __init__(self, hinge: Any, reach: Any, duration: Any) -> None:
        self.hinge = tuple(float(v) for v in hinge)
        self.reach = float(reach)
        self.duration = float(duration)
        #: When the gate last let go, or None while it is shut.
        self.released_at: float | None = None
        self._now = 0.0

    def reset(self, world: Any=None) -> None:  # noqa: ARG002 the resettable protocol passes the world to reset()
        """Shut the flap again, so a restarted run meets the same surprise."""
        self.released_at = None

    def release(self, world: Any, body: Any) -> None:
        """Let the flap go, and wake the body that reached it so it feels the drop."""
        if self.released_at is None:
            self.released_at = self._now
        world.wake(body)

    @property
    def angle(self) -> Any:
        """How far open the flap is, in radians, from shut to a quarter turn."""
        if self.released_at is None:
            return 0.0
        share = (self._now - self.released_at) / self.duration
        return math.pi / 2.0 * min(1.0, max(0.0, share))

    def pose(self, t: float) -> Any:
        """Where the flap is at animator time ``t``: ``(position, quaternion)``."""
        self._now = t
        angle = self.angle
        x, y, z = self.hinge
        return ((x + self.reach * math.cos(angle), y - self.reach * math.sin(angle), z),
                tuple(mathutil.quat_from_axis_angle((0.0, 0.0, 1.0), -angle)))


def _kinematic_box(scene: Any, size: Any, position: Any, color: Any, material_index: Any) -> Any:
    """Add a kinematic box to a ``DemoScene``: render Transform plus physics body.

    ``DemoScene`` offers static and dynamic bodies; a flap the mechanism drives
    itself is neither, so it is assembled here from the same node types the scene
    uses and registered with the scene's physics manager and render children.
    """
    shape = scene.world.add_shape(model.Shape.box(size))
    transform = Transform(translation=tuple(position),
                          children=[basenodes.Shape(
                              geometry=basenodes.Box(size=size),
                              appearance=render.color_appearance(color, metallic=0.3,
                                                                 roughness=0.5))])
    body = PhysicsBody(transform, model.Motion(type=model.KINEMATIC),
                       model.Collider(shape=shape, physicsMaterial=material_index))
    scene.manager.add(body)
    scene.children.append(transform)
    return body
