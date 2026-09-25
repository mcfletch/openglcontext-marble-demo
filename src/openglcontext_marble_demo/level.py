"""Level data and the code that turns it into a live scene.

A :class:`Level` is *pure data*: the grid of track cells (``(col,row) -> surface
height``), the start/finish cells, a time limit, and a list of **features** (a
finish pad now; ramps, walls, bumpers, and mechanisms in later phases).  Keeping it
data means a level can be generated, hand-authored, serialized, and unit-tested
without any GL.

:meth:`Level.build_into` realizes the data in a
:class:`~OpenGLContext.physics.demo.DemoScene` — one static floor tile per cell,
plus each feature's bodies — and returns a :class:`BuildResult` carrying the
:class:`~openglcontext_marble_demo.track.TrackMap` the controller needs and handles
to the special bodies (the finish trigger, etc.).

Each feature is a small dataclass with a ``build(scene, level, index, result)``
method, so adding a mechanism is adding a class — the level just lists it.
"""
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from . import materials, render
from .track import TrackMap

CELL_SIZE = 4.0

UP = np.array([0.0, 1.0, 0.0])


def _world_direction(grid_dir: Any) -> Any:
    """A grid ``(dcol, drow)`` step as a unit world vector in the XZ plane."""
    v = np.array([grid_dir[0], 0.0, grid_dir[1]], dtype='d')
    n = np.linalg.norm(v)
    return v / n if n else v


def _spawn_kinematic_box(scene: Any, size: Any, position: Any, color: Any, material_index: Any) -> Any:
    """Add a kinematic box to a ``DemoScene`` (render Transform + physics body).

    ``DemoScene`` only exposes static/dynamic bodies, so elevators and arms build
    their kinematic body here from the same node types the scene uses, and register
    it with the scene's physics manager and render children.  Returns the
    :class:`PhysicsBody` (whose ``.index`` the animator drives).
    """
    shape_idx = scene.world.add_shape(model.Shape.box(size))
    geometry = basenodes.Box(size=size)
    # A PBR appearance like every other body in the game: this was the one
    # place a VRML Material was left, in a scene that is otherwise all PBR.
    appearance = render.color_appearance(color, metallic=0.1, roughness=0.5)
    transform = Transform(translation=tuple(position),
                          children=[basenodes.Shape(geometry=geometry,
                                                    appearance=appearance)])
    body = PhysicsBody(transform, model.Motion(type=model.KINEMATIC),
                       model.Collider(shape=shape_idx, physicsMaterial=material_index))
    scene.manager.add(body)
    scene.children.append(transform)
    return body


# -- features ------------------------------------------------------------
#
# Every feature exposes ``owned_cells()`` (cells whose default flat tile it
# replaces — empty for things that sit on top of the track) and ``build(scene,
# level, index, result)`` which creates its bodies and, for anything that acts on
# the marble, records an *effect* in ``result.effects`` keyed by its trigger body.

@dataclass
class Finish:
    """The goal pad: a sensor over one cell that ends the run on overlap."""
    cell: tuple[int, int]

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        surface = level.cells[self.cell]
        # A trigger box is invisible, so lay a glowing pad on the cell to mark the
        # goal, then the sensor above it that actually ends the run.
        pad = scene.add_box(size=(level.cell_size * 0.9, 0.1, level.cell_size * 0.9),
                            position=(x, surface + 0.06, z), color=(0.2, 0.95, 0.5),
                            dynamic=False, material=index[level.surface])
        pad.transform.children[0].appearance = render.color_appearance(
            (0.2, 0.95, 0.5), roughness=0.4)
        body = scene.add_trigger_box(
            size=(level.cell_size * 0.9, 2.0, level.cell_size * 0.9),
            position=(x, surface + 1.0, z), color=(0.2, 0.95, 0.5))
        result.finish_body = body
        result.feature_bodies.append(body)


@dataclass
class Gate:
    """A waypoint the finish waits for: somewhere you have to have been.

    A board whose finish takes anyone who reaches it has no route in it -- every
    way across is as good as every other, and the shape a designer built is
    decoration.  A gate is a sensor like the finish, and the game will not end a
    run until every one of them has been passed.

    ``order`` is what a story means by them, and not a rule the board enforces: a
    player who finds the second one first has found it.  What it is for is
    telling a designer which is which, and drawing them in the order they were
    meant to be met.
    """
    cell: tuple[int, int]
    order: int = 0

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        surface = level.cells[self.cell]
        colour = (0.95, 0.75, 0.15)
        pad = scene.add_box(size=(level.cell_size * 0.9, 0.08, level.cell_size * 0.9),
                            position=(x, surface + 0.05, z), color=colour,
                            dynamic=False, material=index[level.surface])
        pad.transform.children[0].appearance = render.color_appearance(
            colour, metallic=0.4, roughness=0.35)
        body = scene.add_trigger_box(
            size=(level.cell_size * 0.9, 2.0, level.cell_size * 0.9),
            position=(x, surface + 1.0, z), color=colour)
        result.gate_bodies[body.index] = self
        result.feature_bodies.append(body)


@dataclass
class Ramp:
    """A sloped tile: the cell's floor tilted, and optionally a boost over it.

    The face runs from the cell's own height at the near edge to ``rise`` above
    it at the far edge (in ``direction``), so the cells on either side are met at
    *their* heights and a marble crosses a join rather than a step.  ``direction``
    is a grid step — ``(0, 1)``, ``(-1, 0)`` and so on — which is the way the
    ground rises.

    ``boost_speed`` brings the marble's speed *up to* that many metres a second
    along ``direction`` while it is over the tile (a cap, never a brake), and
    ``None`` is a ramp that is only a slope.  A ``launch`` ramp additionally pops
    the marble upward so a fast approach clears the tiles ahead in a real arc.
    """
    cell: tuple[int, int]
    direction: tuple[int, int] = (0, 1)
    rise: float = 0.9
    boost_speed: float | None = 8.0
    launch: bool = False
    launch_up: float = 5.0

    #: How thick the sloped tile is.  Only its top is ever touched; the rest is
    #: there so that nothing arriving fast passes through it.
    THICKNESS = 0.4

    def owned_cells(self) -> Any:
        return {self.cell}

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        cs = level.cell_size
        wdir = _world_direction(self.direction)
        color = _ramp_color(self.launch)

        # The face is the hypotenuse of the cell and the rise, so that its
        # *horizontal* span is exactly one cell: a box a cell long, tilted, would
        # fall short by the cosine and leave a notch at each end.
        tilt = math.atan2(self.rise, cs)
        face = math.hypot(cs, self.rise)
        # Turning about ``up × direction`` by a positive angle carries the far
        # edge down, so a rise is the negative of it.
        axis = np.cross(UP, wdir)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        # The box's *top* is the face, half a thickness along the face's normal
        # from the box's centre -- so the centre sits that far under the middle
        # of the slope, which is at ``base + rise / 2``.
        normal = UP * math.cos(tilt) - wdir * math.sin(tilt)
        centre = np.array([x, base + self.rise / 2.0, z]) - normal * (self.THICKNESS / 2.0)
        along = np.abs(wdir)
        size = (cs + along[0] * (face - cs), self.THICKNESS,
                cs + along[2] * (face - cs))
        tile = scene.add_box(size=size, position=tuple(centre),
                             color=color, dynamic=False, material=index[level.surface],
                             rotation=(axis[0], axis[1], axis[2], -tilt))
        tile.transform.children[0].appearance = render.color_appearance(
            color, metallic=0.2, roughness=0.45)

        if self.boost_speed is None and not self.launch:
            return                       # a slope, and nothing else
        trigger = scene.add_trigger_box(
            size=(cs * 0.9, 2.0, cs * 0.9), position=(x, base + 1.0, z), color=color)
        result.feature_bodies.append(trigger)
        result.effects[trigger.index] = self._boost_effect(wdir)

    def _boost_effect(self, wdir: Any) -> Any:
        boost_speed, launch, launch_up = self.boost_speed, self.launch, self.launch_up

        def effect(world: Any, marble: Any) -> None:
            mass = world.mass[marble]
            if boost_speed is not None:
                along = float(np.dot(world.linear_velocity[marble], wdir))
                if along < boost_speed:                  # cap, never a brake
                    world.apply_impulse(marble, wdir * (boost_speed - along) * mass)
            if launch:
                world.apply_impulse(marble, (0.0, launch_up * mass, 0.0))
        return effect


@dataclass
class Wall:
    """A low static barrier along one edge of a cell (a rail the marble bounces off)."""
    cell: tuple[int, int]
    side: str = "E"                      # 'N'(-Z) 'S'(+Z) 'E'(+X) 'W'(-X)
    height: float = 1.5
    thickness: float = 0.3
    material: str = "stone"

    _OFFSET = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        cs = level.cell_size
        dx, dz = self._OFFSET[self.side]
        px = x + dx * cs / 2.0
        pz = z + dz * cs / 2.0
        if dx:                            # wall runs along Z
            size = (self.thickness, self.height, cs)
        else:                             # wall runs along X
            size = (cs, self.height, self.thickness)
        body = scene.add_box(size=size, position=(px, base + self.height / 2.0, pz),
                             color=(0.35, 0.35, 0.38), dynamic=False,
                             material=index[self.material])
        body.transform.children[0].appearance = render.color_appearance(
            (0.32, 0.33, 0.36), metallic=0.1, roughness=0.7)
        result.feature_bodies.append(body)


def _ramp_color(launch: Any) -> Any:
    return (0.85, 0.45, 0.2) if launch else (0.5, 0.55, 0.62)


@dataclass
class Bumper:
    """A springy post: a static rubber obstacle the marble bounces off with energy.

    All the behaviour is in the material — a high restitution returns the marble's
    speed, and the controller's speed-kill exempts elastic surfaces — so a bumper is
    simply a static body of ``rubber_pad``.
    """
    cell: tuple[int, int]
    radius: Any = 0.4
    height: float = 0.9

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        body = scene.add_box(size=(self.radius * 2, self.height, self.radius * 2),
                             position=(x, base + self.height / 2.0, z),
                             color=(0.9, 0.2, 0.3), dynamic=False,
                             material=index["rubber_pad"])
        body.transform.children[0].appearance = render.color_appearance(
            (0.9, 0.2, 0.3), roughness=0.6)
        result.feature_bodies.append(body)


@dataclass
class SpringTrap:
    """A sensor that flings the marble when it rolls over, re-arming after a delay.

    The impulse is a target change in velocity (m/s); the effect scales it by the
    marble's mass.  Re-arm is timed off ``world.time`` so a marble parked on the pad
    is not launched every frame.
    """
    cell: tuple[int, int]
    impulse: tuple[float, float, float] = (0.0, 8.0, 0.0)
    rearm: float = 1.5

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        pad_size = 1.4          # a small pad, not a cell-wide plate
        pad = scene.add_box(size=(pad_size, 0.12, pad_size),
                            position=(x, base + 0.07, z), color=(0.95, 0.85, 0.2),
                            dynamic=False, material=index[level.surface])
        pad.transform.children[0].appearance = render.color_appearance(
            (0.95, 0.8, 0.15), metallic=0.3, roughness=0.35)
        trigger = scene.add_trigger_box(
            size=(pad_size, 1.0, pad_size),
            position=(x, base + 0.5, z), color=(0.95, 0.85, 0.2))
        result.feature_bodies.append(trigger)
        result.effects[trigger.index] = self._effect()

    def _effect(self) -> Any:
        delta_v = np.asarray(self.impulse, dtype='d')
        rearm = self.rearm
        state = {"last": -1e9}

        def effect(world: Any, marble: Any) -> None:
            if world.time - state["last"] >= rearm:
                world.apply_impulse(marble, delta_v * world.mass[marble])
                state["last"] = world.time
        return effect


@dataclass
class Elevator:
    """A kinematic platform that rides up and down on a sine, carrying the marble."""
    cell: tuple[int, int]
    travel: float = 3.0
    period: float = 3.0
    thickness: float = 0.5

    def owned_cells(self) -> Any:
        return {self.cell}          # replaces the flat tile with the moving platform

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        size = (level.cell_size * 0.9, self.thickness, level.cell_size * 0.9)
        top_offset = self.thickness / 2.0        # keep the platform top at ``base`` low
        body = _spawn_kinematic_box(scene, size, (x, base - top_offset, z),
                                    (0.3, 0.7, 0.85), index[level.surface])
        travel, period, y0 = self.travel, self.period, base - top_offset

        def pose(t: float) -> Any:
            rise = travel * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period))
            return (x, y0 + rise, z), (0.0, 0.0, 0.0, 1.0)

        result.animators.append(KinematicAnimator(scene.world, body.index, pose))
        result.feature_bodies.append(body)


@dataclass
class RotatingArm:
    """A kinematic bar spinning about the cell's vertical axis, sweeping the marble."""
    cell: tuple[int, int]
    length: float = 2.0             # short enough to steer around — a hazard, not a wall
    rpm: float = 20.0
    thickness: float = 0.3
    clearance: float = 0.3          # how high the bar floats above the surface

    def owned_cells(self) -> Any:
        return set()                # sweeps above the tile; the tile stays

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        y = base + self.clearance + self.thickness / 2.0
        size = (self.length, self.thickness, self.thickness)
        body = _spawn_kinematic_box(scene, size, (x, y, z),
                                    (0.9, 0.6, 0.15), index[level.surface])
        rate = self.rpm * 2.0 * math.pi / 60.0   # rad/s

        def pose(t: float) -> Any:
            quat = mathutil.quat_from_axis_angle((0.0, 1.0, 0.0), rate * t)
            return (x, y, z), tuple(quat)

        result.animators.append(KinematicAnimator(scene.world, body.index, pose))
        result.feature_bodies.append(body)


# -- level ---------------------------------------------------------------

@dataclass
class BuildResult:
    """What :meth:`Level.build_into` hands back to the game."""
    track: TrackMap
    finish_body: object = None
    feature_bodies: list = field(default_factory=list)
    # Trigger-body-index -> effect(world, marble_index): boost pads, spring traps,
    # etc.  The game wires one trigger listener that dispatches through this map.
    effects: dict = field(default_factory=dict)
    # KinematicAnimators for always-running mechanisms (elevators, arms); the game
    # calls each one's update(dt) before stepping the world.
    animators: list = field(default_factory=list)
    # Trigger-body-index -> the Gate it belongs to.  The game keeps the set of
    # gates still to be passed and refuses the finish until it is empty.
    gate_bodies: dict = field(default_factory=dict)
    # Anything holding state a restarted run has to forget: a thrown lever, an
    # opened plug.  Each answers to ``reset()``, and the game asks all of them.
    # A board whose levers were still thrown would be a different board from the
    # one the player started on, so the second attempt would not be a second
    # attempt at the same thing.
    resettable: list = field(default_factory=list)
    # Somewhere for mechanisms that have to find each other to do it: a lever
    # and the door it opens, named by a string both carry.  Per build rather
    # than global, so one board's channels are not reachable from another's,
    # and here rather than hung off the result by each mechanism that wants it,
    # because every mechanism that talks to another will want the same table.
    channels: dict = field(default_factory=dict)
    # The marble's body index, which the game sets once it has spawned the
    # marble. A mechanism that listens to the world from the moment it is
    # built reads it here when it hears a contact, to tell the marble from
    # anything else that strikes it.
    marble: Any = None


#: How far below the lowest tile of a board the kill plane sits.
#:
#: Two cells.  Far enough that a marble bouncing along the floor of the board is
#: never taken for one that has left it, near enough that a marble which really
#: has gone over the edge is caught rather than falling for several seconds
#: first.
KILL_MARGIN = 8.0


@dataclass
class Level:
    name: str
    cells: dict                          # (col, row) -> surface height (top Y)
    start_cell: tuple[int, int]
    finish_cell: tuple[int, int]
    time_limit: float
    #: The mechanisms on this board. Every one answers `owned_cells` and
    #: `build`; what else it is is its own, and a level never asks.
    features: list[Any] = field(default_factory=list)
    cell_size: float = CELL_SIZE
    #: The height below which a marble counts as having left the world.  Set
    #: from the board in :meth:`__post_init__` unless it is given something
    #: lower, so it is always under the deepest tile: see :data:`KILL_MARGIN`.
    kill_y: float = -8.0
    respawn_delay: float = 2.0
    surface: str = "stone"
    # Per-cell surface override (cell -> material name); cells absent here use
    # ``surface``.  Lets a track mix stone, metal, ice, and rubber patches — a
    # visual *and* physical (friction) variety.
    cell_surfaces: dict = field(default_factory=dict)
    seed: int | None = None
    difficulty: int = 1

    def __post_init__(self) -> None:
        # The kill plane goes under the board, however deep the board goes.  A
        # constant put it at -8 metres, and a board that descends 0.9 m a cell
        # passes that after nine cells of slope: measured over twelve generated
        # boards, six had floor below it and one had 134 of its 192 cells there,
        # every one of them a tile the player can see and stand on and be killed
        # for standing on.  A marble respawned onto such a cell is killed again
        # at once, which is a run that cannot be continued and cannot be
        # understood -- the plane is invisible and the tile is right there.
        if self.cells:
            self.kill_y = min(self.kill_y,
                              min(self.cells.values()) - KILL_MARGIN)

    def surface_of(self, cell: Any) -> Any:
        return self.cell_surfaces.get(cell, self.surface)

    def cell_center(self, cell: Any) -> Any:
        col, row = cell
        return (col * self.cell_size, row * self.cell_size)

    def track_map(self) -> Any:
        return TrackMap(self.cells, cell_size=self.cell_size,
                        rails=self.rails())

    def rails(self) -> Any:
        """Every ``(cell, step)`` a wall of this level stands across.

        The board's own answer to "can the marble leave this cell that way",
        which the controller needs to know whether somewhere is safe to put a
        respawned marble down on.
        """
        return frozenset(
            (feature.cell, Wall._OFFSET[feature.side])
            for feature in self.features if isinstance(feature, Wall))

    def marble_start(self, radius: Any=0.5) -> Any:
        """World position a marble of ``radius`` rests at over the start cell."""
        x, z = self.cell_center(self.start_cell)
        return (x, self.cells[self.start_cell] + radius + 0.1, z)

    def build_into(self, scene: Any, index: Any) -> Any:
        """Create floor tiles + features in ``scene``; return a :class:`BuildResult`."""
        # One shared grouted tile mesh (not the Box primitive — it supplies the
        # tangents the normal map needs) and a per-surface cached appearance.
        tile_geometry = render.tile_mesh(size=(self.cell_size, 1.0, self.cell_size))
        # Features that replace a cell's tile with their own geometry (ramps) claim
        # it here so the default flat tile is not also built underneath.
        owned = set()
        for feature in self.features:
            owned |= feature.owned_cells()

        for cell, height in self.cells.items():
            if cell in owned:
                continue
            surface_name = self.surface_of(cell)
            surface = materials.SURFACES[surface_name]
            x, z = self.cell_center(cell)
            # A 1-unit-thick tile whose top sits exactly at ``height``.
            body = scene.add_box(size=(self.cell_size, 1.0, self.cell_size),
                                 position=(x, height - 0.5, z),
                                 color=surface.base_color, dynamic=False,
                                 material=index[surface_name])
            body.transform.children[0] = basenodes.Shape(
                geometry=tile_geometry,
                appearance=render.surface_appearance(surface, grout=True))
        result = BuildResult(track=self.track_map())
        for feature in self.features:
            feature.build(scene, self, index, result)
        return result
