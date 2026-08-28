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

import numpy as np
from omi_physics import mathutil, model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from . import materials, render
from .track import TrackMap

CELL_SIZE = 4.0


def _world_direction(grid_dir):
    """A grid ``(dcol, drow)`` step as a unit world vector in the XZ plane."""
    v = np.array([grid_dir[0], 0.0, grid_dir[1]], dtype='d')
    n = np.linalg.norm(v)
    return v / n if n else v


def _spawn_kinematic_box(scene, size, position, color, material_index):
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

    def owned_cells(self):
        return set()

    def build(self, scene, level, index, result):
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
class Ramp:
    """A sloped tile that speeds the marble up (and can launch it into a jump).

    The tile is tilted so its far edge (in ``direction``) is raised by ``rise``;
    rolling onto it, a boost trigger brings the marble's speed *up to* ``boost_speed``
    along ``direction`` (a cap, never a brake).  A ``launch`` ramp additionally pops
    the marble upward so a fast approach clears the tiles ahead in a real arc.
    """
    cell: tuple[int, int]
    direction: tuple[int, int] = (0, 1)
    rise: float = 0.9
    boost_speed: float = 8.0
    launch: bool = False
    launch_up: float = 5.0

    def owned_cells(self):
        return {self.cell}

    def build(self, scene, level, index, result):
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        cs = level.cell_size
        wdir = _world_direction(self.direction)

        # Tilt about the horizontal axis perpendicular to travel (up × dir) so the
        # +direction edge rises by ``rise`` over the cell length.
        angle = math.atan2(self.rise, cs)
        axis = np.cross((0.0, 1.0, 0.0), wdir)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        color = _ramp_color(self.launch)
        tile = scene.add_box(size=(cs, 0.4, cs), position=(x, base + self.rise / 2.0, z),
                             color=color, dynamic=False, material=index[level.surface],
                             rotation=(axis[0], axis[1], axis[2], angle))
        tile.transform.children[0].appearance = render.color_appearance(
            color, metallic=0.2, roughness=0.45)

        trigger = scene.add_trigger_box(
            size=(cs * 0.9, 2.0, cs * 0.9), position=(x, base + 1.0, z), color=color)
        result.feature_bodies.append(trigger)
        result.effects[trigger.index] = self._boost_effect(wdir)

    def _boost_effect(self, wdir):
        boost_speed, launch, launch_up = self.boost_speed, self.launch, self.launch_up

        def effect(world, marble):
            v = world.linear_velocity[marble]
            along = float(np.dot(v, wdir))
            mass = world.mass[marble]
            if along < boost_speed:                      # cap, never a brake
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

    def owned_cells(self):
        return set()

    def build(self, scene, level, index, result):
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


def _ramp_color(launch):
    return (0.85, 0.45, 0.2) if launch else (0.5, 0.55, 0.62)


@dataclass
class Bumper:
    """A springy post: a static rubber obstacle the marble bounces off with energy.

    All the behaviour is in the material — a high restitution returns the marble's
    speed, and the controller's speed-kill exempts elastic surfaces — so a bumper is
    simply a static body of ``rubber_pad``.
    """
    cell: tuple[int, int]
    radius: float = 0.4
    height: float = 0.9

    def owned_cells(self):
        return set()

    def build(self, scene, level, index, result):
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

    def owned_cells(self):
        return set()

    def build(self, scene, level, index, result):
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

    def _effect(self):
        delta_v = np.asarray(self.impulse, dtype='d')
        rearm = self.rearm
        state = {"last": -1e9}

        def effect(world, marble):
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

    def owned_cells(self):
        return {self.cell}          # replaces the flat tile with the moving platform

    def build(self, scene, level, index, result):
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        size = (level.cell_size * 0.9, self.thickness, level.cell_size * 0.9)
        top_offset = self.thickness / 2.0        # keep the platform top at ``base`` low
        body = _spawn_kinematic_box(scene, size, (x, base - top_offset, z),
                                    (0.3, 0.7, 0.85), index[level.surface])
        travel, period, y0 = self.travel, self.period, base - top_offset

        def pose(t):
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

    def owned_cells(self):
        return set()                # sweeps above the tile; the tile stays

    def build(self, scene, level, index, result):
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        y = base + self.clearance + self.thickness / 2.0
        size = (self.length, self.thickness, self.thickness)
        body = _spawn_kinematic_box(scene, size, (x, y, z),
                                    (0.9, 0.6, 0.15), index[level.surface])
        rate = self.rpm * 2.0 * math.pi / 60.0   # rad/s

        def pose(t):
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


@dataclass
class Level:
    name: str
    cells: dict                          # (col, row) -> surface height (top Y)
    start_cell: tuple[int, int]
    finish_cell: tuple[int, int]
    time_limit: float
    features: list[object] = field(default_factory=list)
    cell_size: float = CELL_SIZE
    kill_y: float = -8.0
    respawn_delay: float = 2.0
    surface: str = "stone"
    # Per-cell surface override (cell -> material name); cells absent here use
    # ``surface``.  Lets a track mix stone, metal, ice, and rubber patches — a
    # visual *and* physical (friction) variety.
    cell_surfaces: dict = field(default_factory=dict)
    seed: int | None = None
    difficulty: int = 1

    def surface_of(self, cell):
        return self.cell_surfaces.get(cell, self.surface)

    def cell_center(self, cell):
        col, row = cell
        return (col * self.cell_size, row * self.cell_size)

    def track_map(self):
        return TrackMap(self.cells, cell_size=self.cell_size)

    def marble_start(self, radius=0.5):
        """World position a marble of ``radius`` rests at over the start cell."""
        x, z = self.cell_center(self.start_cell)
        return (x, self.cells[self.start_cell] + radius + 0.1, z)

    def build_into(self, scene, index):
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
