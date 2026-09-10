"""The peg board: a sloping field of pegs, and the slots underneath it.

A marble put in at the top works its way down through the pegs, is turned aside
by each one it meets, and arrives in one of the slots along the bottom. One slot
is a chute that shoots it onward; the rest are ordinary.

What the board is for is **odds rather than outcomes**. Where a marble ends up
is mostly chance -- two runs that start a hand's width apart part company at the
first peg they take differently -- but not only chance: entering toward one side
carries that way more often than entering toward the other, so a player who can
choose where and how fast to enter is playing for the fast slot rather than
waiting to be told which one they got. A board where the answer is always the
same is a corridor, and one where the entry makes no difference is a coin.

The board is a **slope**, which is what keeps the marble going: each cell is
tiled with a tilted slab whose top runs from that cell's height at the near edge
to the next row's at the far edge, so the whole board is one plane a marble rolls
down rather than a stair it drops off. The pegs stand on it, elastic enough to
turn a marble aside and far less so than a bumper, so it works down the board
rather than being fired back up it.

Laying one out takes two steps, because a level owns its cells and a mechanism
owns what stands on them::

    board = PegBoard(cell=(0, 0), width=3, fast_slot=1)
    level.cells.update(board.cells())
    level.features.append(board)

:meth:`PegBoard.cells` renders the heights the board wants; :meth:`PegBoard.build`
reads back whatever the level actually holds, so a hand-edited board is built as
it was edited.

    >>> board = PegBoard(cell=(0, 0), width=3, length=4, slot_length=2)
    >>> len(board.cells()) == 3 * 6
    True
    >>> board.slot_of((4.0, 0.0, 0.0))
    0
"""
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from omi_physics import model
from OpenGLContext.scenegraph import basenodes

from .. import materials, render
from ..level import CELL_SIZE, Wall
from . import mechanism

__all__ = ['PegBoard']

#: Which side of a cell a grid step names, for :class:`~...level.Wall`.
_SIDE_OF = {(0, -1): 'N', (0, 1): 'S', (1, 0): 'E', (-1, 0): 'W'}

#: How thick the sloping slabs are, in metres.
TILE_THICKNESS = 0.4

#: How much room a peg leaves beside a wall, in metres.  A marble is a metre
#: across, so anything less is a pocket to be wedged in rather than a gap to be
#: threaded.
PEG_CHANNEL = 1.15

#: How far a peg is turned from corner-on to the board, in radians.  A post
#: presents an edge to what comes at it, which is what splits a marble one way or
#: the other; the spread gives each its own angle, so a field of them is not a
#: lattice that turns every marble the same way.
PEG_YAW_SPREAD = math.radians(20)

#: How deep a peg is sunk into the tile it stands on, in metres, so a tilted
#: tile never leaves daylight under it.
PEG_SINK = 0.25

_PEG_COLOR = (0.72, 0.74, 0.78)
_CHUTE_COLOR = (0.25, 0.9, 0.55)


def _frame(facing: Any) -> Any:
    """The world axes of a board facing ``facing``, as ``(along, left)``."""
    along = np.array([facing[0], 0.0, facing[1]], dtype='d')
    left = np.array([facing[1], 0.0, -facing[0]], dtype='d')
    return along, left


@mechanism('pegs')
@dataclass
class PegBoard:
    """A field of pegs on a slope, ending in slots, one of which is fast.

    ``cell`` is where the marble comes in: the middle of the top row.  The board
    runs ``length`` rows of pegs and then ``slot_length`` rows of chute, and is
    ``width`` cells across -- one cell per slot, walled apart from each other, so
    a slot is a cell column and :attr:`fast_slot` names one of them counting from
    the left as the marble travels.

    The fast slot's chute cells carry
    :attr:`~openglcontext_marble_demo.level.Ramp.boost_speed`, which brings the
    marble *up to* that speed and never slows one already going faster.  The
    other slots are plain descents, so what they give back is whatever the marble
    brought through the pegs.

    A peg is a static post of its own material, and :attr:`peg_restitution` is
    what a hit off one feels like.  The two numbers it sits between are what
    matter: above the elastic threshold in
    :class:`~openglcontext_marble_demo.controller.MarbleController`, so a peg
    turns the marble aside rather than scrubbing its speed the way a wall does;
    well below a bumper's, so a hit does not send it back up the slope.
    """
    cell: tuple[int, int] = (0, 0)
    facing: tuple[int, int] = (0, 1)
    #: Cells across, which is also how many slots the bottom is divided into.
    width: int = 3
    #: Rows of pegs.
    length: int = 4
    #: Rows of chute below the pegs.
    slot_length: int = 2
    #: Total fall from the entry row to the far end, in metres.
    drop: float = 3.5
    #: Height of the entry row; what :meth:`cells` measures from.
    top: float = 0.0
    #: Which slot is the fast one, counting from the left.
    fast_slot: int = 1
    #: Speed the fast chute brings the marble up to, in m/s.
    boost_speed: float = 16.0
    peg_radius: float = 0.22
    peg_height: float = 1.2
    #: Distance between pegs, in metres, along and across the board.  A marble is
    #: a metre across, so the gap a pair of them leaves is what decides whether
    #: the field is threaded or jammed.
    peg_spacing: float = 2.0
    peg_restitution: float = 0.55
    peg_friction: float = 0.25
    wall_height: float = 1.5
    #: Which way each peg is turned.  Two boards with the same seed are the same
    #: board, which is what makes a run repeatable.
    seed: int = 0

    # -- shape ----------------------------------------------------------

    def rows(self) -> Any:
        """Rows the board occupies: pegs and chute together."""
        return self.length + self.slot_length

    def step(self) -> Any:
        """How far each row falls below the one before it, as a signed height."""
        return -abs(self.drop) / max(self.rows(), 1)

    def offsets(self) -> Any:
        """The cell offset of each slot, left to right as the marble travels."""
        first = self.width // 2
        return [first - index for index in range(self.width)]

    def cell_at(self, row: Any, offset: Any) -> Any:
        """The cell ``row`` rows down the board and ``offset`` cells to the left."""
        left = (self.facing[1], -self.facing[0])
        return (self.cell[0] + self.facing[0] * row + left[0] * offset,
                self.cell[1] + self.facing[1] * row + left[1] * offset)

    def cells(self) -> Any:
        """The heights the board wants, as ``{cell: height}``.

        A level lays these itself -- ``level.cells.update(board.cells())`` --
        because the cells are the track, and the track belongs to the level
        rather than to anything standing on it.
        """
        step = self.step()
        return {self.cell_at(row, offset): round(self.top + step * row, 6)
                for row in range(self.rows())
                for offset in self.offsets()}

    def owned_cells(self) -> Any:
        """Every cell the board tiles itself, so no flat one is laid under it."""
        return set(self.cells())

    def slot_cells(self, index: Any) -> Any:
        """The chute cells of slot ``index``, top to bottom."""
        offset = self.offsets()[index]
        return [self.cell_at(row, offset) for row in range(self.length, self.rows())]

    # -- where things are -----------------------------------------------

    def board_coordinates(self, position: Any, cell_size: float=CELL_SIZE) -> Any:
        """``position`` as ``(along, left)`` metres from the entry cell's centre.

        ``along`` grows down the board and ``left`` toward the marble's left, so
        a slot further left has the larger ``left``.
        """
        along_axis, left_axis = _frame(self.facing)
        origin = np.array([self.cell[0] * cell_size, 0.0,
                           self.cell[1] * cell_size], dtype='d')
        delta = np.asarray(position, dtype='d') - origin
        return float(np.dot(delta, along_axis)), float(np.dot(delta, left_axis))

    def slot_of(self, position: Any, cell_size: float=CELL_SIZE) -> Any:
        """Which slot ``position`` is over, or ``None`` if it is off the board.

        Asked across the board only: a marble still among the pegs is over the
        column it will arrive in if it holds the line it is on.
        """
        _, left = self.board_coordinates(position, cell_size)
        offset = int(round(left / cell_size))
        offsets = self.offsets()
        return offsets.index(offset) if offset in offsets else None

    def past_the_end(self, position: Any, cell_size: float=CELL_SIZE) -> Any:
        """Whether ``position`` is beyond the far end of the slots."""
        along, _ = self.board_coordinates(position, cell_size)
        return along > (self.rows() - 0.5) * cell_size

    def entry_position(self, across: Any=0.0, height: Any=0.55, cell_size: float=CELL_SIZE) -> Any:
        """A world point on the entry row, ``across`` metres to the left of centre.

        ``height`` is clearance above the board's surface, so the default is a
        marble resting on it rather than dropped onto it.  Aiming a peg board is
        choosing this and a speed: where along the top row the marble goes in.
        """
        _, left_axis = _frame(self.facing)
        origin = np.array([self.cell[0] * cell_size,
                           self.top + self.step() / 2.0 + height,
                           self.cell[1] * cell_size], dtype='d')
        return tuple(origin + left_axis * across)

    def peg_places(self, cell_size: float=CELL_SIZE) -> Any:
        """``(along, left)`` of every peg, in metres from the entry cell's centre.

        Rows alternate by half a spacing, which is what makes a field out of a
        lattice: a marble that comes straight down one gap meets a peg in the
        next row rather than another gap.  A peg near enough a wall to leave less
        than :data:`PEG_CHANNEL` beside it is left out, so the board has no
        pocket in it.
        """
        offsets = self.offsets()
        left_edge = (max(offsets) + 0.5) * cell_size
        right_edge = (min(offsets) - 0.5) * cell_size
        room = self.peg_radius * math.sqrt(2.0) + PEG_CHANNEL   # posts stand corner-on
        places = []
        row = 0
        along = self.peg_spacing
        while along <= (self.length - 0.5) * cell_size:
            shift = self.peg_spacing / 2.0 if row % 2 else 0.0
            first = math.ceil((right_edge - shift) / self.peg_spacing)
            count = int((left_edge - right_edge) / self.peg_spacing) + 2
            for step in range(first, first + count):
                across = step * self.peg_spacing + shift
                if right_edge + room <= across <= left_edge - room:
                    places.append((along, across))
            row += 1
            along += self.peg_spacing
        return places

    # -- building -------------------------------------------------------

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        self._require_cells(level)
        self._tiles(scene, level, index, result)
        self._pegs(scene, level, index, result)
        self._walls(scene, level, index, result)

    def _require_cells(self, level: Any) -> None:
        missing = sorted(set(self.cells()) - set(level.cells))
        if missing:
            raise ValueError(
                'a peg board needs its cells laid before it is built: %d of them are '
                'not in the level, starting at %r. Say level.cells.update(board.cells()) '
                'first.' % (len(missing), missing[0]))

    def _rise(self, level: Any, row: Any, offset: Any) -> Any:
        """How much the tile at ``(row, offset)`` climbs toward the next row."""
        here = level.cells[self.cell_at(row, offset)]
        ahead = level.cells.get(self.cell_at(row + 1, offset), here + self.step())
        return ahead - here

    def _tiles(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        """One tilted slab per cell, and the chute trigger under the fast slot."""
        cs = level.cell_size
        mesh = render.tile_mesh(size=(cs, TILE_THICKNESS, cs))
        along_axis, _ = _frame(self.facing)
        axis = np.cross((0.0, 1.0, 0.0), along_axis)
        fast = self.offsets()[self.fast_slot]
        for row in range(self.rows()):
            for offset in self.offsets():
                cell = self.cell_at(row, offset)
                rise = self._rise(level, row, offset)
                x, z = level.cell_center(cell)
                base = level.cells[cell]
                # Turned so the top runs from ``base`` at the near edge to
                # ``base + rise`` at the far one, and dropped by its own
                # half-thickness so it is the *top* that meets those heights.
                angle = math.atan2(-rise, cs)
                surface = materials.SURFACES[level.surface_of(cell)]
                tile = scene.add_box(
                    size=(cs, TILE_THICKNESS, cs),
                    position=(x, base + rise / 2.0
                              - TILE_THICKNESS / 2.0 / math.cos(angle), z),
                    color=surface.base_color, dynamic=False,
                    material=index[surface.name],
                    rotation=(axis[0], axis[1], axis[2], angle))
                tile.transform.children[0] = basenodes.Shape(
                    geometry=mesh,
                    appearance=render.surface_appearance(surface, grout=True))
                result.feature_bodies.append(tile)
                if row >= self.length and offset == fast:
                    self._chute(scene, level, result, cell)

    def _chute(self, scene: Any, level: Any, result: Any, cell: Any) -> None:
        """The trigger over one fast-slot cell, and the speed it gives back."""
        x, z = level.cell_center(cell)
        cs = level.cell_size
        trigger = scene.add_trigger_box(
            size=(cs * 0.9, 2.0, cs * 0.9),
            position=(x, level.cells[cell] + 1.0, z), color=_CHUTE_COLOR)
        result.feature_bodies.append(trigger)
        result.effects[trigger.index] = self._boost()

    def _boost(self) -> Any:
        """Bring the marble *up to* :attr:`boost_speed` down the board.

        A cap and never a brake, so a marble that arrived through the pegs
        already faster than the chute gives keeps what it earned.
        """
        along_axis, _ = _frame(self.facing)
        wanted = self.boost_speed

        def effect(world: Any, marble: Any) -> None:
            travelling = float(np.dot(world.linear_velocity[marble], along_axis))
            if travelling < wanted:
                world.apply_impulse(
                    marble, along_axis * (wanted - travelling) * world.mass[marble])
        return effect

    def _surface(self, level: Any, along: Any, across: Any, cell_size: float) -> Any:
        """Height of the tile top under ``(along, across)``, in metres."""
        offsets = self.offsets()
        offset = min(max(int(round(across / cell_size)), min(offsets)), max(offsets))
        row = min(max(int(round(along / cell_size)), 0), self.rows() - 1)
        base = level.cells[self.cell_at(row, offset)]
        return base + self._rise(level, row, offset) * (along / cell_size - row + 0.5)

    def _pegs(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        """The posts themselves, on a material of their own rather than a surface's."""
        material = scene.raw_material(model.Material(
            staticFriction=self.peg_friction + 0.1,
            dynamicFriction=self.peg_friction,
            restitution=self.peg_restitution))
        appearance = render.color_appearance(_PEG_COLOR, metallic=0.6, roughness=0.35)
        turns = random.Random(self.seed)
        along_axis, left_axis = _frame(self.facing)
        origin = np.array([self.cell[0] * level.cell_size, 0.0,
                           self.cell[1] * level.cell_size], dtype='d')
        size = (self.peg_radius * 2, self.peg_height, self.peg_radius * 2)
        for along, across in self.peg_places(level.cell_size):
            base = self._surface(level, along, across, level.cell_size)
            where = origin + along_axis * along + left_axis * across
            yaw = math.pi / 4.0 + turns.uniform(-PEG_YAW_SPREAD, PEG_YAW_SPREAD)
            body = scene.add_box(
                size=size,
                position=(where[0], base - PEG_SINK + self.peg_height / 2.0, where[2]),
                color=_PEG_COLOR, dynamic=False, material=material,
                rotation=(0.0, 1.0, 0.0, yaw))
            body.transform.children[0].appearance = appearance
            result.feature_bodies.append(body)

    def _walls(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        """Rails down both sides, and a divider between every pair of slots."""
        left = (self.facing[1], -self.facing[0])
        right = (-left[0], -left[1])
        offsets = self.offsets()
        for row in range(self.rows()):
            for place, offset in enumerate(offsets):
                sides = []
                if place == 0:
                    sides.append(_SIDE_OF[left])
                if place == len(offsets) - 1 or row >= self.length:
                    sides.append(_SIDE_OF[right])       # the outer rail, or a divider
                for side in sides:
                    Wall(cell=self.cell_at(row, offset), side=side,
                         height=self.wall_height).build(scene, level, index, result)
