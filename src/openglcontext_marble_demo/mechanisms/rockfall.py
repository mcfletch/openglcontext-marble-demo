"""A slope of shattered rock: the quick way down, and the one that aims you.

A rockfall is laid over cells that already descend.  It tilts each of them so
the terraces under it are a surface rather than a staircase, and scatters hard,
irregular slabs across them at angles.  The slabs are static and low, and there
is room between them, so a marble that commits arrives at the bottom -- pointing
somewhere it did not choose.  Two descents from entry points a metre apart are
two different descents, and that is the whole of what the mechanism is for.

It is not the :func:`~openglcontext_marble_demo.pieces.scatter`, which is a field
of rubber posts on the flat.  A scatter throws the marble because it is springy;
a rockfall throws it because the ground is broken, and it descends while it does
it.

The rock is hard, and that is what keeps a rockfall a scatter rather than a
stop.  :class:`~openglcontext_marble_demo.controller.MarbleController` scrubs a
marble to a tenth of its speed on a heavy sideways contact and exempts any
surface whose restitution reaches its ``elastic_restitution``.  Rock meets that
on its own terms: a steel ball off a rock face keeps most of the speed it
arrived with, where a floor built to deaden absorbs it.  :data:`ROCK` carries
the number.  Laid in ``materials``' ``stone`` instead, at restitution 0.10, the
same field takes the marble by the crash rule eight times in fourteen descents;
in :data:`ROCK` it takes it none in fifty-two, and the marble leaves as fast as
it entered.

Every slab comes from :attr:`Rockfall.seed`, so a board is the same board each
time it is opened, and :meth:`Rockfall.rocks` answers where they all lie as
plain data, with no scene built.

    >>> from openglcontext_marble_demo import mechanisms
    >>> mechanisms.registry()['rockfall'] is Rockfall
    True
"""
import math
import random
from dataclasses import dataclass

from omi_physics import model
from OpenGLContext.scenegraph import basenodes

from .. import materials, render
from . import mechanism

__all__ = ['ROCK', 'ROCK_COLOR', 'Rock', 'Rockfall']

#: The feel of broken rock: it grips, and it gives the marble its speed back.
#: The restitution is what keeps the controller's crash rule off a rockfall (see
#: the module docstring); the friction is stone's, since rock is what stone is.
ROCK = model.Material(staticFriction=0.9, dynamicFriction=0.8, restitution=0.55)

#: Linear RGB for a broken face.  The slabs keep this whatever the floor under
#: them is made of, so the rock reads as rock on an icy slope as well as a stone
#: one.
ROCK_COLOR = (0.34, 0.31, 0.28)

#: How far across a slab is, in metres, and how thick.  Slabs below about a
#: metre wedge a marble between them instead of turning it.
FOOTPRINT = (1.0, 2.2)
THICKNESS = (0.7, 1.4)

#: How far a slab leans out of the slope, in radians, and how far its high
#: corner stands above the surface, in metres.
#:
#: The stand decides whether the field is something to ride over or a wall to
#: meet: a marble of the game's radius rides half a metre, and not much more.
#: The lean is gentle against it because a slab is sunk to its stand and the
#: rest of it is under the slope -- lean it far and only the high corner is
#: above ground, an obstacle a hand's width across that most lines miss.  Held
#: under 16 degrees, a slab's whole face is out and it is as wide an obstacle as
#: it looks.
TILT = (0.10, 0.28)
STAND = (0.20, 0.55)

#: How far up or down its own row a slab may be set, as a fraction of a cell,
#: and how far it may wander from the middle of its lane across the run, as a
#: fraction of the lane.  Both short of a half, so a row stays a row and two
#: neighbouring slabs cannot trade places.
JITTER = 0.4
SWAY = 0.4


@dataclass(frozen=True)
class Rock:
    """One slab, already placed: where it sits, how big it is, and how it lies.

    ``stand`` is how far its highest corner reaches above the slope at that
    point, which is the number that says whether it is an obstacle or paving.
    """
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    axis: tuple[float, float, float]
    angle: float
    stand: float


@mechanism('rockfall')
@dataclass
class Rockfall:
    """Broken rock over a descending run of cells: a way through that aims you.

    ``cell`` is the middle of the top row and ``direction`` the way down;
    ``length`` and ``width`` are the run in cells.  The heights come from the
    level -- a rockfall makes its cells rollable and puts rock on them, and the
    piece around it decides how far they fall.

    ``density`` is slabs per cell.  Below about 1.2 the field has clear lines
    down it that a marble can take without touching anything; above about 2 the
    slabs crowd close enough to wedge one between them, and a descent ends
    stopped rather than aimed.
    """
    cell: tuple[int, int]
    direction: tuple[int, int] = (0, 1)
    length: int = 6
    width: int = 5
    seed: int = 0
    density: float = 1.5

    def cells(self):
        """Every cell of the run, top row first and across before along."""
        dcol, drow = self.direction
        acol, arow = -drow, dcol            # one step across the way down
        half = self.width // 2
        return [(self.cell[0] + dcol * step + acol * offset,
                 self.cell[1] + drow * step + arow * offset)
                for step in range(self.length)
                for offset in range(-half, self.width - half)]

    def owned_cells(self):
        """The rockfall lays its own tilted tiles, so the flat ones are not built."""
        return set(self.cells())

    # -- the rock, as data ----------------------------------------------
    def rocks(self, level):
        """Where every slab lies on ``level``, in world metres.

        A pure function of :attr:`seed` and the level's heights, so a board is
        the same board twice and the field can be measured without a scene.

        The slabs are dealt a row at a time and spaced across the width, each
        wandering only within its own lane.  Scattering them freely over the run
        instead leaves gaps wherever the luck of the draw put none, and gaps
        that line up down the slope are a clear run through the middle of it: a
        marble takes it, touches nothing, and comes out pointing the way it went
        in.
        """
        rng = random.Random(self.seed)
        cs = level.cell_size
        down, across = _ground_axes(self.direction)
        origin_x, origin_z = level.cell_center(self.cell)
        span = self.width * cs                     # how wide the run is, in metres
        per_row = max(1, round(self.width * self.density))
        lane = span / per_row
        placed = []
        for step in range(self.length):
            for slot in range(per_row):
                beside = ((slot + 0.5) * lane - span / 2.0
                          + rng.uniform(-SWAY, SWAY) * lane)
                along = rng.uniform(-JITTER, JITTER) * cs
                cell = self._cell_at(step, beside, cs)
                if cell not in level.cells:
                    continue
                run = step * cs + along
                placed.append(self._one_rock(
                    rng, level, cell, along,
                    (origin_x + down[0] * run + across[0] * beside,
                     origin_z + down[2] * run + across[2] * beside)))
        return placed

    def _cell_at(self, step, beside, cell_size):
        """The cell ``step`` rows down the run and ``beside`` metres off its middle."""
        dcol, drow = self.direction
        half = self.width // 2
        offset = int(math.floor(beside / cell_size + self.width / 2.0))
        offset = min(max(offset, 0), self.width - 1) - half
        return (self.cell[0] + dcol * step - drow * offset,
                self.cell[1] + drow * step + dcol * offset)

    def _one_rock(self, rng, level, cell, in_cell, where):
        """One slab standing at ``where`` (an x, z pair), lying however the rng says.

        ``in_cell`` is how far down its own cell the slab sits, which with
        ``cell`` is what fixes the height of the slope under it.
        """
        size = (rng.uniform(*FOOTPRINT), rng.uniform(*THICKNESS),
                rng.uniform(*FOOTPRINT))
        angle = rng.uniform(*TILT)
        lie = rng.uniform(0.0, 2.0 * math.pi)          # which way it leans
        stand = rng.uniform(*STAND)
        surface = self._surface_at(level, cell, in_cell)
        return Rock(position=(where[0], surface + stand - _reach(size, lie, angle),
                              where[1]),
                    size=size, axis=(math.cos(lie), 0.0, math.sin(lie)),
                    angle=angle, stand=stand)

    def _surface_at(self, level, cell, along):
        """The height of the tilted tile of ``cell``, ``along`` metres down it.

        A cell's height is the height of its uphill edge and the tile falls
        across it to meet the next one, which is the convention
        :func:`~openglcontext_marble_demo.pieces._slope` builds terraces in.
        """
        base = level.cells[cell]
        return base + (0.5 + along / level.cell_size) * self._fall(level, cell)

    def _fall(self, level, cell):
        """How far the tile of ``cell`` drops across itself (negative downhill)."""
        dcol, drow = self.direction
        ahead = (cell[0] + dcol, cell[1] + drow)
        return level.cells.get(ahead, level.cells[cell]) - level.cells[cell]

    # -- building -------------------------------------------------------
    def build(self, scene, level, index, result):
        cs = level.cell_size
        down, _ = _ground_axes(self.direction)
        # Tilt about the horizontal axis across the way down (up x down).
        tilt_axis = (down[2], 0.0, -down[0])
        tile_geometry = render.tile_mesh(size=(cs, 1.0, cs))

        for cell in self.cells():
            if cell not in level.cells:
                continue
            fall = self._fall(level, cell)
            angle = math.atan2(fall, cs)
            x, z = level.cell_center(cell)
            surface = materials.SURFACES[level.surface_of(cell)]
            # Thick as the flat tiles and hung so its top face meets the cells
            # before and after it: a surface across the terrace rather than a
            # step up out of it.  Built here rather than as a
            # :class:`~openglcontext_marble_demo.level.Ramp` because a ramp
            # carries a boost trigger, and a rockfall decides nothing about
            # where the marble is going.
            tile = scene.add_box(
                size=(cs, 1.0, cs),
                position=(x, level.cells[cell] + fall / 2.0 - 0.5 * math.cos(angle), z),
                color=surface.base_color, dynamic=False,
                material=index[surface.name],
                rotation=(tilt_axis[0], tilt_axis[1], tilt_axis[2], angle))
            tile.transform.children[0] = basenodes.Shape(
                geometry=tile_geometry,
                appearance=render.surface_appearance(surface, grout=True))
            result.feature_bodies.append(tile)

        rock_material = scene.world.add_material(ROCK)
        rock_appearance = render.color_appearance(ROCK_COLOR, roughness=0.95)
        for rock in self.rocks(level):
            body = scene.add_box(size=rock.size, position=rock.position,
                                 color=ROCK_COLOR, dynamic=False,
                                 material=rock_material,
                                 rotation=(*rock.axis, rock.angle))
            body.transform.children[0].appearance = rock_appearance
            result.feature_bodies.append(body)


def _ground_axes(direction):
    """The way down and the way across it, as unit vectors in the XZ plane."""
    length = math.hypot(direction[0], direction[1]) or 1.0
    down = (direction[0] / length, 0.0, direction[1] / length)
    return down, (-down[2], 0.0, down[0])


def _reach(size, lie, angle):
    """How far the highest corner of a slab reaches above its own centre.

    ``size`` is (width, thickness, depth); the slab is turned by ``angle`` about
    the horizontal axis at bearing ``lie``, so the world-up component of each of
    its own axes is ``sin(lie)*sin(angle)``, ``cos(angle)`` and
    ``-cos(lie)*sin(angle)``, and the highest corner is the one that takes every
    half-extent in the direction that raises it.
    """
    lean = math.sin(angle)
    return (size[0] / 2.0 * abs(math.sin(lie)) * lean
            + size[1] / 2.0 * math.cos(angle)
            + size[2] / 2.0 * abs(math.cos(lie)) * lean)
