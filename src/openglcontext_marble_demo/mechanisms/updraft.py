"""An updraft: a column of rising air that slows a fall without ever stopping it.

Every other field in the library acts on a marble that is rolling --
:class:`~openglcontext_marble_demo.mechanisms.sand.Sand` drags on contact with the
floor, a :class:`~openglcontext_marble_demo.mechanisms.magnet.Magnet` bends a line
that is still on the ground.  An updraft is for the marble that has *left* the
ground: it stands over a gap with nothing under it, and what it changes is how
fast that marble sinks while it is in the air above the gap.

**It is weaker than gravity, always.** ``strength`` is an upward acceleration in
metres per second squared, and it is never set to or above the vertical pull the
board's own lean gives a falling body -- a marble held stationary over the shaft,
with nothing carrying it forward, still comes down, only slower than it would in
open air.  That is what keeps the shaft a draft and not a floor: crossing it is a
question about *speed*, not about whether you can get in at all.  A marble that
is moving fast enough covers the width of the gap before it has sunk far; one
that dawdles sinks below the reach of the piece before it gets there and the
board's own fall rule takes it, exactly as it would over any other gap.

**The reach is a box, not a point.** ``cells`` is the shaft's footprint in the
grid -- the gap's own cells, which carry no floor of their own -- and the lift
acts on any dynamic body over that footprint whose height is between ``floor``
and ``floor + reach``. Both ends are hard lines, in the manner of
:class:`~openglcontext_marble_demo.mechanisms.sand.Sand`'s ``catch_height``: a
body outside the box in any of the three axes feels nothing at all.

:class:`UpdraftLift` rides in ``BuildResult.animators``, the list the game
updates once per frame before it steps the world.  It finds bodies by looking
rather than by being registered, which is what lets it work on a marble that did
not exist when the level was built.

    >>> from openglcontext_marble_demo import mechanisms
    >>> mechanisms.registry()['updraft'] is Updraft
    True
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.transform import Transform

from . import mechanism

__all__ = ['UPDRAFT_COLOR', 'Updraft', 'UpdraftLift']

#: The colour of the visible column of moving air: pale, cool, and translucent,
#: so it reads as something to look *through* rather than an obstacle.
UPDRAFT_COLOR = (0.78, 0.90, 0.95)


class UpdraftLift:
    """Applies one field's lift to every dynamic body inside its box.

    Built with the world it acts on, the shaft's cells, the level's cell size,
    and the box's vertical extent and strength. :meth:`update` is called once
    per frame before the world steps.
    """

    def __init__(self, world: Any, cells: Any, cell_size: float, floor: Any, reach: Any, strength: Any) -> None:
        self.world = world
        self.floor = float(floor)
        self.top = float(floor) + float(reach)
        self.strength = float(strength)
        half = float(cell_size) / 2.0
        cols = [col for col, _row in cells]
        rows = [row for _col, row in cells]
        self._west = min(cols) * cell_size - half
        self._east = max(cols) * cell_size + half
        self._north = min(rows) * cell_size - half
        self._south = max(rows) * cell_size + half

    def holds(self, point: Any) -> Any:
        """Is world ``point`` inside the shaft's footprint and its reach?"""
        x, y, z = point[0], point[1], point[2]
        return (self._west <= x <= self._east and self._north <= z <= self._south
                and self.floor <= y <= self.top)

    def bodies_in_reach(self) -> Any:
        """The indices of the dynamic bodies currently inside the shaft.

        One vector pass over the body arrays, in the manner of
        :class:`~openglcontext_marble_demo.mechanisms.magnet.MagnetPull`: the
        animator runs every frame of every board carrying an updraft, and its
        cost should grow with the number of bodies near the shaft, not with a
        Python loop over every body on the board.
        """
        world = self.world
        position = world.position
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        near = ((world.inv_mass > 0.0)
                & (x >= self._west) & (x <= self._east)
                & (z >= self._north) & (z <= self._south)
                & (y >= self.floor) & (y <= self.top))
        return [int(index) for index in np.flatnonzero(near)]

    def update(self, dt: float=0.0) -> None:
        """Apply one frame of lift to everything inside the shaft."""
        if not dt:
            return
        world = self.world
        for index in self.bodies_in_reach():
            mass = world.mass[index]
            world.apply_impulse(index, (0.0, self.strength * mass * dt, 0.0))


@mechanism('updraft')
@dataclass
class Updraft:
    """A shaft of rising air over ``cells``, which carry no floor of their own.

    ``strength`` is the lift in metres per second squared, always meant to be
    set below the vertical pull gravity gives a falling body -- see the module
    docstring for why a stationary marble still sinks. ``floor`` is the height,
    in the same units as a level's cells, that the column starts from, and
    ``reach`` how far above that the lift still acts: together they are the box
    a body has to be inside to feel anything.

    The cells the shaft covers are never built a floor tile -- there is nothing
    to stand on over a gap -- which is why :meth:`owned_cells` answers empty:
    it is not claiming a cell's tile, it is a field over cells nothing ever laid
    one for.
    """
    #: The cells this covers, as `(col, row)` pairs. Taken as any sequence
    #: of them, because a level gives tuples and a file gives back lists;
    #: `__post_init__` settles it to tuples so the two are one board.
    cells: Sequence[Sequence[int]]
    strength: float = 7.5
    floor: float = -6.0
    reach: float = 7.5

    def __post_init__(self) -> None:
        # A file gives back lists where a level gave tuples; one spelling here
        # means a board that has been through the file format is the same board.
        self.cells = tuple((int(col), int(row)) for col, row in self.cells)

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        if not self.cells:
            return
        self._draw(scene, level)
        result.animators.append(UpdraftLift(
            scene.world, self.cells, level.cell_size, self.floor, self.reach,
            self.strength))

    def _draw(self, scene: Any, level: Any) -> None:
        """A translucent column over the shaft, drawn and never collided with.

        Built the way :class:`~openglcontext_marble_demo.mechanisms.water.Water`
        draws its own column: a plain node appended to the scene graph rather
        than one of ``scene.add_box``'s solid bodies, so a marble crossing above
        it passes through the same air the lift acts in.
        """
        cs = level.cell_size
        cols = [col for col, _row in self.cells]
        rows = [row for _col, row in self.cells]
        x0, x1 = min(cols) * cs - cs / 2.0, max(cols) * cs + cs / 2.0
        z0, z1 = min(rows) * cs - cs / 2.0, max(rows) * cs + cs / 2.0
        middle = self.floor + self.reach / 2.0
        geometry = basenodes.Box(size=(x1 - x0, self.reach, z1 - z0))
        appearance = basenodes.Appearance(material=PBRMaterial(
            baseColor=UPDRAFT_COLOR, metallic=0.0, roughness=0.05,
            transparency=0.82, alphaMode='BLEND', doubleSided=True))
        scene.children.append(Transform(
            translation=((x0 + x1) / 2.0, middle, (z0 + z1) / 2.0),
            children=[basenodes.Shape(geometry=geometry, appearance=appearance)]))
