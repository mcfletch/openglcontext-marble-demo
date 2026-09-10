"""A patch of hot plate that destroys a marble it holds long enough.

The burner is the trap you can run through.  It does not take a marble that
touches it: it takes one that is *slow* across it.  A marble rolling over at pace
is out the far side before it has taken enough heat to matter, and one that
arrives without the speed to leave -- steered in on the wrong line, tipped in off
a ramp, sat in the middle of the field working out where to go -- burns.  So the
skill it asks for is carrying speed through, not hitting a window.

**The heat is a clock, not a switch.** Each body in the fire gains a second of
heat per second and, once it is out, loses ``cool_rate`` seconds per second down
to none.  At :attr:`Burner.burn_time` the marble is gone.  The cooling is what
makes two quick dips across a wide field different from one long crawl through
it, and what keeps a marble that got out with its skin intact from carrying the
last field's heat into the next one.

**The edge is a line.** A body is in the fire when its centre is over one of the
burner's cells *and* within :attr:`Burner.catch_height` of that cell's surface,
so clipping the corner costs exactly the corner and an arc over the top costs
nothing at all: the ramp route past a burner is a real route.

:class:`BurnerHeat` rides in ``BuildResult.animators``, the list the game updates
once per frame before it steps the world, and in ``BuildResult.resettable``, so a
restarted run starts cold.  It finds bodies by looking rather than by being
registered, which is what lets it work on a marble that did not exist when the
level was built -- the game spawns the marble after the level, and swaps it for
another mid-run.

    >>> Burner(cells=((3, 0), (4, 0))).owned_cells() == {(3, 0), (4, 0)}
    True
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial

from .. import materials
from ..controller import BURNED
from . import mechanism

__all__ = ['Burner', 'BurnerHeat', 'EMBER']

#: The look and contact feel of the plate: dark, dead metal, hot enough to glow.
#: It grips about as well as stone, so the way out of a burner is the way a
#: player expects it to be -- the trap is the time it costs, not a loss of
#: control on top.
EMBER = materials.SurfaceMaterial(
    name='ember', base_color=(0.20, 0.06, 0.03), metallic=0.0, roughness=0.55,
    static_friction=0.75, dynamic_friction=0.65, restitution=0.05)

#: The colour the plate gives off, and how far above merely lit it is drawn, so
#: the field reads as dangerous from across the board rather than as dark tiling.
EMBER_GLOW = (1.0, 0.30, 0.05)
EMBER_GLOW_STRENGTH = 2.5


class BurnerHeat:
    """How long each body has been standing in the fire.

    Built with the world it watches, the burner's cells as ``{(col, row):
    ceiling_y}``, the cell size, and the two rates.  :meth:`update` runs once a
    frame before the world steps and moves every body's heat toward or away from
    the limit; :meth:`lost` is what the game asks to find out whether one of them
    has reached it.

    The search for bodies is a vector test against the field's bounding box, so
    the per-frame cost is one pass over the body arrays and a dictionary lookup
    for the few bodies actually near the fire.
    """

    def __init__(self, world: Any, ceilings: Any, cell_size: float, burn_time: Any, cool_rate: Any) -> None:
        self.world = world
        self.ceilings = dict(ceilings)
        self.cell_size = float(cell_size)
        self.burn_time = float(burn_time)
        self.cool_rate = float(cool_rate)
        #: Body index -> seconds of heat taken so far.  A body that is cold is
        #: not in here at all, so the table stays the size of what is near the
        #: fire rather than the size of the world.
        self.heat: dict[int, float] = {}
        half = self.cell_size / 2.0
        cols = [col for col, _row in self.ceilings]
        rows = [row for _col, row in self.ceilings]
        self._west = min(cols) * self.cell_size - half
        self._east = max(cols) * self.cell_size + half
        self._north = min(rows) * self.cell_size - half
        self._south = max(rows) * self.cell_size + half
        self._top = max(self.ceilings.values())

    def holds(self, point: Any) -> Any:
        """Is world ``point`` down in the fire?

        The cell edge is the whole of the boundary: a hand's breadth outside it,
        or a marble's diameter above it, and the plate has no hold at all.
        """
        x, y, z = point[0], point[1], point[2]
        cell = (int(round(x / self.cell_size)), int(round(z / self.cell_size)))
        ceiling = self.ceilings.get(cell)
        return ceiling is not None and y < ceiling

    def bodies_inside(self) -> Any:
        """The indices of the dynamic bodies currently in the fire."""
        world = self.world
        position = world.position
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        # ``inv_mass`` is non-zero for exactly the bodies that can be moved into
        # the fire; the box test then leaves the cell lookup to the handful of
        # bodies actually near it.
        near = ((world.inv_mass > 0.0) & (y < self._top)
                & (x >= self._west) & (x <= self._east)
                & (z >= self._north) & (z <= self._south))
        return [int(i) for i in np.flatnonzero(near) if self.holds(position[i])]

    def update(self, dt: float=0.0) -> None:
        """Take ``dt`` seconds of heat into everything in the fire, out of the rest."""
        inside = set(self.bodies_inside())
        for index in inside:
            self.heat[index] = min(self.burn_time,
                                   self.heat.get(index, 0.0) + dt)
        for index in list(self.heat):
            if index in inside:
                continue
            cooled = self.heat[index] - self.cool_rate * dt
            if cooled <= 0.0:
                del self.heat[index]
            else:
                self.heat[index] = cooled

    def lost(self, body: Any) -> Any:
        """What ``body`` has been lost to, or ``None`` while it is still whole.

        Answers **once**: the heat is spent on the marble it took, and the one
        that arrives at the checkpoint is a new marble, cold.  This is the hook
        :class:`~openglcontext_marble_demo.game.MarbleGame` asks every hazard on
        the board once a frame.
        """
        if self.heat.get(body, 0.0) < self.burn_time:
            return None
        del self.heat[body]
        return BURNED

    def reset(self, world: Any=None) -> None:
        """Put the fire out on everything: a restarted run starts cold."""
        self.heat.clear()


@mechanism('burner')
@dataclass
class Burner:
    """A field of hot plate over ``cells``.

    ``burn_time`` is how many seconds in the fire a marble survives, and it is
    the whole of the difficulty: at the default a marble crossing a two-cell
    field needs about 7 m/s to be out in time, which is a shade under what the
    board's own lean gives a marble that has been rolling for a while.  A single
    cell is survivable at a walk; a wide field has to be run.

    ``cool_rate`` is how fast the heat comes back off once the marble is clear,
    in seconds of heat per second.  At the default it sheds as fast as it built,
    so a field crossed in two hops is not the same as one crossed in one crawl.

    ``catch_height`` is how far above the plate, in metres, a body is still in
    the fire.  It wants to be roughly a marble's diameter: enough that one
    rolling or skimming is caught, little enough that a ramp's arc is clear.

    The field covers those of ``cells`` the level has track for; the plate needs
    a floor to lie in, and the track is what says where there is one.
    """
    #: The cells this covers, as `(col, row)` pairs. Taken as any sequence
    #: of them, because a level gives tuples and a file gives back lists;
    #: `__post_init__` settles it to tuples so the two are one board.
    cells: Sequence[Sequence[int]]
    burn_time: Any = 1.2
    cool_rate: float = 1.0
    catch_height: float = 1.0

    def __post_init__(self) -> None:
        # A file gives back lists where a level gave tuples; one spelling here
        # means a board that has been through the file format is the same board.
        self.cells = tuple((int(col), int(row)) for col, row in self.cells)

    def owned_cells(self) -> Any:
        return set(self.cells)

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        material = scene.world.add_material(materials.physics_material(EMBER))
        appearance = _ember_appearance()
        ceilings = {}
        for cell in self.cells:
            if cell not in level.cells:
                continue
            height = level.cells[cell]
            x, z = level.cell_center(cell)
            tile = scene.add_box(size=(level.cell_size, 1.0, level.cell_size),
                                 position=(x, height - 0.5, z),
                                 color=EMBER.base_color, dynamic=False,
                                 material=material)
            tile.transform.children[0].appearance = appearance
            result.feature_bodies.append(tile)
            ceilings[cell] = height + self.catch_height
        if ceilings:
            heat = BurnerHeat(scene.world, ceilings, level.cell_size,
                              self.burn_time, self.cool_rate)
            result.animators.append(heat)
            result.resettable.append(heat)


def _ember_appearance() -> Any:
    """A dark plate with a hot glow coming off it, shared by every tile."""
    return basenodes.Appearance(material=PBRMaterial(
        baseColor=EMBER.base_color, metallic=EMBER.metallic,
        roughness=EMBER.roughness, emissiveColor=EMBER_GLOW,
        emissiveStrength=EMBER_GLOW_STRENGTH))
