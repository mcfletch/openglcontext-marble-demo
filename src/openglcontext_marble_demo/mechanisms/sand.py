"""Sand: a bounded field that a marble has to churn its way out of.

A marble that rolls in slows to a crawl and takes seconds to work its way to the
far edge, so the two ways past a sand field are to steer round it -- along the
edge, giving up the line -- or to be launched over it by a ramp and land beyond.

**The drag is damping, not friction.** Friction acts where two surfaces slide
across each other, and a marble rolling on the floor does not slide: raising the
sand's friction coefficient, or its pairwise coefficient against every marble,
leaves the crossing time unchanged to the hundredth of a second. What sand
actually does to a ball is resist it through the whole of its motion, which is
what :attr:`~omi_physics.world.PhysicsWorld.linear_damping` and
:attr:`~omi_physics.world.PhysicsWorld.angular_damping` are -- per-body rates,
added to the world's own, that bleed speed and spin away every second. Both
matter: the linear rate slows the travel, and the angular rate takes the spin
that grip would otherwise turn back into travel. The sand still carries a
gripping, dead surface material, because that is what a marble landing in it
should meet.

**The edge is a line.** A body is in the sand when its centre is over one of the
sand's cells *and* within :attr:`Sand.catch_height` of that cell's surface --
nothing tapers, so clipping the corner of the field costs exactly the corner and
skimming over the top costs nothing. The height is what leaves the ramp route
open: a marble rolling through has its centre half a radius up and is in the
sand; one carried over in an arc is clear of it and crosses at full speed.

:class:`SandDrag` rides in ``BuildResult.animators``, the list the game updates
once per frame before it steps the world. It matches every dynamic body's
damping to where that body is, which is what lets it work on a marble that did
not exist when the level was built -- the game spawns the marble after the
level, and swaps it for another mid-run.
"""
from dataclasses import dataclass

import numpy as np

from .. import materials, render
from . import mechanism

#: The look and contact feel of sand: a dry, rough, gripping surface that
#: returns none of what lands on it.  The speed a marble loses in the sand comes
#: from :class:`SandDrag`; this is what the surface is made of.
SAND = materials.SurfaceMaterial(
    name='sand', base_color=(0.78, 0.68, 0.44), metallic=0.0, roughness=0.95,
    static_friction=0.95, dynamic_friction=0.85, restitution=0.0)


class SandDrag:
    """Holds every body inside a sand field at a raised damping.

    Built with the world it acts on, the sand's cells as ``{(col, row):
    ceiling_y}``, and the two rates to add.  :meth:`update` is called once per
    frame before the world steps: it raises the damping of each body that is in
    the sand and gives back exactly what it took from each body that has left,
    so two fields overlapping, or a marble swapped mid-run, leave nothing behind.

    Bodies are found by looking rather than by being registered, because the
    marble is spawned after the level is built and can be replaced while it is
    running.  The search is a vector test against the field's bounding box, so
    the per-frame cost is one pass over the body arrays and a dictionary lookup
    for the few bodies actually near the sand.
    """

    def __init__(self, world, ceilings, cell_size, linear_drag, angular_drag):
        self.world = world
        self.ceilings = dict(ceilings)
        self.cell_size = float(cell_size)
        self.linear_drag = float(linear_drag)
        self.angular_drag = float(angular_drag)
        #: Body index -> the damping it had before it went in.
        self.caught: dict[int, tuple[float, float]] = {}
        half = self.cell_size / 2.0
        cols = [col for col, _row in self.ceilings]
        rows = [row for _col, row in self.ceilings]
        self._west = min(cols) * self.cell_size - half
        self._east = max(cols) * self.cell_size + half
        self._north = min(rows) * self.cell_size - half
        self._south = max(rows) * self.cell_size + half
        self._top = max(self.ceilings.values())

    def holds(self, point):
        """Is world ``point`` down in the sand?

        The cell edge is the whole of the boundary: a hand's breadth outside it,
        or a marble's diameter above it, and the sand has no hold at all.
        """
        x, y, z = point[0], point[1], point[2]
        cell = (int(round(x / self.cell_size)), int(round(z / self.cell_size)))
        ceiling = self.ceilings.get(cell)
        return ceiling is not None and y < ceiling

    def bodies_inside(self):
        """The indices of the dynamic bodies currently in the sand."""
        world = self.world
        position = world.position
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        # ``inv_mass`` is non-zero for exactly the bodies the integrator damps;
        # the box test then leaves the cell lookup to the handful near the field.
        near = ((world.inv_mass > 0.0) & (y < self._top)
                & (x >= self._west) & (x <= self._east)
                & (z >= self._north) & (z <= self._south))
        return [int(i) for i in np.flatnonzero(near) if self.holds(position[i])]

    def update(self, dt=0.0):
        """Match each body's damping to whether it is in the sand this frame."""
        inside = set(self.bodies_inside())
        for index in set(self.caught) - inside:
            self._release(index)
        for index in inside - set(self.caught):
            self._catch(index)

    def _catch(self, index):
        world = self.world
        linear = float(world.linear_damping[index])
        angular = float(world.angular_damping[index])
        self.caught[index] = (linear, angular)
        world.linear_damping[index] = linear + self.linear_drag
        world.angular_damping[index] = angular + self.angular_drag

    def _release(self, index):
        linear, angular = self.caught.pop(index)
        self.world.linear_damping[index] = linear
        self.world.angular_damping[index] = angular


@mechanism('sand')
@dataclass
class Sand:
    """A field of sand over ``cells``, with hard edges and a heavy drag.

    ``linear_drag`` and ``angular_drag`` are rates per second added to the
    world's own damping while a body is in the field: at the defaults a steel
    marble arriving at 8 m/s takes about eight times as long to cross two cells
    as it does to cross two cells of stone, and it always gets out -- the board's
    downhill lean carries it at a walking pace rather than stopping it dead.

    ``catch_height`` is how far above the sand's surface, in metres, a body is
    still in it.  It wants to be roughly a marble's diameter: enough that a
    marble rolling or skimming is caught, little enough that a ramp's arc is
    clear.

    The field covers those of ``cells`` the level has track for; sand needs a
    floor to lie on, and the track is what says where there is one.
    """
    cells: tuple[tuple[int, int], ...]
    linear_drag: float = 2.0
    angular_drag: float = 3.0
    catch_height: float = 1.0

    def __post_init__(self):
        # A file gives back lists where a level gave tuples; one spelling here
        # means a board that has been through the file format is the same board.
        self.cells = tuple((int(col), int(row)) for col, row in self.cells)

    def owned_cells(self):
        return set(self.cells)

    def build(self, scene, level, index, result):
        material = scene.world.add_material(materials.physics_material(SAND))
        appearance = render.surface_appearance(SAND)
        ceilings = {}
        for cell in self.cells:
            if cell not in level.cells:
                continue
            height = level.cells[cell]
            x, z = level.cell_center(cell)
            tile = scene.add_box(size=(level.cell_size, 1.0, level.cell_size),
                                 position=(x, height - 0.5, z),
                                 color=SAND.base_color, dynamic=False,
                                 material=material)
            tile.transform.children[0].appearance = appearance
            result.feature_bodies.append(tile)
            ceilings[cell] = height + self.catch_height
        if ceilings:
            result.animators.append(SandDrag(
                scene.world, ceilings, level.cell_size,
                self.linear_drag, self.angular_drag))
