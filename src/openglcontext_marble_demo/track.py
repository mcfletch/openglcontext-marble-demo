"""The logical track: a grid of occupied cells the marble rolls across.

The :class:`TrackMap` is the level's *navigational* description, kept apart from
the physics colliders and render meshes.  The controller consults it every frame
to decide whether the marble is over solid track (and how high its surface is) or
out over the void — the distinction that drives grounding, checkpoints, and the
fall-vs-jump test.

Cell ``(col, row)`` occupies the square of side ``cell_size`` centred at world
``(col * cell_size, ·, row * cell_size)``; its stored value is the Y of that
tile's top surface (so stepped/terraced tracks are just cells at different
heights).
"""


class TrackMap:
    """A grid of ``(col, row) -> surface_height`` cells with a fixed cell size."""

    def __init__(self, cells, cell_size=4.0):
        self.cells = dict(cells)
        self.cell_size = float(cell_size)

    def cell_of(self, x, z):
        """The ``(col, row)`` whose centre is nearest world point ``(x, z)``."""
        return (int(round(x / self.cell_size)), int(round(z / self.cell_size)))

    def cell_center(self, col, row):
        """World ``(x, z)`` of a cell's centre."""
        return (col * self.cell_size, row * self.cell_size)

    def is_track(self, x, z):
        """Is world point ``(x, z)`` over an occupied track cell?"""
        return self.cell_of(x, z) in self.cells

    def surface_height(self, x, z):
        """Top-surface Y under ``(x, z)``, or ``None`` if it is over the void."""
        return self.cells.get(self.cell_of(x, z))

    def __contains__(self, cell):
        return cell in self.cells

    def __len__(self):
        return len(self.cells)
