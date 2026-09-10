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
from typing import Any


class TrackMap:
    """A grid of ``(col, row) -> surface_height`` cells with a fixed cell size.

    ``rails`` is the ``(cell, step)`` pairs a wall stands across, where ``step``
    is a grid direction out of that cell.  Navigational rather than decorative:
    what a rail means to anything reading the board is that the marble cannot
    leave by that side, which is why it belongs here beside the cells rather
    than only in the scene.
    """

    def __init__(self, cells: Any, cell_size: float=4.0, rails: Any=()) -> None:
        self.cells = dict(cells)
        self.cell_size = float(cell_size)
        self.rails = frozenset(rails)

    def railed(self, cell: Any, step: Any) -> Any:
        """Is there a wall across the ``step`` side of ``cell``?"""
        return (cell, step) in self.rails

    def cell_of(self, x: Any, z: Any) -> Any:
        """The ``(col, row)`` whose centre is nearest world point ``(x, z)``."""
        return (int(round(x / self.cell_size)), int(round(z / self.cell_size)))

    def cell_center(self, col: Any, row: Any) -> Any:
        """World ``(x, z)`` of a cell's centre."""
        return (col * self.cell_size, row * self.cell_size)

    def is_track(self, x: Any, z: Any) -> Any:
        """Is world point ``(x, z)`` over an occupied track cell?"""
        return self.cell_of(x, z) in self.cells

    def surface_height(self, x: Any, z: Any) -> Any:
        """Top-surface Y under ``(x, z)``, or ``None`` if it is over the void."""
        return self.cells.get(self.cell_of(x, z))

    def __contains__(self, cell: Any) -> Any:
        return cell in self.cells

    def __len__(self) -> Any:
        return len(self.cells)
