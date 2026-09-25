"""A disc that turns, and the bar mounted on it that carries whoever is on it.

Nothing about the *disc itself* moves a marble: a rolling ball answers to
per-body damping, not to friction with a spinning floor
(:mod:`~openglcontext_marble_demo.mechanisms.sand`'s finding that friction does
nothing to one applies here too), so a floor that only spun under a marble
would turn and leave the marble sitting where it was. What does the carrying is
the bar bolted across the disc, exactly as
:class:`~openglcontext_marble_demo.level.RotatingArm` carries a marble down a
lane: a real body, in real contact, moved by a real collision.

:class:`Turntable` is that arrangement as one mechanism: a
:class:`~openglcontext_marble_demo.level.RotatingArm` long enough to reach a rim
``radius`` cells out, mounted on ``cell``, with a wide flat disc turning at the
same rate underneath it, purely for the eye -- so what a player watches turn is
what is actually the thing turning them.

    >>> from openglcontext_marble_demo import mechanisms
    >>> mechanisms.registry()['turntable'] is Turntable
    True
"""
import math
from dataclasses import dataclass
from typing import Any

from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.transform import Transform

from .. import render
from ..level import RotatingArm
from . import mechanism

__all__ = ['Turntable']

#: Cold slate, so the disc reads as machinery under the bar rather than as more
#: floor.
DISC_COLOUR = (0.55, 0.50, 0.44)

#: How much shy of the rim the drawn disc sits, so it never pokes through the
#: wall standing at the edge of its own cell.
DISC_MARGIN = 0.94


@mechanism('turntable')
@dataclass
class Turntable:
    """A bar spinning about ``cell``, long enough to reach a rim ``radius`` cells out.

    ``rpm`` is the turn rate, read exactly as
    :class:`~openglcontext_marble_demo.level.RotatingArm` reads it -- negative
    turns the other way. ``radius`` is in cells, so one number sizes both the
    bar and the disc drawn under it. ``bar_thickness`` and ``clearance`` pass
    straight through to the arm; ``disc_thickness`` is the drawn disc's own,
    kept well under it so the two never meet.
    """
    cell: tuple[int, int]
    radius: Any = 2.0
    rpm: float = 20.0
    bar_thickness: float = 0.3
    clearance: float = 0.3
    disc_thickness: float = 0.12

    def owned_cells(self) -> Any:
        return set()

    def build(self, scene: Any, level: Any, index: Any, result: Any) -> None:
        self._disc(scene, level, result)
        RotatingArm(cell=self.cell, length=self.span(level), rpm=self.rpm,
                    thickness=self.bar_thickness,
                    clearance=self.clearance).build(scene, level, index, result)

    def span(self, level: Any) -> Any:
        """How far the bar and the disc reach, in metres -- twice the radius."""
        return self.radius * 2.0 * level.cell_size

    def _disc(self, scene: Any, level: Any, result: Any) -> None:
        """The platform, turning at the bar's own rate: drawn, not collided with.

        Undriven, the way the water in
        :mod:`~openglcontext_marble_demo.mechanisms.water` is drawn without a
        body of its own: nothing rides on the disc's own spin, so leaving it
        un-collidable loses nothing and saves a broad-phase entry for something
        a marble never actually touches.
        """
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell]
        span = self.span(level) * DISC_MARGIN
        transform = Transform(
            translation=(x, base + self.disc_thickness / 2.0, z),
            children=[basenodes.Shape(
                geometry=basenodes.Box(size=(span, self.disc_thickness, span)),
                appearance=render.color_appearance(DISC_COLOUR, metallic=0.5,
                                                   roughness=0.4))])
        scene.children.append(transform)
        rate = self.rpm * 2.0 * math.pi / 60.0
        result.animators.append(_Spin(transform, rate))


class _Spin:
    """Turns a :class:`~OpenGLContext.scenegraph.transform.Transform` steadily.

    The whole of what makes the disc read as the thing turning the bar. It
    carries no state a restarted run has to forget: starting over rebuilds the
    level, and this along with it.
    """

    def __init__(self, transform: Any, rate: Any) -> None:
        self.transform = transform
        self.rate = rate
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt
        self.transform.rotation = (0.0, 1.0, 0.0, self.rate * self.time)
