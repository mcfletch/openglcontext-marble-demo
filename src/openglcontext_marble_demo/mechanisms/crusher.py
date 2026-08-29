"""A press that strikes down on a cycle, and takes a marble caught beneath it.

A :class:`Crusher` is a kinematic slab hanging over ``cells``, raised clear of
the lane for most of its cycle and striking down to a hand's breadth off the
floor and back, on a fixed :attr:`period`. It puts nothing between a player and
the lane except time: crossing while the press is up costs nothing, and the
press is up far more than it is down.

**The controller does the crushing; the mechanism only swings.** A marble held
between two opposed surfaces closer together than its own width, for longer
than :attr:`~openglcontext_marble_demo.controller.MarbleController.crush_time`,
is destroyed with cause
:data:`~openglcontext_marble_demo.controller.CRUSHED` --
:meth:`~openglcontext_marble_demo.controller.MarbleController._held_too_long`
reads it straight off this step's contacts, floor against press, the same way
it would read any two surfaces that closed on the marble. Nothing here calls
:meth:`~openglcontext_marble_demo.controller.MarbleController.destroy`; the
press only has to arrive and dwell, and :attr:`dwell` is set with margin over
``crush_time`` so a marble still under the blow when it lands is squeezed for
the whole of it.

**Why timing is not exact.** The press eases into its stroke and out again
(a half-cosine, never a jump), it dwells at the top for most of the period, and
it only closes to :attr:`gap` for :attr:`dwell` seconds at the bottom -- a
window a rolling marble crosses in a fraction of that time. What kills a marble
is not being in the lane when the press happens to be low: it is *stopping*
there, because only standing still keeps it under the blow for the whole
dwell. A player who reads the rhythm and keeps moving is under the press for
less time than it takes to close; one who dithers is caught by the next blow,
whichever moment that is.

    >>> Crusher(cells=((0, 0), (1, 0), (2, 0))).owned_cells()
    set()
"""
import math
from dataclasses import dataclass

from omi_physics import model
from omi_physics.kinematic import KinematicAnimator
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.physicsbody import PhysicsBody
from OpenGLContext.scenegraph.transform import Transform

from .. import render
from . import mechanism

__all__ = ['Crusher']

#: Dull, dead iron -- a thing that does not look like it would give.
COLOUR = (0.45, 0.16, 0.14)


@mechanism('crusher')
@dataclass
class Crusher:
    """A press hanging over ``cells``, striking down and back on a cycle.

    ``cells`` is the footprint: every cell of the lane the slab is wide enough
    to cover, so there is no way round it, only under it. It hangs above the
    track rather than replacing it -- :meth:`owned_cells` is empty, exactly
    like :class:`~openglcontext_marble_demo.level.RotatingArm` -- so the floor
    stays and a marble rolls under freely whenever the press is up.

    ``clearance`` and ``gap`` are how far the press's underside stands above
    the floor, raised and struck home; ``clearance`` wants to clear a marble
    with room, ``gap`` wants to be well under a marble's diameter so a body
    caught there is genuinely pinched.  ``strike`` is the seconds the press
    takes to travel each way and ``dwell`` how long it holds the blow down --
    set with margin over
    :attr:`~openglcontext_marble_demo.controller.MarbleController.crush_time`
    so a marble still under it when it lands is squeezed the whole way through.
    ``period`` is the whole cycle, so the press is raised, and the lane clear,
    for ``period - 2 * strike - dwell`` seconds out of every one -- the window
    a player crosses in, and the reason getting it right does not mean getting
    it exact.

    ``phase`` offsets the animator's own clock, in seconds, so two presses on
    one board need not fall together.

    ``depth`` is how far the slab reaches along ``travel``, in metres, thinner
    than the cell it hangs over -- a blade rather than a lid.  It is what
    keeps the rule about *stopping*, not about *being there*: a marble that
    keeps rolling is only ever briefly within reach of the blow (roughly
    ``depth`` plus its own diameter, divided by its speed), and that is well
    under ``crush_time`` at any pace worth calling rolling.  ``None`` covers
    the whole cell, which is what a mechanism placed by hand and not asking
    for the timing rule at all would want.

    ``travel`` is the grid direction a marble crosses the press in -- a step
    like :attr:`~openglcontext_marble_demo.level.Ramp.direction`, ``(0, 1)`` by
    default.  It says which of the press's two horizontal extents is the one
    ``depth`` narrows: the other stays the full width of ``cells``, which is
    what keeps a wide press a gate rather than, narrowed on the wrong axis, a
    pillar with a way round it on either side.
    """
    cells: tuple[tuple[int, int], ...]
    period: float = 3.0
    strike: float = 0.25
    dwell: float = 0.4
    clearance: float = 1.6
    gap: float = 0.2
    thickness: float = 0.5
    phase: float = 0.0
    depth: float | None = 0.6
    travel: tuple[int, int] = (0, 1)

    def __post_init__(self):
        # A file gives back lists where a level gave tuples; one spelling here
        # means a board that has been through the file format is the same board.
        self.cells = tuple((int(col), int(row)) for col, row in self.cells)

    def owned_cells(self):
        return set()

    def build(self, scene, level, index, result):
        present = [cell for cell in self.cells if cell in level.cells]
        if not present:
            return
        centres = [level.cell_center(cell) for cell in present]
        cols = [cell[0] for cell in present]
        rows = [cell[1] for cell in present]
        cs = level.cell_size
        x = (min(x for x, _ in centres) + max(x for x, _ in centres)) / 2.0
        z = (min(z for _, z in centres) + max(z for _, z in centres)) / 2.0
        span_x = (max(cols) - min(cols) + 1) * cs
        span_z = (max(rows) - min(rows) + 1) * cs
        # Narrowed along the way a marble travels, never across it -- the
        # across span is the lane's own width, and shrinking that would open a
        # way round the press rather than a quick way under it.
        if self.depth is not None:
            if self.travel[1]:
                span_z = min(self.depth, span_z)
            else:
                span_x = min(self.depth, span_x)
        size = (span_x, self.thickness, span_z)
        base = max(level.cells[cell] for cell in present)

        top = base + self.clearance + self.thickness / 2.0
        bottom = base + self.gap + self.thickness / 2.0
        period, strike, dwell = self.period, self.strike, self.dwell

        body = _kinematic_box(scene, size, (x, top, z), index[level.surface])

        def pose(t):
            y = _height(t % period, period, strike, dwell, top, bottom)
            return (x, y, z), (0.0, 0.0, 0.0, 1.0)

        result.animators.append(
            KinematicAnimator(scene.world, body.index, pose, time=self.phase))
        result.feature_bodies.append(body)


def _height(frac, period, strike, dwell, top, bottom):
    """The press's height at ``frac`` seconds into its cycle.

    Raised for most of the period; the stroke down, the dwell at the bottom,
    and the stroke back up are ``2 * strike + dwell`` seconds out of it.  Each
    stroke is a half-cosine ease, so the press arrives at rest at both ends --
    the dwell is undisturbed rather than a sudden stop, and there is no jump
    in velocity for a marble to be caught by, only the squeeze itself.
    """
    if frac < strike:
        return top - (top - bottom) * _ease(frac / strike)
    frac -= strike
    if frac < dwell:
        return bottom
    frac -= dwell
    if frac < strike:
        return bottom + (top - bottom) * _ease(frac / strike)
    return top


def _ease(phase):
    return 0.5 * (1.0 - math.cos(math.pi * phase))


def _kinematic_box(scene, size, position, material_index):
    """Add a kinematic box to a ``DemoScene``: render Transform plus physics body.

    ``DemoScene`` offers static and dynamic bodies; a press that drives its own
    motion is neither, so it is assembled here from the same node types the
    scene uses and registered with the scene's physics manager and render
    children -- the same construction
    :func:`~openglcontext_marble_demo.level._spawn_kinematic_box` and
    :mod:`~openglcontext_marble_demo.mechanisms.water` each do for their own
    moving parts.
    """
    shape = scene.world.add_shape(model.Shape.box(size))
    transform = Transform(translation=tuple(position),
                          children=[basenodes.Shape(
                              geometry=basenodes.Box(size=size),
                              appearance=render.color_appearance(
                                  COLOUR, metallic=0.7, roughness=0.4))])
    body = PhysicsBody(transform, model.Motion(type=model.KINEMATIC),
                       model.Collider(shape=shape, physicsMaterial=material_index))
    scene.manager.add(body)
    scene.children.append(transform)
    return body
