"""A floor that only holds for a moment, and the slower way down once it does not.

Crossing it is rolling: a marble moving at any ordinary pace is off the far
edge long before the floor has taken enough of its weight to matter, and the
top of the piece reads as an unremarkable stretch of track. Stop on it, ease
off it, or work out a line from the middle of it, and the tile gives way —
:class:`~openglcontext_marble_demo.mechanisms.collapse.Collapse` measures the
time it has been carrying weight, not how fast that weight was moving, because
the game already has pieces about speed and this one is about a clock instead.

What is on the other side of the hole is not the void: immediately past the
panel a graduated way down carries on to its own exit, so falling through is a
worse route rather than a lost run. The panel does not reset itself once it has
gone -- a second crossing of the same board meets a hole where the floor was,
which is the whole point of a decision that cannot be taken back.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = collapse(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits and 'missed' in piece.exits
    True
"""
import math
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.collapse import Collapse
from openglcontext_marble_demo.pieces import LANE, MAX_STEP, Piece, Port, lay, rails

__all__ = ['collapse', 'RULE', 'VARIANTS']

RULE = 'cross it moving; stop on it and the floor is gone for good'

#: Material, layout and effect: what the floor is made of, how many tiles of it
#: there are, and how long it bears weight before it gives.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain':   {'theme': 'stone',   'length': 2, 'hold_time': 0.9, 'below': 1.8},
    'quick':   {'theme': 'ice',     'length': 1, 'hold_time': 0.5, 'below': 1.8},
    'foundry': {'theme': 'foundry', 'length': 3, 'hold_time': 1.5, 'below': 2.7},
    'sunken':  {'theme': 'rubber',  'length': 2, 'hold_time': 1.0, 'below': 3.6},
}


def _steps(cells: Any, port: Any, drop: Any, width: Any) -> Any:
    """Cells stepping down by at most :data:`MAX_STEP` each, unramped.

    A stepped slope rolls downhill and cannot be climbed -- a riser is a wall
    in front of whatever meets it going up -- which is exactly what a floor
    with no going back wants of the way past it. Unramped rather than built
    with :func:`~openglcontext_marble_demo.pieces.slope`, whose ramp tiles
    are thin, tilted slabs: a body carried across the collapsed floor lands on
    whichever one of these full blocks its fall first reaches, and a thin
    slab is a worse target to hit than a block a metre deep.
    """
    steps = max(2, math.ceil(abs(drop) / MAX_STEP) + 1)
    each = drop / (steps - 1)
    height = port.height
    for step in range(steps):
        for cell in port.ahead(step).cells():
            cells[cell] = height
        height = round(height + each, 6)
    return Port(cell=port.ahead(steps - 1).cell, facing=port.facing,
               height=round(port.height + each * (steps - 1), 6), width=width)


@fragment('collapse', tags=('hazard', 'gate'), rule=RULE, cost=5.0,
          variants=tuple(VARIANTS))
def collapse(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, length: Any=None, hold_time: Any=None,
            below: Any=None) -> Any:
    """An approach, a floor that only holds a moment, and a way down past it."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    length = length or settings['length']
    hold_time = hold_time if hold_time is not None else settings['hold_time']
    below = below if below is not None else settings['below']

    cells: dict = {}
    features: list = []

    approach = lay(cells, entry, 1, width=LANE)
    panel_port = approach.ahead(1)
    before = set(cells)
    panel_end = lay(cells, panel_port, length, width=LANE)
    panel_area = tuple(sorted(set(cells) - before))

    ok_exit = Port(cell=panel_end.cell, facing=entry.facing, height=entry.height,
                   width=LANE)
    features.extend(rails(cells, entry, 1 + length, width=LANE))

    # The way past the hole: a stepped descent -- respecting the same slope
    # budget as any other joiner -- straight on from where the floor was, so a
    # marble that falls through lands with somewhere to go rather than with
    # nothing under it.  Its first cell holds the panel's own height, so the
    # carry below needs only clear whatever is left of the panel, not the
    # descent as well.
    foot = _steps(cells, panel_end.ahead(1), -abs(below), width=LANE)
    run = 3
    landing = lay(cells, foot.ahead(1), run, height=foot.height, width=LANE)
    features.extend(rails(cells, foot.ahead(1), run, width=LANE))
    missed_exit = Port(cell=landing.cell, facing=entry.facing, height=foot.height,
                       width=LANE)

    # The floor hinges at its near edge and swings down and forward, toward
    # the descent immediately past it, and carries a body caught on it the
    # same way -- so a marble that was standing still when it gave way still
    # reaches solid ground rather than dropping straight through in place.
    features.append(Collapse(cells=panel_area, direction=entry.facing, hold_time=hold_time))

    return Piece(name='collapse', cells=cells, entry=entry,
                 exits={'ok': ok_exit, 'missed': missed_exit},
                 features=features, theme=theme, rule=RULE)
