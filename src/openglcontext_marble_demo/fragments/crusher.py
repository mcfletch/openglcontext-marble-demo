"""A lane with a press on it, and going under it between blows.

One or more :class:`~openglcontext_marble_demo.mechanisms.crusher.Crusher`
slabs hang across the full width of a walled lane, each raised clear for most
of its cycle and striking down to a hand's breadth off the floor for a moment.
Crossing while a press is up costs nothing; a marble still under one when it
lands is squeezed until
:class:`~openglcontext_marble_demo.controller.MarbleController` calls it
:data:`~openglcontext_marble_demo.controller.CRUSHED`.

The rule does not ask for a frame: a press is down, and low enough to pinch, for
a small fraction of its own period, so a marble that keeps moving is past the
danger zone before a blow can hold it there for the
:attr:`~openglcontext_marble_demo.controller.MarbleController.crush_time`
it needs. What it asks is that a player not stop in the lane -- read the
rhythm, keep rolling, and a press is furniture; stall under one and the next
blow is the one that counts.
"""
from typing import Any

from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.crusher import Crusher
from openglcontext_marble_demo.pieces import LANE, Piece, Port

__all__ = ['crusher']

#: Cells of clear lane before the first press, so a marble is rolling straight
#: and at its own pace by the time the press matters.
LEAD = 2
#: Cells between one press and the next, for the multi-press variant.
SPACING = 3
#: Cells of clear lane after the last press.
RUN_OUT = 2

#: Variants: the material, how many presses stand in the lane, how wide the
#: lane is under them, and the press's own cycle.  ``wide`` keeps the timing of
#: ``plain`` but gives the lane more room either side of the press, ``foundry``
#: puts two blows in the lane out of step with each other, and ``brisk`` keeps
#: a single press but on a shorter cycle -- the same rule, read faster.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
_VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'presses': 1, 'width': LANE,
              'period': 3.0, 'strike': 0.25, 'dwell': 0.4},
    'foundry': {'theme': 'foundry', 'presses': 2, 'width': LANE,
                'period': 2.6, 'strike': 0.22, 'dwell': 0.4},
    'brisk': {'theme': 'ice', 'presses': 1, 'width': LANE,
              'period': 2.0, 'strike': 0.2, 'dwell': 0.4},
    'wide': {'theme': 'rubber', 'presses': 1, 'width': LANE + 2,
             'period': 3.0, 'strike': 0.25, 'dwell': 0.4},
}


@fragment('crusher', tags=('gate', 'hazard'), cost=8.0,
          rule='go under the press between blows; stopping there is what it catches',
          variants=tuple(_VARIANTS))
def crusher(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, presses: Any=None, width: Any=None,
           period: Any=None, strike: Any=None, dwell: Any=None) -> Any:
    """A walled lane with ``presses`` presses striking across it in turn."""
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    presses = presses or settings['presses']
    width = width or settings['width']
    period = period or settings['period']
    strike = strike or settings['strike']
    dwell = dwell or settings['dwell']

    length = LEAD + (presses - 1) * SPACING + RUN_OUT + 1
    cells: dict = {}
    lane = Port(cell=entry.cell, facing=entry.facing, height=entry.height,
                width=width)
    end = pieces._lay(cells, lane, length)

    slabs = []
    for row in range(presses):
        at = lane.ahead(LEAD + row * SPACING)
        slabs.append(Crusher(cells=tuple(at.cells()), period=period,
                             strike=strike, dwell=dwell, travel=entry.facing,
                             phase=row * period / presses))

    walls = pieces._rails(cells, lane, length, width=width)
    exit_port = Port(cell=end.cell, facing=entry.facing, height=entry.height,
                     width=entry.width)
    return Piece(name='crusher', cells=cells, entry=entry, exits={'ok': exit_port},
                features=slabs + walls, theme=theme,
                rule='go under the press between blows; stopping there is what '
                     'it catches')
