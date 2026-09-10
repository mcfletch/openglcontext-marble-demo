"""A straight run with bars sweeping across it, and a moment each one is open.

Three arms abreast fill the lane between them, so a row is a door: shut while
the bars lie across the way, open while they lie along it.  The rows turn at
different rates, so what a player is reading is not one door but the pattern the
whole set makes, and the run through is the moment the pattern lines up.

The rates are worked out rather than picked.  A marble crossing at ``pace``
metres a second reaches the ``k``-th row after a known time; the row's rate is
the one that has it lying along the lane just then, taken as few half-turns as
:data:`MIN_RATE` allows so the bars are seen to move.  A marble that leaves the
mouth at some other moment meets the pattern out of step and is turned back into
the run to try again, which costs it the whole length of the piece.

Nothing here stops a player: an arm sweeps a marble aside rather than ending the
run, and the lane is walled, so being caught is a delay.  What the piece is worth
is what the delay costs against the clock, and that is the whole spread.  The
autopilot crosses ``brisk`` in 8.3 seconds if it meets the arms right, against
8.0 down the same lane with them taken out, and is still trying at 20 seconds if
it meets them wrong.

``tests/fragments/test_gauntlet.py`` measures that by starting the arms part way
round, which is the same question as a marble reaching the mouth part way
through their turn.
"""
import math
from typing import Any

from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import CELL_SIZE, RotatingArm
from openglcontext_marble_demo.pieces import Piece

__all__ = ['gauntlet', 'rates']

#: Cells of clear lane before the first row of arms, so a marble is up to speed
#: and aimed by the time the pattern starts to matter.
LEAD = 2
#: Cells between one row of arms and the next.
SPACING = 2
#: Cells of clear lane after the last row.
RUN_OUT = 2

#: Slowest an arm may turn, in radians a second -- 3.5 is a little over 33 rpm,
#: which carries the tip of a cell-long bar at about 6 metres a second.  That is
#: the floor at which a row is a hazard rather than a fence: turning slower than
#: the marble travels, a bar nudges a mistimed arrival instead of turning it
#: back, and a run through costs about the same whenever it is made.
MIN_RATE = 3.5

#: Variants: the material, how many rows of arms, the pace the phasing is worked
#: out for in metres a second, and how long each bar is in metres.  A bar shorter
#: than a cell leaves a gap between neighbours that a marble can be steered
#: through whatever the timing, which is what ``sparse`` is.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
_VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'rows': 3, 'pace': 3.4, 'bar': 3.6},
    'foundry': {'theme': 'foundry', 'rows': 4, 'pace': 3.4, 'bar': 3.6},
    'brisk': {'theme': 'ice', 'rows': 3, 'pace': 4.5, 'bar': 3.6},
    'sparse': {'theme': 'stone', 'rows': 3, 'pace': 3.0, 'bar': 2.6},
}


def rates(rows: Any, pace: Any, facing: Any, cell_size: float=CELL_SIZE) -> Any:
    """The turn rate for each row of arms, in rpm.

    An arm points along its own +X at rest and turns about the vertical, so a
    row lies *along* a lane running east-west at no rotation at all and across
    one running north-south.  ``facing`` decides which, and the answer is the
    rate that has the row lying along the lane when a marble travelling at
    ``pace`` arrives at it.
    """
    open_at = math.pi / 2.0 if facing[1] else 0.0
    found = []
    for row in range(rows):
        arrive = (LEAD + row * SPACING) * cell_size / pace
        turns = 0
        while (open_at + math.pi * turns) / arrive < MIN_RATE:
            turns += 1
        rate = (open_at + math.pi * turns) / arrive
        found.append(rate * 30.0 / math.pi)
    return found


@fragment('gauntlet', tags=('gate', 'hazard'), cost=5.0,
          rule='go through when the arms are lying along the lane, not across it',
          variants=tuple(_VARIANTS))
def gauntlet(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, rows: Any=None, pace: Any=None,
             bar: Any=None) -> Any:
    """A walled straight with ``rows`` rows of arms sweeping across it."""
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    rows = rows or settings['rows']
    pace = pace or settings['pace']
    bar = bar or settings['bar']

    length = LEAD + (rows - 1) * SPACING + RUN_OUT + 1
    cells: dict = {}
    end = pieces._lay(cells, entry, length)
    arms = []
    for row, rpm in enumerate(rates(rows, pace, entry.facing)):
        at = entry.ahead(LEAD + row * SPACING)
        for cell in at.cells():
            arms.append(RotatingArm(cell=cell, length=bar, rpm=rpm))
    return Piece(name='gauntlet', cells=cells, entry=entry, exits={'ok': end},
                 features=arms + pieces._rails(cells, entry, length),
                 theme=theme,
                 rule='go through when the arms are lying along the lane, not across it')
