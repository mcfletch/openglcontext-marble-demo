"""A column of rising air over a gap that no jump would ever clear.

The floor stops, and there is nothing built across what comes after it for cell
after cell — a span nothing could leap. A marble that merely trickles off the
lip drops into the open air over the gap and is gone, exactly as it would be
over any other fall too wide for it. What changes that is the shaft of rising
air standing in the gap: it is weaker than the pull trying to bring the marble
down, so it never holds one that has stopped, but it costs that pull enough of
its edge that a marble which is *still moving* crosses the width of the gap
before it has sunk out of the piece's reach. Speed is the only thing spent here
— there is no line to find and nothing to time, only how fast the marble was
going when the floor ran out.

A player who does not trust the jump never has to take it. A narrow bridge runs
beside the gap the whole way across, dropping just enough to meet the far
landing at its own height -- the quick way over, for a marble with the speed to
use it, and a slow, sure way for one without. On the *other* side a second,
steeper stair drops all the way to the shaft's own floor, well below the piece's
own, and comes up at its own door out, ``'sunk'``. Both are built the way
:mod:`~openglcontext_marble_demo.fragments.cistern`'s stair is: a walkable
descent, laid far enough from the lane's own edge that neither ever has to
answer for a cliff standing next to it, whatever height it settles at. Every way
through this piece gets a marble across; what differs is how many seconds it
costs.
"""
import math

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.updraft import Updraft
from openglcontext_marble_demo.pieces import LANE, MAX_STEP, Piece, Port, _lay, _rails, _slope

__all__ = ['updraft', 'RULE', 'VARIANTS']

RULE = 'carry speed across the gap; ease off and the draft only slows the sink'

#: How many cells of straight, walled approach lie before the lip -- long enough
#: that a marble is rolling true, rather than still settling from wherever it
#: was aimed onto the piece, by the time the floor runs out.
RUN_UP = 3

#: How many cells of floor lie on the far side of the gap before ``'ok'``.
LANDING = 4

#: How far below the piece's own height the far landing -- and so ``'ok'`` --
#: sits.  Small on purpose: what makes this piece a question is the gap's
#: *width*, not one more slope to carry a marble down.
LANDING_DROP = 0.6

#: How many cells the lower stair runs at the shaft's own floor before its own
#: door out, ``'sunk'``.
UNDERPASS = 3

#: The three axes a variant bundles: the material (grip as much as colour), the
#: layout (how wide the gap is), and the effect (how strong the draft is and how
#: far up the shaft it reaches).
VARIANTS = {
    'plain': {'theme': 'stone', 'gap': 4, 'strength': 9.2, 'depth': 6.0,
              'reach': 7.5},
    'wide': {'theme': 'stone', 'gap': 6, 'strength': 9.5, 'depth': 6.3,
             'reach': 7.8},
    'foundry': {'theme': 'foundry', 'gap': 4, 'strength': 9.4, 'depth': 7.2,
                'reach': 8.5},
    'gale': {'theme': 'ice', 'gap': 3, 'strength': 8.6, 'depth': 5.4,
             'reach': 6.6},
}


@fragment('updraft', tags=('speed', 'hazard'), cost=8.0, rule=RULE,
          variants=tuple(VARIANTS))
def updraft(rng, entry, variant='plain', **named):
    """A run-up, a gap with a shaft of rising air over it, and two ways down.

    ``gap`` is how many cells wide the open span is. ``strength`` is the
    shaft's lift in metres per second squared, always kept below the vertical
    pull gravity gives a falling body, and ``reach`` how far above the shaft's
    own floor -- ``depth`` below the piece's own -- the lift still acts. See
    :mod:`~openglcontext_marble_demo.mechanisms.updraft` for what those two
    numbers do to a body inside the shaft.
    """
    settings = dict(VARIANTS[variant])
    settings.update(named)
    theme = settings.pop('theme')
    return _updraft(rng, entry, theme=theme, **settings)


def _updraft(rng, entry, theme='stone', gap=4, strength=9.2, depth=6.0, reach=7.5,
             width=None):
    width = entry.width if width is None else width
    half = width // 2
    cells: dict = {}
    features: list = []

    lip = _lay(cells, entry, RUN_UP, width=width)

    # The gap itself: never laid, so it carries no floor at all -- a marble
    # over it is off the track exactly as it would be over any other void.
    shaft_cells = [cell for step in range(gap)
                   for cell in lip.ahead(step + 1).cells()]

    landing_height = entry.height - LANDING_DROP
    landing_entry = Port(cell=lip.ahead(gap + 1).cell, facing=entry.facing,
                         height=landing_height, width=width)
    landing = _lay(cells, landing_entry, LANDING, height=landing_height, width=width)

    features.append(Updraft(cells=shaft_cells, strength=strength,
                            floor=entry.height - depth, reach=reach))
    features.extend(_rails(cells, entry, RUN_UP, width=width))
    features.extend(_rails(cells, landing_entry, LANDING, width=width))

    across = entry.across()
    hand = rng.choice((-1, 1))

    # The bridge: one cell wide, standing where ``drop``'s own ledge does --
    # immediately outside the lane's edge -- and dropping only as far as the
    # landing itself does, over the same number of rows the gap and the
    # landing's approach take between them.  It meets the landing at exactly
    # its own height on exactly its own row, so the two are neighbours that
    # agree rather than a cliff standing next to a floor.
    bridge_start = Port(cell=(lip.cell[0] + across[0] * hand * (half + 1),
                              lip.cell[1]),
                        facing=entry.facing, height=entry.height, width=1)
    _slope(cells, features, bridge_start, gap + 2, -LANDING_DROP, width=1)
    features.extend(_rails(cells, bridge_start, gap + 2,
                           sides=('left', 'right'), width=1))

    # The stair down to the shaft's own floor: on the *other* side, so it never
    # shares a column with the bridge, and stood two cells clear of the lane
    # rather than one -- the bridge only ever drifts as far as the landing
    # does, but this one falls the whole depth of the shaft, and two cells
    # clear is what keeps it from ever bordering the landing while it does.
    elbow_near = (lip.cell[0] + across[0] * -hand * (half + 1), lip.cell[1])
    elbow_far = (lip.cell[0] + across[0] * -hand * (half + 2), lip.cell[1])
    cells[elbow_near] = entry.height
    cells[elbow_far] = entry.height
    stair_start = Port(cell=elbow_far, facing=entry.facing, height=entry.height,
                       width=1)
    foot = _slope(cells, features, stair_start, gap, -depth, width=1)
    # ``_slope`` may lengthen the run past ``gap`` to hold the slope budget;
    # this is the same rule it uses to decide how many steps that took.
    descent_steps = max(gap, math.ceil(depth / MAX_STEP) + 1, 2)
    features.extend(_rails(cells, stair_start, descent_steps,
                           sides=('left', 'right'), width=1))
    stair_end = _lay(cells, foot.ahead(1), UNDERPASS, height=foot.height, width=1)
    features.extend(_rails(cells, foot.ahead(1), UNDERPASS,
                           sides=('left', 'right'), width=1))
    sunk_exit = Port(cell=stair_end.cell, facing=entry.facing,
                     height=foot.height, width=1)

    return Piece(name='updraft', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.cell, facing=entry.facing,
                                   height=landing_height, width=min(width, LANE)),
                       'sunk': sunk_exit},
                 features=features, theme=theme, rule=RULE)
