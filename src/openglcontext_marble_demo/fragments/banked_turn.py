"""A corner whose outside is floor rather than air.

Two straight legs and the square where they meet, laid whole and walled all the
way round.  The floor of that square rises toward the outside of the bend, a
terrace at a time, each rise tilted so it is a surface rather than a step, and
the wall stands on top of it.

That is the difference from the other corner in the library.
:func:`~openglcontext_marble_demo.pieces.hairpin` lays its two legs across each
other, which leaves the outside of the bend missing, so a marble arriving faster
than it can turn goes out through the gap and the corner has to be braked for.
Here there is nothing to go out through: a marble that runs wide runs *up*, and
the worst it meets is the wall at the top.  Arriving quickly costs it its line
rather than its run, and the corner is quicker for having been taken quickly --
which makes it the one corner a player is not asked to slow down for.

``run`` is how long the legs are, and a long approach is a gentler corner
because the marble arrives already turning.
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import Ramp
from openglcontext_marble_demo.pieces import MAX_STEP, Piece, Port, _lay, _ring

__all__ = ['banked_turn', 'VARIANTS']

RULE = 'find the speed for it: a crawl and a charge both cost you'

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

#: A :class:`~openglcontext_marble_demo.level.Ramp`'s ``boost_speed`` is the
#: speed it brings a marble up to along the tile's own direction, and a figure
#: below every speed brings it up to nothing.  The bank is a surface to roll on
#: rather than something that pushes: what leaves the corner is what arrived at
#: it, less what the climb and the wall cost.
NO_BOOST = -1e6

#: Material, layout and effect: what it is made of, how long the legs are and
#: which way it turns, and how high each terrace of the bank stands.
VARIANTS = {
    'plain': {'theme': 'stone', 'run': 4, 'turn': 1, 'bank': MAX_STEP - 0.05},
    'sweeping': {'theme': 'stone', 'run': 6, 'turn': -1, 'bank': MAX_STEP - 0.05},
    'foundry': {'theme': 'foundry', 'run': 4, 'turn': -1, 'bank': MAX_STEP - 0.05},
    'shallow': {'theme': 'stone', 'run': 4, 'turn': 1, 'bank': 0.4},
}


@fragment('banked_turn', tags=('speed', 'aim'), cost=2.0, rule=RULE,
          variants=tuple(VARIANTS))
def banked_turn(rng, entry, variant='plain', **named):
    """A right-angle whose outside is built up into a bank.

    ``run`` is how long each leg is, ``turn`` which way the corner goes (``1``
    for the way the entry's ``across`` points, ``-1`` for the other), ``bank``
    how much each terrace of the bank is raised and ``banks`` how many of them
    there are.  A ``bank`` of zero is the same corner flat, which is what the
    bank is worth measuring against.
    """
    settings = dict(VARIANTS[variant])
    settings.update(named)
    theme = settings.pop('theme')
    return _banked_turn(entry, theme=theme, **settings)


def _banked_turn(entry, theme='stone', run=4, turn=1, bank=MAX_STEP - 0.05, banks=1,
                 width=None):
    width = entry.width if width is None else width
    half = width // 2
    banks = min(banks, half)
    across = entry.across()
    aside = (across[0] * turn, across[1] * turn)
    cells: dict = {}
    straight = _lay(cells, entry, run, width=width)
    # The square where the legs meet is laid whole.  Two legs crossing leave the
    # outside of the bend missing, which is a hole exactly where the bank goes.
    _lay(cells, straight, half + 1, width=width)
    corner = Port(cell=straight.cell, facing=aside, height=entry.height, width=width)
    end = _lay(cells, corner.ahead(half + 1), run, width=width)
    for cell in cells:
        cells[cell] = round(entry.height + bank * _bank_lanes(entry, cell, run,
                                                              half, banks), 6)
    exit_port = Port(cell=end.cell, facing=aside, height=cells[end.cell], width=width)
    ways_out = set(entry.cells()) | set(exit_port.cells())
    return Piece(name='banked_turn', cells=cells, entry=entry,
                 exits={'ok': exit_port},
                 features=_ramps(cells) + _ring(cells, set(cells), gaps=ways_out),
                 theme=theme, rule=RULE)


def _bank_lanes(entry, cell, run, half, banks):
    """How many terraces above the floor of the bend ``cell`` sits.

    The bank is the last ``banks`` lanes of the square the legs meet in, carried
    on down the outer edge of the leg that leaves.  It rises in one direction
    only, which is what lets each terrace be a single tilted tile: a surface
    that rose two ways at once would want two tiles on one cell, and what a
    marble would meet instead is the riser of whichever was left out.
    """
    along = ((cell[0] - entry.cell[0]) * entry.facing[0]
             + (cell[1] - entry.cell[1]) * entry.facing[1])
    return min(max(along - (run - 1 + half - banks), 0), banks)


def _ramps(cells):
    """A tilted tile on every cell the bank steps up from.

    Terraces at different heights are a staircase, and a marble meets each riser
    as a wall in front of it; a
    :class:`~openglcontext_marble_demo.level.Ramp` tilts the tile so its raised
    edge meets the terrace above and the bank is something to roll up.
    """
    made = []
    for cell, height in sorted(cells.items()):
        for step in _STEPS:
            beside = (cell[0] + step[0], cell[1] + step[1])
            rise = cells.get(beside, height) - height
            if rise > 1e-6:
                made.append(Ramp(cell=cell, direction=step, rise=round(rise, 6),
                                 boost_speed=NO_BOOST))
                break
    return made
