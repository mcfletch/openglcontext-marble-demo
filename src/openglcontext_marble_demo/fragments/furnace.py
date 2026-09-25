"""A floor of fire with cool islands in it: cross it in hops, not in a crawl.

The :class:`~openglcontext_marble_demo.mechanisms.burner.Burner` counts time
rather than touches -- a body in the fire gains a second of heat per second and,
once it is out, loses it again -- so what a wide field of it punishes is
hesitating, not entering.  The islands are what turn that into a route: cool
tiles set across the chamber in a stagger, so a marble that goes island to island
sheds between hops what it took on the way, and one that tries the whole crossing
in a straight line takes it all at once.

There is no safe edge round the side.  A hard shoulder would make this the
:func:`~openglcontext_marble_demo.fragments.sand_pit.sand_pit` again, and the
sand pit is where the careful line belongs: sand costs seconds and fire costs the
marble, so a piece that can be threaded slowly is the one that should be threaded
slowly.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = furnace(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.burner import Burner
from openglcontext_marble_demo.pieces import LANE, Piece, Port, lay, rails

__all__ = ['furnace', 'RULE', 'VARIANTS']

RULE = 'island to island, and never stop on the fire'

#: ``length`` and ``width`` are the chamber in cells; ``islands`` is how many
#: cool tiles are set in each row of it, and ``burn_time`` how long the fire
#: takes.  A longer burn_time is a wider field a marble can afford to cross
#: badly, which is the difficulty dial.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'foundry', 'length': 4, 'width': 5, 'islands': 2,
              'burn_time': 1.6},
    # A longer chamber has to give more time or nothing crosses it: five cells
    # of fire is 1.85 seconds even at 14 m/s, because the marble is slowing the
    # whole way and the board's lean is only 8 degrees.
    'wide': {'theme': 'foundry', 'length': 5, 'width': 7, 'islands': 2,
             'burn_time': 2.6},
    'sparse': {'theme': 'foundry', 'length': 4, 'width': 5, 'islands': 1,
               'burn_time': 2.2},
    'stone': {'theme': 'stone', 'length': 3, 'width': 5, 'islands': 2,
              'burn_time': 1.4},
}


@fragment('furnace', tags=('hazard', 'aim'), rule=RULE, cost=8.0,
          variants=tuple(VARIANTS))
def furnace(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, length: Any=None, width: Any=None,
            islands: Any=None, burn_time: Any=None) -> Any:
    """A chamber floored with fire, with cool tiles staggered across it."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    length = length or settings['length']
    width = width or settings['width']
    islands = settings['islands'] if islands is None else islands
    burn_time = burn_time or settings['burn_time']

    cells: dict = {}
    features: list = []
    approach = lay(cells, entry, 2, width=width)
    chamber = Port(cell=approach.ahead(1).cell, facing=entry.facing,
                   height=entry.height, width=width)
    lay(cells, chamber, length, width=width)

    # One column is fire the whole way down and never gets an island.  Without
    # it every line across the chamber touches something cool sooner or later,
    # and since the burner sheds heat as fast as it takes it that is a crossing
    # a player can make without looking -- the piece would be a picture of a
    # fire.  Which column is the rng's, so it is not always the middle.
    across = entry.across()
    half = width // 2
    columns = list(range(-half, width - half))
    spine = rng.choice(columns)

    # The rest are dealt a row at a time and never in the same columns twice
    # running, so no column is a clear run down the middle either and every
    # crossing is a change of line.
    offerable = [column for column in columns if column != spine]
    cool: set = set()
    last: list = []
    for step in range(length):
        offered = [column for column in offerable if column not in last] or offerable
        last = rng.sample(offered, min(islands, len(offered)))
        for column in last:
            cell = chamber.ahead(step).cell
            cool.add((cell[0] + across[0] * column, cell[1] + across[1] * column))

    fire = sorted(set(_span(cells, chamber, length)) - cool)
    if fire:
        features.append(Burner(cells=tuple(fire), burn_time=burn_time))

    landing = lay(cells, chamber.ahead(length), 2, width=width)
    features.extend(rails(cells, entry, length + 4, width=width))
    return Piece(name='furnace', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.cell, facing=entry.facing,
                                   height=entry.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)


def _span(cells: Any, port: Any, length: Any) -> Any:
    """Every cell of ``length`` rows of the lane starting at ``port``."""
    return [cell for step in range(length) for cell in port.ahead(step).cells()
            if cell in cells]
