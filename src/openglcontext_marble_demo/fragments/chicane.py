"""A quick left-right-left, and the rhythm it asks for.

The lane steps three cells across the board and then three cells back, so the
way in and the way out are the same line and nothing in between is on it.  Both
steps are made a cell at a time, a cell across per cell along, which keeps the
outside of each bend a slope of corners rather than a square pocket for the
board to press a marble into.

Nothing is blocked: the lane is open its whole length at any speed.  What the
piece asks is that the marble is *across* by the time the lane is, and how long
it has to get there is the length of a leg divided by how fast it is going.  A
marble crossing a leg at a walking pace has seconds to lean itself a lane over;
one crossing at a run has a fraction of a second, meets the wall on the outside
of the bend instead, and a wall met above the crash threshold returns a tenth of
what went into it.  So the piece has a speed it is quickest at, and above that
hurrying is the slow way through.

Long legs make a lazy chicane and short ones a sharp one, which is the whole of
the difference between the variants: ``leg`` is the time a player is given.
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import LANE, Piece, Port, lay, ring

__all__ = ['chicane', 'VARIANTS']

RULE = 'weave it in rhythm; too much speed and the walls have you'

#: The three axes a variant bundles: material (the theme, which is grip as much
#: as colour), layout (how long a leg is and which way the bends go) and effect
#: (how soon the rhythm breaks, which is what the leg length sets).
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'leg': 4, 'bends': (3,)},
    'tight': {'theme': 'stone', 'leg': 3, 'bends': (3,)},
    'foundry': {'theme': 'foundry', 'leg': 4, 'bends': (-3,)},
    'slick': {'theme': 'ice', 'leg': 5, 'bends': (3,)},
}


@fragment('chicane', tags=('aim', 'speed'), cost=4.0, rule=RULE,
          variants=tuple(VARIANTS))
def chicane(rng: Any, entry: Any, variant: Any='plain', **named: Any) -> Any:
    """A lane that steps across the board and back again.

    ``leg`` is how many cells of lane lie between one bend and the next, which
    is the time a player has to move over.  ``bends`` is where the lane's centre
    line goes, in cells across from the one it came in on; it returns to that
    line at the end whatever it says.  ``width`` is the lane.
    """
    settings = dict(VARIANTS[variant])
    settings.update(named)
    theme = settings.pop('theme')
    return _chicane(entry, theme=theme, **settings)


def _chicane(entry: Any, theme: Any='stone', leg: Any=4, bends: Any=(3,), width: Any=None) -> Any:
    width = entry.width if width is None else width
    across = entry.across()
    cells: dict = {}
    port = lay(cells, entry, leg, width=width)
    at = 0
    for offset in tuple(bends) + (0,):
        while at != offset:
            way = 1 if offset > at else -1
            at += way
            moved = Port(cell=(port.cell[0] + across[0] * way,
                               port.cell[1] + across[1] * way),
                         facing=entry.facing, height=entry.height, width=width)
            port = lay(cells, moved.ahead(1), 1, width=width)
        port = lay(cells, port.ahead(1), leg, width=width)
    exit_port = Port(cell=port.cell, facing=entry.facing, height=entry.height,
                     width=min(width, LANE))
    ways_out = set(entry.cells()) | set(exit_port.cells())
    return Piece(name='chicane', cells=cells, entry=entry, exits={'ok': exit_port},
                 features=ring(cells, set(cells), gaps=ways_out), theme=theme,
                 rule=RULE)
