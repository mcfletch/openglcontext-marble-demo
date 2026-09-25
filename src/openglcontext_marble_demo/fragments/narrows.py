"""A hall that closes to one lane and opens again, and the aim it asks for.

The mouth is wide enough to arrive anywhere across it.  From there the walls
converge, a lane at a time, to a throat a single cell across, and then open out
again into somewhere to land.  Every cell of it is floor: there is nothing to
fall into and nothing to time.  What it asks is that the marble is **on the
centre line** by the time the walls reach it.

Speed is what decides whether that is easy.  The lane a marble has to move
across is fixed by the shape; the time it has to move across it is the length of
the funnel divided by how fast it is going.  Rolling in gently, a player has
seconds to gather the line and the throat costs nothing; arriving at a run and
off to one side, the converging wall arrives first, and a wall met at that speed
returns a tenth of what went into it.

The throat is the same width whatever the mouth is, so ``mouth`` is how far
wrong a player can be and ``taper`` is how long they have to put it right.
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import LANE, Piece, Port, lay, ring

__all__ = ['narrows', 'VARIANTS']

RULE = 'be on the line by the throat, or the closing wall will have you'

#: Material, layout and effect, per variant: what it is made of, how wide the
#: mouth is against how long the taper is, and how long the throat holds.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'mouth': 7, 'taper': 2, 'hold': 2},
    'funnel': {'theme': 'stone', 'mouth': 9, 'taper': 1, 'hold': 1},
    'foundry': {'theme': 'foundry', 'mouth': 7, 'taper': 3, 'hold': 3},
    'slick': {'theme': 'ice', 'mouth': 7, 'taper': 2, 'hold': 2},
}


@fragment('narrows', tags=('aim', 'gate'), cost=5.0, rule=RULE,
          variants=tuple(VARIANTS))
def narrows(rng: Any, entry: Any, variant: Any='plain', **named: Any) -> Any:  # noqa: ARG001 the fragment builder protocol passes rng, and this fragment makes no random choice
    """A converging hall with a one-cell throat in the middle of it.

    ``mouth`` is how wide it starts (and so how far off line a marble may
    arrive), ``taper`` how many cells of lane each step inward gets, ``hold``
    how long the throat runs before it opens out.
    """
    settings = dict(VARIANTS[variant])
    settings.update(named)
    theme = settings.pop('theme')
    return _narrows(entry, theme=theme, **settings)


def _narrows(entry: Any, theme: Any='stone', mouth: Any=7, taper: Any=2, hold: Any=2, throat: Any=1) -> Any:
    cells: dict = {}
    port = lay(cells, entry, taper, width=mouth)
    for width in range(mouth - 2, throat, -2):
        port = lay(cells, port.ahead(1), taper, width=width)
    port = lay(cells, port.ahead(1), hold, width=throat)
    for width in range(throat + 2, mouth + 1, 2):
        port = lay(cells, port.ahead(1), taper, width=width)
    exit_port = Port(cell=port.cell, facing=entry.facing, height=entry.height,
                     width=LANE)
    ways_out = set(entry.cells()) | set(exit_port.cells())
    return Piece(name='narrows', cells=cells, entry=entry,
                 exits={'ok': exit_port},
                 features=ring(cells, set(cells), gaps=ways_out), theme=theme,
                 rule=RULE)
