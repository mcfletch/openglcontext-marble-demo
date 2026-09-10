"""Stones with air between them, and the speed it takes to cross it.

Each stone sits a terrace below the one before it, and between them is a gap
with nothing over it.  A marble that leaves a stone is a projectile: it covers
the gap at the speed it left with, and it falls at the rate everything falls.
Fast enough and it is over the far edge while it is still high; short of that it
is a metre down with the gap still ahead of it, which is the point at which the
game stops counting it as jumping and starts counting it as fallen.

So the rule is about **committing**.  There is no line to find and nothing to
time -- the only thing that carries a marble across is the speed it already had,
and the one way to fail is to arrive without it.  Braking on the stone before a
gap is how a run ends here, and so is easing off half way along the run: every
gap asks the same question again, and the stones are short.

The stones are laced together at their corners -- one cell of lip, on alternating
sides, stepping down between the terraces -- so the piece is one region and a
player picking their way at walking pace can get across it.  That is the slow
way round, and it is the only one: down the middle there is nothing but air.
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import LANE, MAX_STEP, Piece, Port, _lay

__all__ = ['stepping_stones', 'VARIANTS']

RULE = 'carry enough speed to cross the gaps; easing off is how you fall in'

#: Material, layout and effect: what the stones are made of, how many there are
#: and how long each is, and how much air is between them.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'stones': 4, 'run': 2, 'gap': 1},
    'short': {'theme': 'stone', 'stones': 5, 'run': 1, 'gap': 1},
    'foundry': {'theme': 'foundry', 'stones': 4, 'run': 3, 'gap': 1},
    'wide': {'theme': 'stone', 'stones': 3, 'run': 3, 'gap': 2},
}


@fragment('stepping_stones', tags=('speed', 'hazard'), cost=8.0, rule=RULE,
          variants=tuple(VARIANTS))
def stepping_stones(rng: Any, entry: Any, variant: Any='plain', **named: Any) -> Any:
    """A run of stones with gaps between them, each stone a terrace lower.

    ``stones`` is how many there are and ``run`` how many cells long each is --
    together, how much room there is to build speed back up.  ``gap`` is how
    many cells of air lie between one and the next, which is what sets the speed
    the piece asks for.
    """
    settings = dict(VARIANTS[variant])
    settings.update(named)
    theme = settings.pop('theme')
    return _stepping_stones(entry, theme=theme, **settings)


def _stepping_stones(entry: Any, theme: Any='stone', stones: Any=4, run: Any=2, gap: Any=1, step: Any=MAX_STEP - 0.05,
                     width: Any=None) -> Any:
    width = entry.width if width is None else width
    across = entry.across()
    cells: dict = {}
    height = entry.height
    port = _lay(cells, entry, run, height=height, width=width)
    for index in range(stones - 1):
        # The lip alternates sides, so no lane of the piece runs clear through
        # it: whichever side a player creeps down, the next gap is on the other.
        side = 1 if index % 2 == 0 else -1
        for cell in range(gap):
            height = round(height - step, 6)
            at = port.ahead(cell + 1)
            cells[(at.cell[0] + across[0] * side,
                   at.cell[1] + across[1] * side)] = height
        height = round(height - step, 6)
        landing = Port(cell=port.ahead(gap + 1).cell, facing=entry.facing,
                       height=height, width=width)
        port = _lay(cells, landing, run, height=height, width=width)
    exit_port = Port(cell=port.cell, facing=entry.facing, height=height,
                     width=min(width, LANE))
    return Piece(name='stepping_stones', cells=cells, entry=entry,
                 exits={'ok': exit_port}, theme=theme, rule=RULE)
