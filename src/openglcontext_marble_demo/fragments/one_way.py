"""A ledge you can go down and not come back up.

A marble rolls down a step and cannot roll up one — every slope in the game is
built of tilted tiles for exactly that reason.  So a **step left untilted** is a
one-way gate, made of nothing but geometry, and it is the cheapest commitment a
board can ask for: past it, the only way is on.

What that is for is making a choice final.  A split whose two ways can be
swapped between at leisure is not really two ways; put one of these on a branch
and the decision is a decision.
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import LANE, MAX_STEP, Piece, Port, lay, rails

__all__ = ['one_way', 'RULE', 'VARIANTS']

RULE = 'over the lip, and there is no going back'

#: ``lip`` is how far the step drops, and it is the slope budget exactly.  It
#: does not need to be more: a tile left untilted is a step whatever its height,
#: and a marble rolls down a step and cannot roll up one -- which is why every
#: slope in the game is built of tilted tiles.  Making it deeper would only put
#: a cliff on the board that nothing else is allowed to build.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'lip': MAX_STEP, 'before': 3, 'after': 3},
    'long': {'theme': 'stone', 'lip': MAX_STEP, 'before': 5, 'after': 5},
    'foundry': {'theme': 'foundry', 'lip': MAX_STEP, 'before': 4, 'after': 3},
    'slick': {'theme': 'ice', 'lip': MAX_STEP, 'before': 3, 'after': 5},
}


@fragment('one_way', tags=('gate', 'place'), rule=RULE, cost=1.0,
          variants=tuple(VARIANTS))
def one_way(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, lip: Any=None, before: Any=None,  # noqa: ARG001 the fragment builder protocol passes rng, and this fragment makes no random choice
            after: Any=None) -> Any:
    """A shelf, a lip that cannot be climbed, and the floor below it."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    lip = lip or settings['lip']
    before = before or settings['before']
    after = after or settings['after']

    cells: dict = {}
    above = lay(cells, entry, before, width=LANE)
    floor = round(entry.height - lip, 6)
    below = Port(cell=above.ahead(1).cell, facing=entry.facing, height=floor,
                 width=LANE)
    lay(cells, below, after, height=floor, width=LANE)
    features = rails(cells, entry, before + after + 1, width=LANE)
    return Piece(name='one_way', cells=cells, entry=entry,
                 exits={'ok': Port(cell=below.ahead(after - 1).cell,
                                   facing=entry.facing, height=floor,
                                   width=LANE)},
                 features=features, theme=theme, rule=RULE)
