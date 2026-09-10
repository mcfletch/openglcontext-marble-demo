"""A peg board with one fast slot among ordinary ones.

The peg board exists as a mechanism and, until this chapter, no board could hold
one.  What makes it a chapter is what surrounds it: a mouth wide enough that
where you enter is a decision, the board itself, and a landing that gathers
every slot back onto one way on — because a fragment that only worked if you hit
the right slot would be a fragment most runs failed for no reason they could see.

Entering left takes the left slot about two-thirds of the time and the right
slot about one in twenty, so aim shifts the odds hard without settling them.
The fast slot leaves at 16 m/s against about 7 for an ordinary one, which is the
whole reward: not a shortcut, a *run-up*, and what a player does with it is the
next chapter's business.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = plinko(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.pegs import PegBoard
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails

__all__ = ['plinko', 'RULE', 'VARIANTS']

RULE = 'aim for the fast slot; the pegs will argue about it'

#: ``width`` is how many slots, ``length`` how many rows of pegs, ``drop`` how
#: far the board falls over them, and ``fast_slot`` which slot pays.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'foundry', 'width': 3, 'length': 4, 'drop': 3.5,
              'fast_slot': 1},
    'broad': {'theme': 'foundry', 'width': 5, 'length': 5, 'drop': 4.5,
              'fast_slot': 2},
    'steep': {'theme': 'stone', 'width': 3, 'length': 6, 'drop': 5.4,
              'fast_slot': 0},
    'slick': {'theme': 'ice', 'width': 5, 'length': 4, 'drop': 3.5,
              'fast_slot': 4},
}


@fragment('plinko', tags=('luck', 'speed'), rule=RULE, cost=4.0,
          variants=tuple(VARIANTS))
def plinko(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, width: Any=None, length: Any=None,
           drop: Any=None, fast_slot: Any=None) -> Any:
    """A mouth, a peg board, and a landing that gathers the slots back."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    width = width or settings['width']
    length = length or settings['length']
    drop = drop or settings['drop']
    fast_slot = settings['fast_slot'] if fast_slot is None else fast_slot

    cells: dict = {}
    features: list = []

    # A mouth as wide as the board, so where a player enters is a choice.
    mouth = _lay(cells, entry, 2, width=width)

    board = PegBoard(cell=mouth.ahead(1).cell, facing=entry.facing, width=width,
                     length=length, drop=drop, top=entry.height,
                     fast_slot=fast_slot)
    # The board tiles its own cells; the level lays them, as its docstring says.
    cells.update(board.cells())
    features.append(board)

    bottom = min(board.cells().values())
    rows = max(row for _, row in board.cells())
    landing = Port(cell=(entry.cell[0], rows + 1), facing=entry.facing,
                   height=bottom, width=width)
    _lay(cells, landing, 2, height=bottom, width=width)
    features.extend(_rails(cells, entry, length + 6, width=width))
    return Piece(name='plinko', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.ahead(1).cell,
                                   facing=entry.facing, height=bottom,
                                   width=LANE)},
                 features=features, theme=theme, rule=RULE)
