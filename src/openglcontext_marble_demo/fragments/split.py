"""Two ways on from one room, and the player picks which.

Every other joiner asks a player to do a thing well.  This one asks them to
*choose*, which is a different question and the one a graph of chapters exists
to be able to put.  The way straight on is a plank one cell wide with nothing
built beside it, and it wants the marble travelling straight.  The way round is
three cells wide and walled the whole distance, and it costs the detour: the
autopilot takes 7.2 seconds over the plank and 12.6 round the side.

Both ways leave at the same distance down the board and at the same height, so a
story can send ``ok`` and ``long`` to the same chapter and have them meet there
-- :func:`~openglcontext_marble_demo.stories._rejoin` joins whichever arrives
second to where the first laid it.  Sending them to *different* chapters works
just as well, and is how a board rewards the plank with more than time.

The room in front of the fork is walled on every side but the way in and the two
ways out, so a marble that takes neither ends up back in front of them.
"""
from typing import Any

from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import Piece, Port

__all__ = ['split']

#: How deep the room in front of the fork is, in cells.  Three gives a marble
#: room to line up on the plank, which is the whole of what it is for.
FORK = 3

#: Variants: the material, how long the plank is, and how far round the long way
#: goes in cells sideways.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
_VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'plank': 4, 'detour': 5},
    'long': {'theme': 'stone', 'plank': 6, 'detour': 6},
    'foundry': {'theme': 'foundry', 'plank': 4, 'detour': 8},
    'slick': {'theme': 'ice', 'plank': 5, 'detour': 5},
}

#: Least detour that leaves a clear column between the plank's mouth and the
#: long way where the two of them end.
MIN_DETOUR = 4


@fragment('split', tags=('aim', 'place'), cost=5.0,
          rule='take the plank or pay for the long way round',
          variants=tuple(_VARIANTS))
def split(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, plank: Any=None, detour: Any=None,
          side: Any=None) -> Any:
    """A room with two ways out: ``ok`` over the plank, ``long`` round the side.

    ``side`` is which way round the long one goes, ``1`` or ``-1`` across the
    facing; left to the rng when it is not asked for.
    """
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    plank = plank or settings['plank']
    detour = max(detour or settings['detour'], MIN_DETOUR)
    if side is None:
        side = rng.choice((1, -1))
    across = entry.across()
    sideways = (across[0] * side, across[1] * side)

    cells: dict = {}
    room = pieces._lay(cells, entry, FORK, width=entry.width + 2)

    # The quick way: one cell across, and nothing built beside it.
    narrow = Port(cell=room.cell, facing=entry.facing, height=entry.height, width=1)
    plank_end = pieces._lay(cells, narrow.ahead(1), plank, width=1)
    mouth = pieces._lay(cells, plank_end.ahead(1), 1, width=entry.width)
    quick = Port(cell=mouth.cell, facing=entry.facing, height=entry.height,
                 width=entry.width)

    # The long way: out of the room's side, then down to the row the plank ends
    # on, so the two ways are level with one another where they finish.
    turn = Port(cell=room.ahead(-1).cell, facing=sideways, height=entry.height,
                width=entry.width)
    out = pieces._lay(cells, turn.ahead(1), detour)
    down = Port(cell=out.cell, facing=entry.facing, height=entry.height,
                width=entry.width)
    along = pieces._lay(cells, down.ahead(1), plank + FORK - 1)
    long_way = Port(cell=along.cell, facing=entry.facing, height=entry.height,
                    width=entry.width)

    # Walled wherever the floor stops, except at the three mouths and along the
    # plank.  ``_ring`` puts a wall only where there is no neighbouring cell, so
    # one call fences the room and the long way round without ever putting a
    # wall across the way between them.
    open_at = set(entry.cells()) | set(quick.cells()) | set(long_way.cells())
    open_at |= {narrow.ahead(step + 1).cell for step in range(plank)}
    return Piece(name='split', cells=cells, entry=entry,
                 exits={'ok': quick, 'long': long_way},
                 features=pieces._ring(cells, set(cells), gaps=open_at),
                 theme=theme,
                 rule='take the plank or pay for the long way round')
