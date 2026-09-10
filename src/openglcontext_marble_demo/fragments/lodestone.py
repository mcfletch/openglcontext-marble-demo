"""A corridor of magnets: the straight line is the one you have to work for.

Every other joiner in the library asks the player to steer *around* something --
a wall, a pit, a gap.  A lodestone run asks them to steer *against* something.
The posts stand beside the lane, alternately one side and the other, and each
pulls the marble towards it as it passes, so a marble left alone walks itself
into the wall it is nearest.  Holding the middle means leaning away from
whichever post is closest, and letting go at the right moment is how the pull
becomes a corner.

The posts are not solid: a marble runs through one rather than into it.  What the
piece is about is the line, and a post that could be hit would make it a slalom
with a force painted on.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = lodestone(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.magnet import Magnet
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails

__all__ = ['lodestone', 'RULE', 'VARIANTS']

RULE = 'lean away from the posts, or be walked into the wall'

#: ``posts`` is how many stand along the run and ``spacing`` how many cells apart,
#: which together fix the length.  ``strength`` is the pull at a post in metres
#: per second squared and ``reach`` how far it carries; ``repel`` turns the sign
#: round, so the posts push the marble to the middle instead and what the run
#: asks for is the nerve to stay off them.
#:
#: A post stands two cells off the middle of the corridor, which at the game's
#: cell size is 8 metres, so a reach short of that is a field the lane never
#: enters and a piece that does nothing.  Twelve leaves 3 m/s/s of pull on the
#: middle line at the plain strength, and the full 9 against the wall.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'foundry', 'posts': 3, 'spacing': 3, 'strength': 9.0,
              'reach': 12.0, 'repel': False},
    'strong': {'theme': 'foundry', 'posts': 3, 'spacing': 3, 'strength': 14.0,
               'reach': 12.0, 'repel': False},
    'long': {'theme': 'stone', 'posts': 5, 'spacing': 3, 'strength': 9.0,
             'reach': 12.0, 'repel': False},
    'repel': {'theme': 'stone', 'posts': 3, 'spacing': 3, 'strength': 11.0,
              'reach': 12.0, 'repel': True},
}


@fragment('lodestone', tags=('aim',), rule=RULE, cost=5.0,
          variants=tuple(VARIANTS))
def lodestone(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, posts: Any=None, spacing: Any=None,
              strength: Any=None, reach: Any=None, repel: Any=None) -> Any:
    """A walled run with magnets set alternately down either side of it."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    posts = posts or settings['posts']
    spacing = spacing or settings['spacing']
    strength = strength or settings['strength']
    reach = reach or settings['reach']
    repel = settings['repel'] if repel is None else repel

    width = LANE + 2                    # room to be pulled off the middle in
    cells: dict = {}
    features: list = []
    run = _lay(cells, entry, 1, width=width)
    corridor = Port(cell=run.ahead(1).cell, facing=entry.facing,
                    height=entry.height, width=width)
    length = posts * spacing + 1
    _lay(cells, corridor, length, width=width)

    across = entry.across()
    edge = width // 2
    hand = rng.choice((-1, 1))
    pull = -abs(strength) if repel else abs(strength)
    for post in range(posts):
        # Alternately one side and the other, so the pull the marble is leaning
        # against changes hands and the run cannot be crossed on one lean.
        side = hand * (1 if post % 2 == 0 else -1)
        cell = corridor.ahead(post * spacing + 1).cell
        features.append(Magnet(
            cell=(cell[0] + across[0] * side * edge,
                  cell[1] + across[1] * side * edge),
            strength=pull, radius=reach, axis=across))

    end = _lay(cells, corridor.ahead(length), 2, width=width)
    features.extend(_rails(cells, entry, length + 3, width=width))
    return Piece(name='lodestone', cells=cells, entry=entry,
                 exits={'ok': Port(cell=end.cell, facing=entry.facing,
                                   height=entry.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)
