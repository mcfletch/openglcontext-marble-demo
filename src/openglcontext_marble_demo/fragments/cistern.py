"""A pool that sinks you, and a plug that gives when you reach the bottom.

The water sinks a marble at about a third of the speed of falling, and the plug
in its floor opens only when the marble arrives at it — which a player cannot
know until it has happened, so the first cistern is a surprise and every one
after is a decision about whether the quick way through is worth the wait.

The mechanism's own documentation says a board using it **wants somewhere under
the plug to land**, because the marble leaves through the floor and keeps
falling.  That is most of what this chapter is: the pool, and the chamber
underneath that catches what comes out of it and carries on.

And a stair down beside it, which is the long way to the same place.  Not
politeness: the drop through the plug is several metres, which is not a step a
marble can roll, so without the stair the two halves of this chapter would be
joined only by falling — and a piece a marble cannot walk between is a piece
half of which is unreachable to anything reasoning about the board.  The stair
is also the answer to the water for a player who would rather not wait.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = cistern(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.water import Water
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails, _slope

__all__ = ['cistern', 'RULE', 'VARIANTS']

RULE = 'sink through, and wait to find out the floor gives'

#: ``depth`` is how far the marble sinks before it reaches the plug, and
#: ``below`` how far under the pool the chamber that catches it sits.  The
#: chamber must be deeper than the plug or the marble lands back in the pool.
VARIANTS = {
    'plain': {'theme': 'stone', 'depth': 3.0, 'below': 4.5, 'run': 3},
    'deep': {'theme': 'stone', 'depth': 4.5, 'below': 6.3, 'run': 3},
    'cold': {'theme': 'ice', 'depth': 3.0, 'below': 4.5, 'run': 4},
    'foundry': {'theme': 'foundry', 'depth': 3.0, 'below': 5.4, 'run': 4},
}


@fragment('cistern', tags=('hazard', 'gate'), rule=RULE, cost=3.0,
          variants=tuple(VARIANTS))
def cistern(rng, entry, variant='plain', theme=None, depth=None, below=None,
            run=None):
    """An approach, a pool with a plug, and the chamber that catches you."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    depth = depth or settings['depth']
    below = below or settings['below']
    run = run or settings['run']

    cells: dict = {}
    features: list = []

    approach = _lay(cells, entry, 2, width=LANE)
    pool_at = approach.ahead(1)
    _lay(cells, pool_at, 1, width=LANE)
    features.append(Water(cell=pool_at.cell, depth=depth))

    # The stair down beside the pool: the long way to the same place, and what
    # makes the two halves of the piece one region rather than two.
    across = entry.across()
    hand = rng.choice((-1, 1))
    beside = Port(cell=(approach.cell[0] + across[0] * hand * 2,
                        approach.cell[1] + across[1] * hand * 2),
                  facing=entry.facing, height=entry.height, width=LANE)
    _lay(cells, Port(cell=(approach.cell[0] + across[0] * hand,
                           approach.cell[1] + across[1] * hand),
                     facing=entry.facing, height=entry.height, width=1), 1)
    floor = entry.height - below
    stair_features: list = []
    # The stair decides where the chamber goes, rather than the other way about:
    # a slope is only allowed to fall so fast, so how many cells it needs is a
    # consequence of the depth and not something to be chosen and then violated.
    foot = _slope(cells, stair_features, beside, 4, floor - entry.height,
                  width=LANE)
    features.extend(stair_features)

    # The chamber that catches what the plug lets out, set two cells beyond the
    # pool rather than against it.  Against it, the pool's rim and the chamber
    # floor would be neighbouring cells several metres apart -- a step nothing
    # can roll, which is a cliff to everything that reasons about the board even
    # though a marble arrives there by falling.  Two cells along, they do not
    # touch, and the stair is the only way between them that walks.
    catcher = Port(cell=(pool_at.cell[0], foot.cell[1]), facing=entry.facing,
                   height=floor, width=LANE + 2)
    _lay(cells, catcher, run, height=floor, width=LANE + 2)
    features.extend(_rails(cells, catcher, run, width=LANE + 2))
    return Piece(name='cistern', cells=cells, entry=entry,
                 exits={'ok': Port(cell=catcher.ahead(run - 1).cell,
                                   facing=entry.facing, height=floor,
                                   width=LANE)},
                 features=features, theme=theme, rule=RULE)
