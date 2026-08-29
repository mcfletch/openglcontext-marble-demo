"""Several lanes into one launcher, and the lane picks the landing.

A shared mouth, wide enough that where a marble sits across it is a decision,
narrows into one lane per barrel.  Each barrel is a
:class:`~openglcontext_marble_demo.level.Ramp` with ``launch=True`` and its own
``launch_up``, and every one of them fires unconditionally the moment a marble
crosses it -- so unlike :mod:`~openglcontext_marble_demo.fragments.drop`, where
the rule is the speed a marble carries to the lip, here the aim is made *before*
the launcher and the speed barely matters. A gentle barrel drops its marble a
short way on; a hard one throws it much further down the hall, and further is
better, because it is that much less floor left to cross before the next
chapter starts.

Every barrel but the hardest ends in a wall: a marble that took a weaker lane
comes down inside its own corridor and stops there, unable to drift sideways
into a neighbour's, because each lane is walled along both sides for its whole
length.  Only the strongest barrel's lane stays open, which is what makes it
``'ok'``; the weakest is named ``'short'`` so a story can send it somewhere of
its own, and any lanes in between are reachable but unnamed -- a marble that
takes one is not stuck, since the way back up through the mouth is never
walled, only slower for having tried.

Measured landing distance past the mouth, metres, entering at 3, 6 and 9 m/s
(:data:`~openglcontext_marble_demo.fragments.cannon.BOOST_SPEED` brings every
lane up to the same speed along its own direction, which is why a lane's reach
barely moves against a threefold change in how fast a marble arrived):

=========  ==============  ==============  ==============  ==============
variant    ``short``       middle lane(s)                  ``ok``
=========  ==============  ==============  ==============  ==============
``twin``   10.3/10.4/12.1  --                              16.6/16.5/20.0
``triple`` 10.3/10.4/12.1  14.7/14.6/17.5                  18.0/17.9/21.3
``quad``   10.3/10.4/12.1  13.7/14.0/16.6  15.2/15.7/18.5   18.0/17.9/21.3
``far``    10.3/10.4/12.1  16.6/16.5/20.0                  20.0/19.9/24.3
=========  ==============  ==============  ==============  ==============

``ok`` clears ``short`` by at least 4.4 m at every one of those speeds, and by
as much as 7.8 m on ``far`` -- which is the whole of "much better", and why a
story can afford to send the two exits somewhere genuinely different.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = cannon(random.Random(1), Port(cell=(0, 0)))
    >>> sorted(piece.exits)
    ['ok', 'short']
"""
from openglcontext_marble_demo.level import Ramp, Wall
from openglcontext_marble_demo.pieces import _SIDE_OF, LANE, Piece, Port, _lay, _rails

from . import fragment

__all__ = ['cannon', 'RULE', 'VARIANTS']

RULE = 'pick the lane before the launcher fires; the lane decides how far you go'

#: Cells of shared mouth before the lanes diverge -- enough that sitting over a
#: particular barrel is a deliberate choice rather than an accident of arrival.
RUN_UP = 2

#: Every barrel brings a marble up to the same speed along its lane, so a
#: lane's reach is set by how hard it throws rather than by how a player
#: entered -- the aim happens at the mouth, not on the approach.
BOOST_SPEED = 6.0

#: How far a barrel's face tilts, purely cosmetic: the cells it sits on keep a
#: single flat height throughout, so a cannon never touches the slope budget.
RISE = 0.9

#: Each variant is a tuple of ``(launch_up, rows)`` per lane, weakest first, so
#: the last lane is always the one that goes furthest and stays open.  ``rows``
#: is sized against the distance a lane throws a marble at up to 12 m/s, so
#: nothing built here throws a marble past its own far wall.
VARIANTS = {
    'twin': {'theme': 'stone',
             'lanes': ((0.20, 6), (0.45, 8))},
    'triple': {'theme': 'foundry',
               'lanes': ((0.20, 6), (0.34, 8), (0.55, 9))},
    'quad': {'theme': 'ice',
             'lanes': ((0.20, 6), (0.30, 7), (0.40, 8), (0.55, 9))},
    'far': {'theme': 'rubber',
            'lanes': ((0.20, 6), (0.45, 8), (0.70, 10))},
}


@fragment('cannon', tags=('aim', 'speed'), rule=RULE, cost=6.0,
          variants=tuple(VARIANTS))
def cannon(rng, entry, variant='twin', theme=None, lanes=None):
    """A mouth, one launch ramp per lane, and a wall behind every lane but the
    strongest.

    ``lanes`` overrides the variant's own ``(launch_up, rows)`` tuples, weakest
    first.
    """
    settings = VARIANTS[variant]
    theme = theme or settings['theme']
    lane_specs = lanes or settings['lanes']
    width = len(lane_specs)

    cells: dict = {}
    features: list = []

    mouth = _lay(cells, entry, RUN_UP, width=width)
    features.extend(_rails(cells, entry, RUN_UP, width=width))
    divergence = mouth.ahead(1)
    across = entry.across()
    half = width // 2

    exits = {}
    best = len(lane_specs) - 1
    for index, (launch_up, rows) in enumerate(lane_specs):
        offset = index - half
        lane_entry = Port(cell=(divergence.cell[0] + across[0] * offset,
                                divergence.cell[1] + across[1] * offset),
                          facing=entry.facing, height=divergence.height, width=1)
        end = _lay(cells, lane_entry, rows, width=1)
        features.append(Ramp(cell=lane_entry.cell, direction=entry.facing,
                             rise=RISE, boost_speed=BOOST_SPEED, launch=True,
                             launch_up=launch_up))
        features.extend(_rails(cells, lane_entry, rows, width=1))
        if index == best:
            pad = _lay(cells, end.ahead(1), 2, width=LANE)
            features.extend(_rails(cells, end.ahead(1), 2, width=LANE))
            exits['ok'] = Port(cell=pad.cell, facing=entry.facing,
                               height=entry.height, width=LANE)
        else:
            features.append(Wall(cell=end.cell, side=_SIDE_OF[entry.facing]))
            name = 'short' if index == 0 else 'lane%d' % index
            exits[name] = Port(cell=end.cell, facing=entry.facing,
                               height=entry.height, width=1)

    return Piece(name='cannon', cells=cells, entry=entry, exits=exits,
                 features=features, theme=theme, rule=RULE)
