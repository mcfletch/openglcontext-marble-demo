"""A shaft you have to go up, on the speed you brought with you.

The counterpart of the drop.  A rising slope, walled both sides, and the only
way past it is to arrive fast enough — a marble that comes in at a crawl climbs
part of the way, stops, and rolls back down to try again.

It is the fragment that makes a run-up worth having.  A piece cannot manufacture
speed for the piece after it, so a chimney's difficulty is decided by whatever a
player did *before* they got here: a conveyor or a fast plinko slot two chapters
back is what gets you up this one.
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails, _slope

__all__ = ['chimney', 'RULE', 'VARIANTS']

RULE = 'arrive fast enough to climb it, or roll back and try again'

#: ``rise`` is how far the shaft climbs; the slope decides how many cells it
#: needs, since a slope may only rise so fast.
VARIANTS = {
    'plain': {'theme': 'stone', 'rise': 2.7, 'run': 4},
    'tall': {'theme': 'stone', 'rise': 4.5, 'run': 6},
    'foundry': {'theme': 'foundry', 'rise': 3.6, 'run': 5},
    'slick': {'theme': 'ice', 'rise': 2.7, 'run': 4},
}


@fragment('chimney', tags=('speed', 'gate'), rule=RULE, cost=6.0,
          variants=tuple(VARIANTS))
def chimney(rng, entry, variant='plain', theme=None, rise=None, run=None):
    """A walled rising slope, with a shelf at the top to arrive on."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    rise = abs(rise or settings['rise'])
    run = run or settings['run']

    cells: dict = {}
    features: list = []
    approach = _lay(cells, entry, 2, width=LANE)
    top = _slope(cells, features, approach.ahead(1), run, rise, width=LANE)
    shelf = _lay(cells, top.ahead(1), 2, height=top.height, width=LANE)
    rows = max(row for _, row in cells) - min(row for _, row in cells) + 1
    features.extend(_rails(cells, entry, rows, width=LANE))
    return Piece(name='chimney', cells=cells, entry=entry,
                 exits={'ok': Port(cell=shelf.cell, facing=entry.facing,
                                   height=top.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)
