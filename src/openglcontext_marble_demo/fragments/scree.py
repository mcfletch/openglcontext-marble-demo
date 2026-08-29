"""A shattered slope: commit to it, and deal with where it puts you.

The rockfall randomises the way a marble leaves — exit headings spread seven to
twelve degrees where the same slope with the rock taken off spreads by nothing
at all.  That is only worth having if there is somewhere to *be* put, so this
chapter gives it a wide landing at the bottom: the piece is not about getting
through a gap, it is about arriving somewhere you did not choose and carrying on
from there.

Ninety per cent of descents get through, and a marble usually leaves faster than
it arrived, because a slope is still a slope.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = scree(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.rockfall import Rockfall
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails, _slope

__all__ = ['scree', 'RULE', 'VARIANTS']

RULE = 'commit to the broken slope and deal with where it leaves you'

#: ``drop`` is how far the broken slope falls over its length; the rock is laid
#: on top of a run the fragment lays, because the cells are the track and the
#: track belongs to the level.
VARIANTS = {
    'plain': {'theme': 'stone', 'length': 6, 'width': 5, 'density': 1.5,
              'drop': 3.6},
    'long': {'theme': 'stone', 'length': 9, 'width': 5, 'density': 1.5,
             'drop': 5.4},
    'thick': {'theme': 'foundry', 'length': 6, 'width': 7, 'density': 2.2,
              'drop': 3.6},
    'sparse': {'theme': 'stone', 'length': 6, 'width': 7, 'density': 0.8,
               'drop': 2.7},
}


@fragment('scree', tags=('luck', 'hazard'), rule=RULE, cost=5.0,
          variants=tuple(VARIANTS))
def scree(rng, entry, variant='plain', theme=None, length=None, width=None,
          density=None, drop=None):
    """A rockfall with a mouth above it and a wide landing below."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    length = length or settings['length']
    width = width or settings['width']
    density = settings['density'] if density is None else density
    drop = -abs(drop if drop is not None else settings['drop'])

    cells: dict = {}
    features: list = []
    mouth = _lay(cells, entry, 2, width=width)

    # The run the rock stands on.  The rockfall owns these cells -- it lays its
    # own tilted tiles over them -- but they are the track and the track belongs
    # to the level, so they are laid here.
    slope_features: list = []
    foot = _slope(cells, slope_features, Port(cell=mouth.ahead(1).cell,
                                              facing=entry.facing,
                                              height=entry.height, width=width),
                  length, drop, width=width)
    fall = Rockfall(cell=mouth.ahead(1).cell, direction=entry.facing,
                    length=length, width=width, density=density,
                    seed=rng.randrange(1 << 30))
    features.append(fall)

    bottom = foot.height
    rows = max(row for _, row in cells)
    # A wide landing, because the whole point is arriving somewhere you did not
    # aim at: a narrow one would turn a randomiser into a coin toss.
    landing = Port(cell=(entry.cell[0], rows + 1), facing=entry.facing,
                   height=bottom, width=width + 2)
    _lay(cells, landing, 3, height=bottom, width=width + 2)
    features.extend(_rails(cells, entry, length + 8, width=width + 2))
    return Piece(name='scree', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.ahead(2).cell,
                                   facing=entry.facing, height=bottom,
                                   width=LANE)},
                 features=features, theme=theme, rule=RULE)
