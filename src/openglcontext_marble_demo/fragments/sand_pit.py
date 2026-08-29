"""A pit of sand with hard edges, and two ways past it.

Sand slows a marble to a churn — eight times as long to cross as plain floor —
and it stops dead at its edges, so clipping the corner of it is a decision
rather than a gradient. This chapter is built around that: a lane through a
chamber whose middle is sand, with **hard floor left along one side**.

So there are two answers, and they cost differently. Pick the careful line
along the edge and pay nothing but the care. Or take the ramp on the approach
and be thrown clear over the sand, which is quicker still and is the thing that
goes wrong if the launch is muffed — and a muffed launch is not a loss, because
the sand always lets a marble out of its far end eventually. Getting it wrong
costs seconds, which is the currency the whole game is denominated in.

The edge is deliberately narrow. A hard shoulder wide enough to be comfortable
is a corridor with some decoration in the middle of it.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = sand_pit(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import Ramp
from openglcontext_marble_demo.mechanisms.sand import Sand
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails

__all__ = ['sand_pit', 'RULE', 'VARIANTS']

RULE = 'thread the hard edge, or take the ramp and clear it'

#: Material, layout and effect together.  ``length`` is how many cells of sand,
#: ``width`` how wide the chamber, ``shoulder`` how many cells of hard floor are
#: left along one side, and ``launch`` whether a ramp is offered on the approach.
VARIANTS = {
    'plain': {'theme': 'stone', 'length': 3, 'width': 5, 'shoulder': 1,
              'launch': True},
    'wide': {'theme': 'stone', 'length': 4, 'width': 7, 'shoulder': 2,
             'launch': True},
    'sheer': {'theme': 'foundry', 'length': 4, 'width': 5, 'shoulder': 1,
              'launch': False},
    'quarry': {'theme': 'foundry', 'length': 5, 'width': 7, 'shoulder': 1,
               'launch': True},
}


@fragment('sand_pit', tags=('aim', 'hazard'), rule=RULE, cost=7.0,
          variants=tuple(VARIANTS))
def sand_pit(rng, entry, variant='plain', theme=None, length=None, width=None,
             shoulder=None, launch=None):
    """A chamber floored with sand, a hard shoulder down one side, a ramp in."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    length = length or settings['length']
    width = width or settings['width']
    shoulder = settings['shoulder'] if shoulder is None else shoulder
    launch = settings['launch'] if launch is None else launch

    cells: dict = {}
    features: list = []

    across = entry.across()
    hand = rng.choice((-1, 1))
    half = width // 2

    # A run-up, so a ramp has something to be entered from.
    approach = _lay(cells, entry, 2, width=width)
    chamber = Port(cell=approach.ahead(1).cell, facing=entry.facing,
                   height=entry.height, width=width)
    _lay(cells, chamber, length, width=width)

    # The hard shoulder runs down one side of the chamber -- which side is the
    # rng's, so two pits on one board are not the same pit -- and the sand fills
    # everything else.
    shoulder_columns = {half - offset for offset in range(shoulder)}

    def _at(port, step, offset):
        cell = port.ahead(step).cell
        return (cell[0] + across[0] * hand * offset,
                cell[1] + across[1] * hand * offset)

    keep = {_at(chamber, step, offset)
            for step in range(length) for offset in shoulder_columns}
    sand = sorted(set(_span(cells, chamber, length)) - keep)
    if sand:
        features.append(Sand(cells=[list(cell) for cell in sand]))

    if launch:
        # The ramp covers the lanes the sand is under, and **not** the shoulder.
        # That is what makes this a choice rather than a corridor: line up over
        # the sand and be thrown clear of it, or thread the shoulder and keep
        # your feet.  A ramp across the whole mouth would launch a marble
        # whatever it did, and the pit would be scenery under it.
        for offset in range(-half, half + 1):
            if offset in shoulder_columns:
                continue
            for step in range(2):
                cell = _at(approach.ahead(step - 1), 0, offset)
                if cell in cells:
                    features.append(Ramp(cell=cell, direction=entry.facing,
                                         rise=0.9, boost_speed=11.0,
                                         launch=True, launch_up=5.5))

    landing = _lay(cells, chamber.ahead(length), 2, width=width)
    # Walled all round the chamber: being in the sand is slow, and being in the
    # sand *and* off the board is a different piece.
    features.extend(_rails(cells, entry, length + 4, width=width))
    return Piece(name='sand_pit', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.cell, facing=entry.facing,
                                   height=entry.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)


def _span(cells, port, length):
    """Every cell of ``length`` rows of the lane starting at ``port``."""
    return [cell for step in range(length) for cell in port.ahead(step).cells()
            if cell in cells]
