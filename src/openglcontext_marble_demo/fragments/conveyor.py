"""A belt that brings a slow marble up to speed, and lets a fast one alone.

A piece cannot manufacture speed for the piece after it -- a marble driven down
a descent settles at about five metres a second however far it has fallen,
because steering across the board's lean spends the pull.  So a story that wants
a chimney or a long jump in it needs somewhere for the speed to come from, and
this is that somewhere.

The belt brings the marble *up to* its own speed and never brakes it, so
arriving fast costs nothing and arriving slow costs only the length of the belt.
What a player decides is whether to spend the ground: the belt is several cells
of board that go nowhere but forward, and the chapter it feeds is the reason to
have crossed it.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = conveyor(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import Ramp
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _rails

__all__ = ['conveyor', 'RULE', 'VARIANTS']

RULE = 'the belt sets the speed you leave at, however you arrived'

#: ``speed`` is what the belt brings a marble up to, in metres a second, and
#: ``length`` how many cells of it there are.  A longer belt is not faster; it is
#: more ground spent on being sure.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'foundry', 'speed': 12.0, 'length': 4},
    'fast': {'theme': 'foundry', 'speed': 18.0, 'length': 4},
    'long': {'theme': 'stone', 'speed': 12.0, 'length': 7},
    'slick': {'theme': 'ice', 'speed': 14.0, 'length': 5},
}


@fragment('conveyor', tags=('speed', 'run-up'), rule=RULE, cost=4.0,
          variants=tuple(VARIANTS))
def conveyor(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, speed: Any=None, length: Any=None) -> Any:
    """A run of boosting tiles along the lane, walled both sides."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    speed = speed or settings['speed']
    length = length or settings['length']

    cells: dict = {}
    features: list = []
    mouth = _lay(cells, entry, 1, width=LANE)
    belt = mouth.ahead(1)
    _lay(cells, belt, length, width=LANE)
    for step in range(length):
        for cell in belt.ahead(step).cells():
            # ``rise=0``: the tile is flat, and all the belt does is the boost.
            # A ramp is how a cell puts a force on the marble crossing it, and a
            # sloped one would be a hill as well as a belt.
            features.append(Ramp(cell=cell, direction=entry.facing, rise=0.0,
                                 boost_speed=speed))
    end = _lay(cells, belt.ahead(length), 2, width=LANE)
    features.extend(_rails(cells, entry, length + 3, width=LANE))
    return Piece(name='conveyor', cells=cells, entry=entry,
                 exits={'ok': Port(cell=end.cell, facing=entry.facing,
                                   height=entry.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)
