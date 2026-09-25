"""A plank across the way that dips under whatever lingers on it.

A mouth, a :class:`~openglcontext_marble_demo.mechanisms.seesaw.Seesaw` hinged
where it meets the landing beyond it, and that landing. The plank is not
scripted: it dips toward whatever is standing on it, up to a limit, and settles
level again once nothing is. Crossing it fast means the plank barely
notices — it is chasing a target that has already moved on by the time it
catches up. Crossing it slow gives the plank time to sink fully under the
marble's own weight, which turns the whole span into a climb the marble has to
carry itself out of, one that eases the closer it gets to the far end — the
hinge never moves, so the way on is exactly as flush after the plank has
sagged to its limit as it is when nothing is on it at all.

Getting it wrong costs seconds, never the run: the board's own downhill lean is
kept shallower than the plank's own maximum tilt (see the mechanism's module
docstring), so a marble that stalls on the climb keeps creeping across rather
than sitting there forever.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = seesaw(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from typing import Any

from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.seesaw import Seesaw
from openglcontext_marble_demo.pieces import LANE, Piece, Port, lay, rails

__all__ = ['seesaw', 'RULE', 'VARIANTS']

RULE = ('cross with speed before the plank finds you; dawdle and your own '
        'weight digs the climb you have to carry yourself over')

#: Material, layout and effect together. ``length`` and ``width`` are the
#: plank's span in cells; ``max_tilt`` (degrees) and ``response`` (seconds) are
#: how far it leans and how readily — the two numbers that decide how hard a
#: slow crossing bites.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
VARIANTS: dict[str, dict[str, Any]] = {
    'plain':  {'theme': 'stone', 'length': 5, 'width': 3,
               'max_tilt': 7.0, 'response': 0.5},
    'long':   {'theme': 'stone', 'length': 7, 'width': 3,
               'max_tilt': 6.0, 'response': 0.8},
    'narrow': {'theme': 'ice', 'length': 5, 'width': 1,
               'max_tilt': 7.0, 'response': 0.3},
    'heavy':  {'theme': 'foundry', 'length': 5, 'width': 5,
               'max_tilt': 7.5, 'response': 0.9},
}


@fragment('seesaw', tags=('speed', 'gate'), rule=RULE, cost=6.0,
          variants=tuple(VARIANTS))
def seesaw(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, length: Any=None, width: Any=None,
           max_tilt: Any=None, response: Any=None) -> Any:
    """A mouth, a plank hinged at its far row, and a landing beyond it."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    length = length or settings['length']
    width = width or settings['width']
    max_tilt = settings['max_tilt'] if max_tilt is None else max_tilt
    response = settings['response'] if response is None else response

    cells: dict = {}

    # A short mouth at the plank's own width, so lining up for it is a choice.
    mouth = lay(cells, entry, 2, width=width)
    plank_start = mouth.ahead(1)
    plank_end = lay(cells, plank_start, length, width=width)
    landing = lay(cells, plank_end.ahead(1), 2, width=width)

    plank = Seesaw(cell=plank_start.cell, facing=entry.facing, length=length,
                   width=width, max_tilt=max_tilt, response=response)
    total = 2 + length + 2
    features = [plank] + rails(cells, entry, total, width=width)

    return Piece(name='seesaw', cells=cells, entry=entry,
                 exits={'ok': Port(cell=landing.cell, facing=entry.facing,
                                   height=entry.height, width=LANE)},
                 features=features, theme=theme, rule=RULE)
