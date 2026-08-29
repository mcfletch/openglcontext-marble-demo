"""A door, and the lever that opens it: a room you have to earn your way out of.

A lever throws at a blow of four metres a second or harder, and the blow also
stops the marble — so a run at one costs a pass, and that is the shape of the
chapter.  The room gives you somewhere to build the speed, the lever is on the
far wall, and the door is beside it.  Hit it hard enough and it goes; hit it
softly and nothing happens and you are where you started, a lap behind.

Getting it wrong is a lap of the room, not a loss.  That matters: a door that
could be permanently failed would be a board a player is stuck on, and the point
of the lever is that it is worth a run-up rather than that it is a wall.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> piece = locked_room(random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True
"""
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.lever import Door, Lever
from openglcontext_marble_demo.pieces import LANE, Piece, Port, _lay, _ring

__all__ = ['locked_room', 'RULE', 'VARIANTS']

RULE = 'hit the lever hard enough, which takes a run at it'

#: ``across`` and ``along`` are the room; ``hardness`` is what the lever wants,
#: in metres a second of closing speed.  A bigger room is an easier lever,
#: because a bigger room is more floor to build speed on.
VARIANTS = {
    'plain': {'theme': 'foundry', 'across': 7, 'along': 6, 'hardness': 4.0},
    'tight': {'theme': 'foundry', 'across': 5, 'along': 5, 'hardness': 4.0},
    'stiff': {'theme': 'stone', 'across': 7, 'along': 7, 'hardness': 6.0},
    'slick': {'theme': 'ice', 'across': 7, 'along': 6, 'hardness': 4.0},
}


@fragment('locked_room', tags=('speed', 'gate'), rule=RULE, cost=6.0,
          variants=tuple(VARIANTS))
def locked_room(rng, entry, variant='plain', theme=None, across=None,
                along=None, hardness=None):
    """A walled room with a lever at the far end and the door it opens."""
    settings = dict(VARIANTS[variant])
    theme = theme or settings['theme']
    across = across or settings['across']
    along = along or settings['along']
    hardness = hardness or settings['hardness']

    cells: dict = {}
    room = _lay(cells, entry, along, width=across)
    channel = 'lock-%d' % rng.randrange(1 << 20)

    # The lever stands on the last cell of the room, facing the way in, so a
    # marble that has run the length of the floor meets it square.
    lever = Lever(cell=room.cell, channel=channel, hardness=hardness,
                  facing=entry.facing)
    # The doorway is the way out, and the door fills it until the lever goes.
    doorway = room.ahead(1)
    _lay(cells, doorway, 2, width=LANE)
    door = Door(cell=doorway.cell, channel=channel,
                side={(0, 1): 'N', (0, -1): 'S', (1, 0): 'W',
                      (-1, 0): 'E'}[entry.facing])

    ways_out = set(entry.cells()) | set(doorway.cells())
    features = _ring(cells, set(cells), gaps=ways_out) + [lever, door]
    return Piece(name='locked_room', cells=cells, entry=entry,
                 exits={'ok': Port(cell=doorway.ahead(1).cell,
                                   facing=entry.facing, height=entry.height,
                                   width=LANE)},
                 features=features, theme=theme, rule=RULE)
