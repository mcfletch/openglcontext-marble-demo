"""A fall onto a landing, and a rule about the speed you leave the lip at.

The floor stops.  Some way on and some way down there is a pad, and between them
is nothing.  A marble that trickles off the lip comes down in the gap; one that
leaves fast enough puts itself on the pad; one that leaves *too* fast crosses the
pad's height beyond its far edge and comes down past it.  Every part of that is
decided before the marble leaves the ground, which is what makes the piece a
question about the run-up rather than about the jump.

Where the window falls is set by the gap and the landing, both of them variant
settings, so a story can ask for the hop or for the one that wants a descent in
front of it.  Entering the piece, in metres a second, and measured by
``tests/fragments/test_drop.py``:

=========  ===========  ==============  =================
variant    lands short  on the landing  past the far edge
=========  ===========  ==============  =================
``plain``  up to 4.0    4.5 to 11.0     from 11.5
``wide``   up to 4.0    4.5 to 16.0     from 16.5
``long``   up to 9.5    10.0 to 18.5    from 19.0
``slick``  up to 4.5    5.0 to 12.0     from 12.5
=========  ===========  ==============  =================

The gap sets the bottom of the window and the landing sets the top: ``long``
puts two cells of nothing in front of the lip and so asks for twice the speed
``plain`` does, and ``wide`` lands on two cells rather than one and so forgives
half as much again at the top.

**The landing is walled along its side and open at both ends** -- open at the
near end because that is where a marble arrives from, and at the far end because
that is the way on.  The run-up is walled on one side only: the other side is a
ledge that steps down around the gap, which is how a marble that will not commit
still gets to the bottom.  It is one cell wide with the pit beside it, so it is
slow and it is not free.
"""
from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import Ramp
from openglcontext_marble_demo.pieces import MAX_STEP, Piece, Port

__all__ = ['drop']

#: Variants: what it is made of, how far the fall is, how wide the gap is in
#: cells, and how many cells deep the landing is.  ``drop`` is bounded by the
#: ledge, which has ``gap + 2`` cells to get down in and may not step further
#: than :data:`~openglcontext_marble_demo.pieces.MAX_STEP` in any of them.
_VARIANTS = {
    'plain': {'theme': 'stone', 'fall': 2.7, 'gap': 1, 'landing': 1},
    'wide': {'theme': 'foundry', 'fall': 2.7, 'gap': 1, 'landing': 2},
    'long': {'theme': 'stone', 'fall': 3.6, 'gap': 2, 'landing': 2},
    'slick': {'theme': 'ice', 'fall': 1.8, 'gap': 1, 'landing': 1},
}

#: Cells of run-up before the lip.  Short on purpose: a long approach turns
#: whatever speed a marble arrived with into whatever speed the board gives it,
#: and then the rule is about the board rather than about the player.
RUN_UP = 2


@fragment('drop', tags=('speed', 'hazard'), cost=7.0,
          rule='leave the lip fast enough to reach the landing, and no faster',
          variants=tuple(_VARIANTS))
def drop(rng, entry, variant='plain', theme=None, fall=None, gap=None,
         landing=None, side=None):
    """A lip, a gap, and a pad to come down on.

    ``side`` is which side of the run-up the ledge steps down, ``1`` or ``-1``
    across the facing; left to the rng when it is not asked for.
    """
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    fall = abs(fall if fall is not None else settings['fall'])
    gap = gap or settings['gap']
    landing = landing or settings['landing']
    if side is None:
        side = rng.choice((1, -1))
    steps = gap + 2
    if fall > MAX_STEP * steps + 1e-9:
        raise ValueError(
            'a drop of %.2f over a gap of %d cells wants a ledge stepping %.2f, '
            'which is more than the %.2f a marble can roll'
            % (fall, gap, fall / steps, MAX_STEP))

    across = entry.across()
    sideways = (across[0] * side, across[1] * side)
    # Which of ``_rails``' two sides is the one the ledge is on.
    toward, away = ('right', 'left') if side > 0 else ('left', 'right')

    cells: dict = {}
    lip = pieces._lay(cells, entry, RUN_UP)
    floor = entry.height - fall
    pad_entry = Port(cell=lip.ahead(gap + 1).cell, facing=entry.facing,
                     height=floor, width=entry.width)
    pad = pieces._lay(cells, pad_entry, landing)

    # The ledge: one cell across, outside the lane on ``side``, stepping down
    # beside the gap from the lip's row to the landing's first row.
    reach = entry.width // 2 + 1
    ledge = Port(cell=(lip.cell[0] + sideways[0] * reach,
                       lip.cell[1] + sideways[1] * reach),
                 facing=entry.facing, height=entry.height, width=1)
    rise = -fall / steps
    ramps = []
    for step in range(steps):
        at = ledge.ahead(step)
        cells[at.cell] = round(entry.height + rise * (step + 1), 6)
        if step < steps - 1:
            # Shape and nothing else: what a marble carries down the ledge is
            # what it arrived with plus what the step gives it.
            ramps.append(Ramp(cell=at.cell, direction=entry.facing, rise=rise,
                              boost_speed=None))

    walls = pieces._rails(cells, entry, RUN_UP, sides=(away,))
    # The run-up's ledge side is closed until the lip, so the only way onto the
    # ledge is at the point where the way straight on has run out.
    walls += pieces._rails(cells, entry, RUN_UP - 1, sides=(toward,))
    walls += pieces._rails(cells, ledge, steps, sides=(toward,), width=1)
    walls += pieces._rails(cells, pad_entry, landing, sides=(away,))
    # The landing's ledge side is open only where the ledge arrives.
    if landing > 1:
        walls += pieces._rails(cells, pad_entry.ahead(1), landing - 1,
                               sides=(toward,))
    return Piece(name='drop', cells=cells, entry=entry, exits={'ok': pad},
                 features=ramps + [wall for wall in walls if wall.cell in cells],
                 theme=theme,
                 rule='leave the lip fast enough to reach the landing, and no faster')
