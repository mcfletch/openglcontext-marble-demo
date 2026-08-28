"""A zigzag descent, with a right-angle at the foot of every fall.

Down a slope, hard across, down another slope, hard across again.  Each corner
is a right-angle with nothing beyond it, so the way straight on is off the board,
and each of them sits where a marble is fastest -- at the bottom of a descent.
Driven by the autopilot, one arrives at 14 metres a second and comes round; one
arriving at 16 goes straight over the edge, and pays a respawn for it: 11.5
seconds down the one-descent piece against 9.2.

Speed on the way down is the reason the corners are placed where they are, and
it is worth knowing what a driven marble actually carries into them.  Left to
free-roll it settles at about 5 m/s and stays there however far it has fallen --
5.08 on the first descent and 5.15 on the second -- because a driver steering
across the lean spends the pull that would otherwise be speed.  So the corners
bite on what a marble was *given* by whatever came before, rather than on what
the piece itself builds up; ``tests/fragments/test_switchback.py`` records that
as a strict xfail against the day it changes.

``legs`` is how many descent-and-corner pairs there are, and is a variant setting
as well as an argument, so a story can put the long version down as a climax and
the short one as an introduction to it -- and so a test can hold the corner
constant and change only the run at it.  The autopilot takes 10.0 seconds over
one descent and 17.1 over two.
"""
from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.pieces import Piece, Port

__all__ = ['switchback']

#: Cells of flat lane after the last corner, so the piece hands the next one a
#: marble that is travelling the way it was entered.
RUN_OUT = 2

#: Cells of flat lane between the foot of a descent and the corner.  A crossing
#: is three cells wide, so it reaches back one cell past the corner; landing that
#: reach on level ground rather than on the descent is what keeps the piece from
#: putting a step in front of a marble that no marble could roll.
APRON = 2

#: Variants: the material, how far each descent falls, how long the descents and
#: the crossings are in cells, and how many of each there are.
_VARIANTS = {
    'plain': {'theme': 'stone', 'fall': 2.7, 'run': 3, 'across': 4, 'legs': 2},
    'steep': {'theme': 'foundry', 'fall': 3.6, 'run': 3, 'across': 4, 'legs': 2},
    'slick': {'theme': 'ice', 'fall': 2.7, 'run': 3, 'across': 5, 'legs': 2},
    'short': {'theme': 'stone', 'fall': 2.7, 'run': 3, 'across': 4, 'legs': 1},
}


@fragment('switchback', tags=('brake', 'speed'), cost=6.0,
          rule='take the corners slowly enough to turn: they are at the foot of the descents',
          variants=tuple(_VARIANTS))
def switchback(rng, entry, variant='plain', theme=None, fall=None, run=None,
               across=None, legs=None, side=None):
    """Descents joined by right-angles, alternating which way they turn.

    ``side`` is which way the first corner turns, ``1`` or ``-1`` across the
    facing; left to the rng when it is not asked for.
    """
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    fall = abs(fall if fall is not None else settings['fall'])
    run = run or settings['run']
    across = across or settings['across']
    legs = legs or settings['legs']
    if side is None:
        side = rng.choice((1, -1))

    sidestep = entry.across()
    cells: dict = {}
    ramps: list = []
    corners: set = set()
    where, turning = entry, side
    for _ in range(legs):
        bottom = pieces._slope(cells, ramps, where, run, -fall)
        bottom = pieces._lay(cells, bottom.ahead(1), APRON, height=bottom.height)
        corners |= set(bottom.cells())
        turn = Port(cell=bottom.cell,
                    facing=(sidestep[0] * turning, sidestep[1] * turning),
                    height=bottom.height, width=entry.width)
        crossing = pieces._lay(cells, turn.ahead(1), across)
        where = Port(cell=crossing.cell, facing=entry.facing,
                     height=crossing.height, width=entry.width).ahead(1)
        turning = -turning
    exit_port = pieces._lay(cells, where, RUN_OUT)

    # Fenced wherever the floor stops, except at the two mouths and straight
    # ahead of each corner.  That opening is the piece: a wall there would take
    # nine tenths of whatever a marble brought to it, which is the descents'
    # work undone at every turn and a run that asks the same question twice.
    open_at = set(entry.cells()) | set(exit_port.cells()) | corners
    return Piece(name='switchback', cells=cells, entry=entry,
                 exits={'ok': exit_port},
                 features=ramps + pieces._ring(cells, set(cells), gaps=open_at),
                 theme=theme,
                 rule='take the corners slowly enough to turn: they are at the foot of the descents')
