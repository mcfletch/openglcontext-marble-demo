"""A platform that rises and falls, and more than one place to leave it.

``level.Elevator`` already rides a sine between a resting height and a peak;
on its own, riding one is not a decision -- there is nowhere else to go, so a
marble either waits for it or does not. This fragment lays two or three lanes
side by side after a shared approach: one is flat and always open, the others
each carry a platform that must be ridden up to a landing of its own, higher
the longer a marble stays aboard. Picking a lane is picking how much of the
wait to pay, and the wait is not the same wait twice -- whatever crossed the
board before this fragment took its own time doing it, so the platform is
never at the same point in its cycle for two runs. The piece is timed the same
way ``gauntlet`` is: entering at several phases of the cycle and comparing
what each one costs.

Nothing here can lose a marble. A lane's platform starts flush with the
approach, so a marble that reaches it while the platform is elsewhere simply
cannot climb the gap yet -- exactly the wait at a lift door that has not
arrived -- and it sits there until the platform comes home rather than
falling anywhere. Once aboard, every landing is either level with the
platform or a step *down* from it, which a marble takes without being hurt.
"""
from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.level import Elevator, Wall
from openglcontext_marble_demo.pieces import Piece, Port

__all__ = ['lift']

#: Cells of clear, wide approach before the lanes split off -- room enough
#: that a marble can drift into whichever one it wants without being forced
#: into a last-instant diagonal move the pilot reads as a corner to cut.
LEAD = 3
#: Cells of run-out past a landing, before the piece hands off to whatever
#: comes next.
RUN_OUT = 2
#: How far above a lane's own platform a wall built to contain it stands,
#: clear of the marble riding on top too.
WALL_MARGIN = 2.0

#: Variants: the material, how far and how often each lane's platform
#: travels, and the height gained by riding each one, in the order the lanes
#: are laid out away from the always-open one.  A single gain is one riding
#: lane named ``'ok'``; two gains adds a nearer one named ``'mid'`` between
#: the bail-out and ``'ok'``.  Every gain stays under
#: :data:`~openglcontext_marble_demo.pieces.MAX_STEP` above the lane's own
#: resting height, which is what keeps the fragment inside the slope budget
#: without a ramp ever being drawn -- the platform is the ramp.
_VARIANTS = {
    'plain':   {'theme': 'stone',   'travel': 1.0, 'period': 3.0, 'gains': (0.6,)},
    'brisk':   {'theme': 'ice',     'travel': 1.0, 'period': 1.8, 'gains': (0.6,)},
    'foundry': {'theme': 'foundry', 'travel': 1.3, 'period': 4.0, 'gains': (0.85,)},
    'tower':   {'theme': 'rubber',  'travel': 1.3, 'period': 3.0, 'gains': (0.35, 0.75)},
}


@fragment('lift', tags=('aim', 'luck'), cost=4.0,
          rule="step off while a platform is level with somewhere -- "
               "wait longer and you go higher",
          variants=tuple(_VARIANTS))
def lift(rng, entry, variant='plain', theme=None, travel=None, period=None,
         gains=None):
    """A wide approach into parallel lanes: one flat, the rest riding lifts.

    ``gains`` is read in the order the lanes sit away from the always-open
    one, each naming the height a marble reaches by staying aboard that
    lane's own platform.  The lanes are walled apart for their whole length,
    so a marble commits to one during the wide approach and cannot drift
    into another once the platforms start moving underneath it.
    """
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    travel = travel if travel is not None else settings['travel']
    period = period if period is not None else settings['period']
    gains = tuple(gains) if gains is not None else settings['gains']

    base = entry.height
    across = entry.across()
    left = (-across[0], -across[1])
    tall = travel + WALL_MARGIN

    names = ('mid', 'ok') if len(gains) == 2 else ('ok',)
    lanes = [None] + list(zip(names, gains, strict=True))  # None: the bail-out lane
    half = len(lanes) // 2

    cells: dict = {}
    walls: list = []
    elevators: list = []
    exits: dict = {}
    open_at = set(entry.cells())

    lead_end = pieces._lay(cells, entry, LEAD, width=entry.width)
    row0 = lead_end.ahead(1)

    def lane_cell(offset, ahead):
        return (row0.cell[0] + entry.facing[0] * ahead + across[0] * offset,
                row0.cell[1] + entry.facing[1] * ahead + across[1] * offset)

    for index, lane in enumerate(lanes):
        offset = index - half
        first, second = lane_cell(offset, 0), lane_cell(offset, 1)
        cells[first] = base
        if lane is None:
            cells[second] = base
            name, run_height = 'early', base
        else:
            name, gain = lane
            elevators.append(Elevator(cell=first, travel=travel, period=period))
            cells[second] = base + gain
            run_height = base + gain
        run_port = Port(cell=second, facing=entry.facing, height=run_height,
                        width=1)
        end = pieces._lay(cells, run_port.ahead(1), RUN_OUT, width=1)
        exits[name] = end
        open_at.add(end.cell)

        # Walled apart for the two rows a platform actually moves through,
        # and the run-out beyond -- everywhere short of the mouth the piece
        # hands off through.  Height clears the tallest ride in the piece,
        # which costs nothing on a lane that never rises.
        for ahead in range(0, RUN_OUT + 1):
            cell = lane_cell(offset, ahead)
            for side in (across, left):
                walls.append(Wall(cell=cell, side=pieces._SIDE_OF[side],
                                  height=tall))

    boundary = pieces._ring(cells, set(cells), gaps=open_at)
    return Piece(name='lift', cells=cells, entry=entry, exits=exits,
                features=elevators + walls + boundary, theme=theme,
                rule="step off while a platform is level with somewhere -- "
                     "wait longer and you go higher")
