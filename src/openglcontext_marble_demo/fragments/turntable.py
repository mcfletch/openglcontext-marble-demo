"""A disc that turns, with several ways off it, and one of them always passing.

A short straight lane brings the marble to a small hub -- walled all round
except at the mouths a story asks for -- with a
:class:`~openglcontext_marble_demo.mechanisms.turntable.Turntable` spinning at
its centre. The board's own downhill lean is what brings a marble to the hub in
the first place -- nothing here has to fetch it -- and the bar meets whatever
is standing there and carries it whichever way the bar happens to be pointing.

Which way that is depends on nothing but when the marble arrived: the bar
sweeps past ``ok`` (straight on) and every other mouth in turn, so a player who
watches the turn and times their approach gets the way they wanted, and one who
does not gets whichever mouth the bar is passing when it catches them. Neither
is stuck: the bar never stops sweeping, so a marble it does not carry straight
out is carried on round to the next mouth instead, and the piece is never a
wall.

The bar reaches only as far as the mouths themselves -- ``hub`` cells, the same
number either side -- and no further, so it never reaches back into the lane a
marble arrives down. A bar long enough to do that meets an arriving marble
edge-on rather than broadside, which is the one contact this mechanism cannot
afford: :mod:`tests/mechanisms/test_turntable.py` holds the lane clear of it.

``tests/fragments/test_turntable.py`` measures the way the gauntlet's own test
does: starting the bar part way round its turn, which is the same question as a
marble reaching the hub part way through it.
"""
from typing import Any

from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment
from openglcontext_marble_demo.mechanisms.turntable import Turntable
from openglcontext_marble_demo.pieces import LANE, Piece, Port

__all__ = ['turntable']

RULE = 'read the turn and time your moment, or take whichever way is passing'

#: Cells of plain lane before the hub, so a marble is fully clear of the bar's
#: reach when it starts and arrives at the hub already rolling straight.
LEAD = 3

#: Cells of run-out beyond an open mouth, so a marble the bar has just let go of
#: is fully clear of the hub -- and so a test can tell which mouth it left by --
#: before the corridor ends.
RUN_OUT = 2

#: Variants: the material, the turn rate in rpm (negative turns the other
#: way), and which of the three possible mouths are open. ``ok`` is always one
#: of them. ``hub`` -- how many cells the bar reaches either side of centre --
#: is a real parameter a story or the editor can still set, but every variant
#: here keeps it at the radius the mechanism was measured safe at: doubled, a
#: mistimed marble was found to be carried round and round the rim by the bar's
#: own sweep, never released into any mouth at all -- spinning forever rather
#: than being delayed, which the piece must never do.
#: What each variant sets, by name. Values of every kind, which is what a
#: variant is: the knob a fragment reads by that name.
_VARIANTS: dict[str, dict[str, Any]] = {
    'plain': {'theme': 'stone', 'hub': 1, 'rpm': 20.0,
              'exits': ('ok', 'left', 'right')},
    'brisk': {'theme': 'ice', 'hub': 1, 'rpm': 30.0,
              'exits': ('ok', 'left', 'right')},
    'reverse': {'theme': 'foundry', 'hub': 1, 'rpm': -20.0,
                'exits': ('ok', 'left', 'right')},
    'twin': {'theme': 'rubber', 'hub': 1, 'rpm': 24.0, 'exits': ('ok', 'left')},
}


def _side_port(hub_cell: Any, across: Any, side: Any, radius: Any, height: Any) -> Any:
    """The mouth ``radius`` cells off the hub in ``across`` turned by ``side``.

    ``side`` is ``-1`` or ``1``; the facing it comes out with points away from
    the hub, which is the direction a marble carries on in once it is through.
    """
    cell = (hub_cell[0] + across[0] * side * radius,
            hub_cell[1] + across[1] * side * radius)
    facing = (across[0] * side, across[1] * side)
    return Port(cell=cell, facing=facing, height=height, width=LANE)


@fragment('turntable', tags=('aim', 'luck'), cost=4.0, rule=RULE,
          variants=tuple(_VARIANTS))
def turntable(rng: Any, entry: Any, variant: Any='plain', theme: Any=None, hub: Any=None, rpm: Any=None,
              exits: Any=None) -> Any:
    """A lead-in lane, a hub with a spinning bar in it, and a mouth per way off."""
    settings = _VARIANTS[variant]
    theme = theme or settings['theme']
    radius = hub or settings['hub']
    rpm = settings['rpm'] if rpm is None else rpm
    exits = tuple(exits) if exits is not None else settings['exits']

    cells: dict = {}
    lead_end = pieces._lay(cells, entry, LEAD)
    hub_width = radius * 2 + 1
    hub_start = lead_end.ahead(1)
    # A cross, not a filled square: a corner the horizontal arm does not reach
    # and the vertical one does not pass through is a pocket the board's lean
    # pushes a marble into and nothing ever pulls it back out of -- found by
    # simulating the hub at radius 2, where the far corners are exactly that.
    far_row = pieces._lay(cells, hub_start, hub_width, width=LANE)
    hub_cell = hub_start.ahead(radius).cell
    across_row = Port(cell=hub_cell, facing=entry.facing, height=entry.height,
                      width=hub_width)
    pieces._lay(cells, across_row, 1, width=hub_width)
    across = entry.across()

    open_at = set(entry.cells())
    exit_ports: dict = {}

    if 'ok' in exits:
        mouth = Port(cell=far_row.cell, facing=entry.facing, height=entry.height,
                     width=LANE)
        open_at |= set(mouth.cells())
        run_out = pieces._lay(cells, mouth.ahead(1), RUN_OUT, width=LANE)
        exit_port = Port(cell=run_out.cell, facing=entry.facing,
                         height=entry.height, width=LANE)
        open_at |= set(exit_port.cells())
        exit_ports['ok'] = exit_port

    for name, side in (('left', -1), ('right', 1)):
        if name not in exits:
            continue
        mouth = _side_port(hub_cell, across, side, radius, entry.height)
        open_at |= set(mouth.cells())
        run_out = pieces._lay(cells, mouth.ahead(1), RUN_OUT, width=LANE)
        exit_port = Port(cell=run_out.cell, facing=mouth.facing,
                         height=entry.height, width=LANE)
        open_at |= set(exit_port.cells())
        exit_ports[name] = exit_port

    walls = pieces._ring(cells, set(cells), gaps=open_at)
    disc = Turntable(cell=hub_cell, radius=radius, rpm=rpm)
    return Piece(name='turntable', cells=cells, entry=entry, exits=exit_ports,
                 features=[disc] + walls, theme=theme, rule=RULE)
