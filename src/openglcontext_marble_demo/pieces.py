"""Places, and the ways between them that are the challenge.

A board made of one wide terraced plane is a floor: there is nowhere you have to
be and nothing to get past, so crossing it is rolling. A board made of **pieces**
is a board with places in it.

Every piece knows two things — where it is **entered** and where it is **left**,
each a cell, a facing and a height — and puts its own cells, walls and mechanisms
down relative to the first. Chaining them is what makes a board, and because a
piece is a function of its entry, the same kicker can be the third thing on one
board and the first on another.

There are two kinds, and the difference is the whole design:

**Places** — :func:`plateau` — are somewhere to be. Walled, wide enough to move
about in, and they ask nothing.

**Joiners** are the ways between places, and each one *is* a challenge, because
it imposes a rule a player has to satisfy to get through:

:func:`kicker`
    A dip whose far side climbs higher than its near side dropped.
:func:`spillway`
    A descent with no wall at the bottom: control it or overshoot.
:func:`hairpin`
    A right-angle you have to brake for, or be carried past.
:func:`bridge`
    One cell wide, with nothing either side.
:func:`scatter`
    A field of bumpers that flings you about.
:func:`ramp_down`
    The way between two terraces that asks nothing; the plain connector.

A rule is only a rule if it can be failed, so each is tested by putting a marble
on it slowly and quickly and asking what happened. A ramp everything climbs is
scenery.

    >>> board = chain(3, ['plateau', 'kicker', 'plateau'])
    >>> joined(board.cells, board.start, board.finish)
    True
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .level import CELL_SIZE, Bumper, Finish, Level, Ramp, Wall

__all__ = ['Port', 'Piece', 'Board', 'Theme', 'THEMES', 'PIECES', 'MAX_STEP',
           'DESIGN_TILT',
           'joined', 'distance', 'navigable', 'reachable_through',
           'chain', 'plateau', 'ramp_down', 'kicker', 'spillway',
           'hairpin', 'bridge', 'scatter']

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))

#: The downhill board lean, in radians, that every rule here and in the fragment
#: library was built and measured against.
#:
#: A rule is about what a piece does to a marble being *carried* through it -- a
#: plank that sinks under one that dawdles, a dip whose far side has to be
#: climbed -- so a piece measured on a board that carries nothing measures
#: nothing.  In a board a player is given, the pace comes from the descent the
#: level was built with (:data:`~openglcontext_marble_demo.boards.TERRACE_STEP`)
#: and from the player; :data:`~openglcontext_marble_demo.game.BASE_TILT` is
#: zero, because a lean the board applies by itself is one a player can add to
#: but never cancel.  A single piece on its own has neither, so anything asking
#: a piece what it does supplies this instead.
DESIGN_TILT = math.radians(8.0)

#: Largest step between neighbouring cells anything here may build.
MAX_STEP = 0.9

#: How wide a way between places is, in cells.
LANE = 3

_SIDE_OF = {(0, -1): 'N', (0, 1): 'S', (1, 0): 'E', (-1, 0): 'W'}


@dataclass(frozen=True)
class Theme:
    """What an area is made of: a look, a feel, and a name for its sounds.

    ``floor`` and ``wall`` are materials from
    :mod:`~openglcontext_marble_demo.materials`, so a theme is grip as much as
    colour -- an icy room is a different room to drive in, not a differently
    coloured one.  ``sound`` names the set a floor and a wall would be heard
    through; nothing plays it yet, and it is here so that the boards being built
    now say which one they wanted.
    """
    name: str
    floor: str
    wall: str
    sound: str = 'stone'


#: The areas a board can be made of.  A board that changes underfoot as you move
#: through it is a board you can navigate by.
THEMES = {
    'stone': Theme('stone', floor='stone', wall='stone', sound='stone'),
    'ice': Theme('ice', floor='ice_sheet', wall='stone', sound='ice'),
    'foundry': Theme('foundry', floor='metal', wall='metal', sound='metal'),
    'rubber': Theme('rubber', floor='rubber_pad', wall='rubber_pad',
                    sound='rubber'),
}


@dataclass(frozen=True)
class Port:
    """Where a piece is entered or left: a cell, a facing, a height, a width."""
    cell: tuple
    facing: tuple = (0, 1)
    height: float = 0.0
    width: int = LANE

    def ahead(self, steps: Any=1) -> Any:
        """The port ``steps`` cells further along the way it faces."""
        return Port(cell=(self.cell[0] + self.facing[0] * steps,
                          self.cell[1] + self.facing[1] * steps),
                    facing=self.facing, height=self.height, width=self.width)

    def across(self) -> Any:
        """The unit step across the facing, which is the way a mouth is wide."""
        return (-self.facing[1], self.facing[0])

    def cells(self) -> Any:
        """The cells of the mouth, centred on :attr:`cell`."""
        side = self.across()
        half = self.width // 2
        return [(self.cell[0] + side[0] * offset, self.cell[1] + side[1] * offset)
                for offset in range(-half, self.width - half)]

    def turned(self, facing: Any) -> Any:
        """The same place, facing another way."""
        return Port(cell=self.cell, facing=facing, height=self.height,
                    width=self.width)


@dataclass
class Piece:
    """One place or one way between places, already positioned.

    ``exits`` is a mapping rather than a single port, because a story needs
    somewhere for a player who gets it wrong to *go*: ``'ok'`` is the way on and
    is always there, and a piece may offer others -- ``'missed'`` for a turn-off
    that was taken too fast -- which lead somewhere slower rather than ending
    the run.  :attr:`exit` is the ``'ok'`` one, which is what a plain chain uses.
    """
    name: str
    cells: dict
    entry: Port
    exits: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    theme: str = 'stone'
    #: What this piece asks of a player, in one line; empty for a place.
    rule: str = ''

    def __post_init__(self) -> None:
        if 'ok' not in self.exits:
            raise ValueError('%s has no way on: exits needs an "ok"' % self.name)

    @property
    def exit(self) -> Any:
        """The way on, which is what a chain follows."""
        return self.exits['ok']

    def level(self, **named: Any) -> Any:
        """This piece on its own, as a playable level -- which is how a joiner's
        rule is tested: put a marble on it and see whether the rule held."""
        return _level([self], self.entry.cell, self.exit.cell, **named)


@dataclass
class Board:
    """A chain of pieces, and the region they make together."""
    pieces: list
    cells: dict
    start: tuple
    finish: tuple

    def level(self, **named: Any) -> Any:
        return _level(self.pieces, self.start, self.finish, **named)


# -- reachability ---------------------------------------------------------

def joined(cells: Any, origin: Any, target: Any) -> Any:
    """Whether ``target`` can be reached from ``origin`` over ``cells``."""
    if origin not in cells:
        return False
    seen = {origin}
    queue = deque([origin])
    while queue:
        col, row = queue.popleft()
        if (col, row) == target:
            return True
        for dcol, drow in NEIGHBOURS:
            nxt = (col + dcol, row + drow)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return target in seen


def distance(cells: Any, origin: Any, target: Any) -> Any:
    """Steps from ``origin`` to ``target`` over ``cells``, or None if unreachable."""
    if origin not in cells:
        return None
    seen = {origin: 0}
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        if cell == target:
            return seen[cell]
        for dcol, drow in NEIGHBOURS:
            nxt = (cell[0] + dcol, cell[1] + drow)
            if nxt in cells and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return None


def walled_pairs(level: Any) -> Any:
    """Every ordered pair of neighbouring cells a wall stands between."""
    blocked = set()
    for feature in level.features:
        if isinstance(feature, Wall):
            step = Wall.OFFSET[feature.side]
            beyond = (feature.cell[0] + step[0], feature.cell[1] + step[1])
            blocked.add((feature.cell, beyond))
            blocked.add((beyond, feature.cell))
    return blocked


def reachable_through(level: Any, origin: Any=None) -> Any:
    """The cells a marble can reach, **through the gaps between walls**.

    :func:`joined` asks whether the *cells* join up, and a wall is not a cell —
    which is how every board could be reported connected while a wall stood
    across the only way forward.  This is the question a marble asks.
    """
    origin = level.start_cell if origin is None else origin
    if origin not in level.cells:
        return set()
    walls = walled_pairs(level)
    seen = {origin}
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        for dcol, drow in NEIGHBOURS:
            beside = (cell[0] + dcol, cell[1] + drow)
            if (beside in level.cells and beside not in seen
                    and (cell, beside) not in walls):
                seen.add(beside)
                queue.append(beside)
    return seen


def navigable(level: Any) -> Any:
    """Whether a marble can get from the start of ``level`` to its finish."""
    return level.finish_cell in reachable_through(level)


def open_the_joins(cells: Any, features: Any) -> Any:
    """Drop every wall that turned out to stand between two cells of the board.

    A piece walls its own edge against the cells *it* knows about, and the piece
    that joins onto it is laid afterwards — so a wall that faced the void when
    it was placed ends up between two floors.  Every board built by chaining had
    one of those across it.

    Deciding this when the board is finished rather than when a piece is built
    is the only place it *can* be decided: no piece knows what will be put next
    to it.
    """
    return [feature for feature in features
            if not (isinstance(feature, Wall)
                    and (feature.cell[0] + Wall.OFFSET[feature.side][0],
                         feature.cell[1] + Wall.OFFSET[feature.side][1]) in cells)]


# -- building blocks ------------------------------------------------------

def _lay(cells: Any, port: Any, length: Any, height: Any=None, width: Any=None) -> Any:
    """Fill ``length`` cells of lane from ``port`` along its facing.

    Answers the port at the far end.  Everything here is built out of this:
    a place is a wide short run, a ramp is a narrow one that steps down.
    """
    width = port.width if width is None else width
    height = port.height if height is None else height
    where = Port(cell=port.cell, facing=port.facing, height=port.height,
                 width=width)
    for step in range(length):
        at = where.ahead(step)
        for cell in at.cells():
            cells[cell] = height
    return Port(cell=where.ahead(length - 1).cell, facing=port.facing,
                height=height, width=width)


def _slope(cells: Any, features: Any, port: Any, length: Any, drop: Any, width: Any=None) -> Any:
    """A run of ``length`` cells falling ``drop`` in total, in equal steps.

    The step is capped at :data:`MAX_STEP`, so a slope that asked for more than
    it has room for is made longer rather than made uncrossable.  The run ends
    level: the last cell is the floor the slope arrives at, which is why
    ``drop`` is divided between one fewer steps than there are cells.

    **Every step gets a ramp.**  Cells at stepped heights are a staircase, and a
    marble rolls *down* a staircase perfectly well and cannot roll up one at all:
    each riser is a wall in front of it.  A
    :class:`~openglcontext_marble_demo.level.Ramp` tilts the tile so its far edge
    meets the next one, which is a surface rather than a step -- and it is why
    the kicker works in one direction and not the other without them.

    They are ramps with no boost on them: a slope is shape, and what a marble
    carries down one is what it arrived with plus what the drop gives it.
    """
    width = port.width if width is None else width
    steps = max(length, math.ceil(abs(drop) / MAX_STEP) + 1, 2)
    each = drop / (steps - 1)
    where = Port(cell=port.cell, facing=port.facing, height=port.height,
                 width=width)
    height = port.height
    for step in range(steps):
        at = where.ahead(step)
        for cell in at.cells():
            cells[cell] = height
            if step < steps - 1 and abs(each) > 1e-6:
                features.append(Ramp(cell=cell, direction=port.facing, rise=each,
                                     boost_speed=None))
        height = round(height + each, 6)
    return Port(cell=where.ahead(steps - 1).cell, facing=port.facing,
                height=round(port.height + each * (steps - 1), 6), width=width)


def _rails(cells: Any, port: Any, length: Any, sides: Any=('left', 'right'), width: Any=None) -> Any:
    """Walls down one or both sides of a run, facing outward."""
    width = port.width if width is None else width
    across = port.across()
    half = width // 2
    made = []
    for step in range(length):
        at = port.ahead(step)
        if 'left' in sides:
            cell = (at.cell[0] - across[0] * half, at.cell[1] - across[1] * half)
            if cell in cells:
                made.append(Wall(cell=cell, side=_SIDE_OF[(-across[0], -across[1])]))
        if 'right' in sides:
            offset = width - 1 - half
            cell = (at.cell[0] + across[0] * offset, at.cell[1] + across[1] * offset)
            if cell in cells:
                made.append(Wall(cell=cell, side=_SIDE_OF[across]))
    return made


def _ring(cells: Any, region: Any, gaps: Any=()) -> Any:
    """Walls round the outside of ``region``, except where a way leads out."""
    spared = set(gaps)
    made = []
    for cell in sorted(region):
        if cell in spared:
            continue
        for step, side in _SIDE_OF.items():
            beside = (cell[0] + step[0], cell[1] + step[1])
            if beside not in cells:
                made.append(Wall(cell=cell, side=side))
    return made


# -- places ---------------------------------------------------------------

def plateau(rng: Any, entry: Any, theme: Any='stone', across: Any=None, along: Any=None) -> Any:
    """Somewhere to be: a walled square, wide enough to move about in.

    It asks nothing.  What a plateau is for is giving a player room to set up
    for whatever comes next, and somewhere to end up when they get it wrong.
    """
    across = across or rng.choice((5, 5, 7))
    along = along or rng.choice((4, 5, 6))
    cells: dict = {}
    end = _lay(cells, entry, along, width=across)
    exit_port = Port(cell=end.cell, facing=entry.facing, height=entry.height,
                     width=LANE)
    ways_out = set(entry.cells()) | set(exit_port.cells())
    return Piece(name='plateau', cells=cells, entry=entry, exits={'ok': exit_port},
                 features=_ring(cells, set(cells), gaps=ways_out), theme=theme)


# -- joiners --------------------------------------------------------------

def ramp_down(rng: Any, entry: Any, theme: Any='stone', drop: Any=None, length: Any=None) -> Any:
    """The plain way between two terraces: a walled slope that asks nothing.

    Every board needs one of these.  A challenge is only a challenge against
    something that is not, and a board of nothing but challenges is a board
    nobody gets across.
    """
    drop = -abs(drop if drop is not None else rng.choice((1.8, 2.7, 3.6)))
    length = length or 4
    cells: dict = {}
    slopes: list = []
    end = _slope(cells, slopes, entry, length, drop)
    walls = _rails(cells, entry, len(cells) // entry.width, width=entry.width)
    return Piece(name='ramp_down', cells=cells, entry=entry, exits={'ok': end},
                 features=slopes + walls, theme=theme)


def kicker(rng: Any, entry: Any, theme: Any='stone', depth: Any=None, lift: Any=None) -> Any:
    """A dip whose far side climbs higher than its near side dropped.

    A short dip and a long way back up, leaving ``lift`` above where it was
    entered.  What that costs is time: the dip hands back what it took, so what
    is left for the climb is the speed the marble brought to it, and a marble
    that arrives at a crawl is still on the far side when a fast one is long
    gone.  Walled along both sides, because the answer to it is speed rather
    than aim.

    The far side has to *rise* for that to be true.  A board that leans downhill
    at a gradient of 0.22 carries a marble up anything shallower for nothing, so
    a dip that came back to the height it started at would be a piece that asks
    for nothing at all.
    """
    depth = abs(depth if depth is not None else rng.choice((1.8, 2.7)))
    lift = abs(lift if lift is not None else rng.choice((3.6, 4.5)))
    cells: dict = {}
    slopes: list = []
    bottom = _slope(cells, slopes, entry, 2, -depth)
    flat = _lay(cells, bottom.ahead(1), 1, height=bottom.height)
    top = _slope(cells, slopes, flat.ahead(1), 5, depth + lift)
    length = max(row for _, row in cells) - min(row for _, row in cells) + 1
    walls = _rails(cells, entry, length + 2)
    return Piece(name='kicker', cells=cells, entry=entry, exits={'ok': top},
                 features=slopes + walls, theme=theme,
                 rule='carry speed into it or crawl out the far side')


def spillway(rng: Any, entry: Any, theme: Any='stone', drop: Any=None, run_out: Any=4) -> Any:
    """A descent with no wall at the bottom: control it, or overshoot.

    Walled down both sides and open at the end, with a flat run-out.  The rule
    is a rule about the *brakes*, and what it is measured in is how much of the
    run-out is left: a marble that comes down it headlong is off the end almost
    as soon as it reaches the bottom, and one that is held back has the length
    of the run-out to do something about it.

    The run-out is time, not a barrier.  A board leaning downhill pulls a marble
    along a flat as hard as anything on the flat can hold it back, so what the
    run-out gives a player is the seconds before the drop rather than a stop.
    """
    drop = -abs(drop if drop is not None else rng.choice((1.8, 2.7)))
    cells: dict = {}
    slopes: list = []
    bottom = _slope(cells, slopes, entry, 5, drop)
    end = _lay(cells, bottom.ahead(1), run_out, height=bottom.height)
    length = max(row for _, row in cells) - min(row for _, row in cells) + 1
    # Sides only.  The missing wall at the bottom is the whole piece.
    return Piece(name='spillway', cells=cells, entry=entry, exits={'ok': end},
                 features=slopes + _rails(cells, entry, length + 2), theme=theme,
                 rule='hold the descent or run off the open end')


def hairpin(rng: Any, entry: Any, theme: Any='stone') -> Any:
    """A right-angle you have to brake for, or be carried past.

    A short straight, a turn, and a wall across where a marble that did not slow
    down will be.  Taking it is the quick way on; missing it leaves the player in
    the corner with no speed, which costs the time the turn would have saved.
    """
    turn = rng.choice(((1, 0), (-1, 0)))
    if entry.facing in ((1, 0), (-1, 0)):
        turn = rng.choice(((0, 1), (0, -1)))
    cells: dict = {}
    straight = _lay(cells, entry, 4)
    corner = Port(cell=straight.cell, facing=turn, height=entry.height,
                  width=entry.width)
    end = _lay(cells, corner.ahead(1), 4)
    # The outside of the corner is walled; the inside is where the line is.
    walls = _rails(cells, entry, 4, sides=('left', 'right'))
    walls += _rails(cells, corner.ahead(1), 4, sides=('left', 'right'))
    walls = [wall for wall in walls if wall.cell in cells]
    return Piece(name='hairpin', cells=cells, entry=entry, exits={'ok': end},
                 features=walls, theme=theme,
                 rule='brake for the right-angle or be carried past it')


def bridge(rng: Any, entry: Any, theme: Any='stone', length: Any=None) -> Any:
    """One cell wide, with nothing either side.

    No walls: a bridge with rails is a corridor.  What it asks is that the
    marble is going straight when it arrives, which is a thing the piece before
    it decides.
    """
    length = length or rng.choice((4, 5, 6))
    cells: dict = {}
    mouth = _lay(cells, entry, 1)
    narrow = _lay(cells, mouth.ahead(1), length, width=1)
    end = _lay(cells, narrow.ahead(1), 1, width=entry.width)
    return Piece(name='bridge', cells=cells, entry=entry,
                 exits={'ok': Port(cell=end.cell, facing=entry.facing,
                                   height=entry.height, width=entry.width)},
                 theme=theme, rule='cross a single cell with nothing beside it')


def scatter(rng: Any, entry: Any, theme: Any='rubber', length: Any=None) -> Any:
    """A field of bumpers that flings you about.

    Getting through is partly luck, which is what it is for: it is the piece
    that makes two runs at the same board different runs.  Walled, so being
    flung about does not simply end the attempt, and wide enough that there is
    somewhere for a marble to be flung *to*.
    """
    length = length or rng.choice((5, 6, 7))
    cells: dict = {}
    end = _lay(cells, entry, length, width=5)
    across = entry.across()
    posts = []
    for step in range(1, length - 1):
        at = entry.ahead(step)
        for offset in (-1, 0, 1):
            if (step + offset) % 2:
                continue
            cell = (at.cell[0] + across[0] * offset, at.cell[1] + across[1] * offset)
            if cell in cells:
                posts.append(Bumper(cell=cell))
    walls = _rails(cells, entry, length, width=5)
    return Piece(name='scatter', cells=cells, entry=entry,
                 exits={'ok': Port(cell=end.cell, facing=entry.facing,
                                   height=entry.height, width=entry.width)},
                 features=walls + posts, theme=theme,
                 rule='get through a field of bumpers that will not have you straight')


#: Every piece a board can be built from, by the name a story calls it.
PIECES = {
    'plateau': plateau,
    'ramp_down': ramp_down,
    'kicker': kicker,
    'spillway': spillway,
    'hairpin': hairpin,
    'bridge': bridge,
    'scatter': scatter,
}


# -- chaining -------------------------------------------------------------

def chain(seed: int, names: Any, entry: Any=None, themes: Any=None) -> Any:
    """Build the pieces ``names`` in order, each entered where the last was left.

    ``themes`` is a name per piece, or one name for all of them, or None for
    whatever each piece prefers -- so a story can say *this room is ice* without
    the piece having to know it will ever be.
    """
    import random
    rng = random.Random(seed)
    where = entry or Port(cell=(0, 0), facing=(0, 1), height=0.0, width=LANE)
    if isinstance(themes, str):
        themes = [themes] * len(names)
    built: list = []
    cells: dict = {}
    for index, name in enumerate(names):
        if name not in PIECES:
            raise KeyError('no piece called %r' % (name,))
        named: dict[str, Any] = {}
        if themes is not None and themes[index] is not None:
            named['theme'] = themes[index]
        piece = PIECES[name](rng, where, **named)
        built.append(piece)
        cells.update(piece.cells)
        where = piece.exit
    return Board(pieces=built, cells=cells, start=built[0].entry.cell,
                 finish=built[-1].exit.cell)


def _level(built: Any, start: Any, finish: Any, name: str='chained', time_limit: Any=None,
           cell_size: float=CELL_SIZE, extra_cells: Any=None) -> Any:
    """The cells and features of ``built`` as a playable level.

    ``extra_cells`` is board that belongs to no piece -- the runs a story lays
    to join a shifted branch back on.  Without them a board is rebuilt from its
    pieces alone and every connector becomes a one-cell hole, which is exactly
    what left branches stranded.
    """
    cells: dict = dict(extra_cells or {})
    surfaces: dict = {}
    features: list = []
    for piece in built:
        theme = THEMES[piece.theme]
        cells.update(piece.cells)
        for cell in piece.cells:
            surfaces[cell] = theme.floor
        for feature in piece.features:
            if isinstance(feature, Wall):
                feature = Wall(cell=feature.cell, side=feature.side,
                               height=feature.height, thickness=feature.thickness,
                               material=theme.wall)
            features.append(feature)
    features = open_the_joins(cells, features)
    features.append(Finish(finish))
    if time_limit is None:
        time_limit = round(len(cells) * 0.5 + 15.0, 1)
    return Level(name=name, cells=cells, start_cell=start, finish_cell=finish,
                 time_limit=time_limit, features=features,
                 cell_surfaces=surfaces, cell_size=cell_size)
