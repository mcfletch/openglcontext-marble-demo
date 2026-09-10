"""The shape of a board: a spine, the width around it, and the ways round.

A board is not a path. A path one cell wide has one solution, and every hazard
put on it is a gate rather than something to steer around — so this builds a
*region*, in four steps that each answer one question:

:func:`carve_spine`
    Where does the route go? A forward-biased walk from the start toward the
    finish, with sideways runs bounded, so the direction the board leans and the
    direction the route goes are the same direction.
:func:`widen`
    How much room is there? Two or three cells across, opening into plazas, so a
    hazard leaves a lane beside it.
:func:`braid`
    Is there more than one way? Strands that leave the spine and rejoin it
    further down — the loops that turn a maze from a solution into a choice.
:func:`terrace`
    How far has the marble come? Height falls with distance from the start, in
    flat runs with steps between them, which gives the terraced look *and*
    guarantees the slope budget: neighbouring cells are one step of the distance
    apart at most, so they are one terrace apart at most.

The result is a :class:`Board` — cells, heights, and the **strands** that
decoration hangs on, because "the short way is the dangerous one" is a statement
about strands and cannot be said about loose cells.

Nothing here draws anything or touches the physics. It is grid arithmetic, and
:mod:`~openglcontext_marble_demo.generator` turns what comes out into a
:class:`~openglcontext_marble_demo.level.Level`.

    >>> import random
    >>> board = build(random.Random(1), length=14, width=3, difficulty=2)
    >>> board.start in board.cells and board.finish in board.cells
    True
    >>> board.finish[1] > board.start[1]          # downhill is forward
    True
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ['Cell', 'Strand', 'Board', 'SPINE', 'SHORTCUT', 'SCENIC',
           'MAX_SIDEWAYS_RUN', 'TERRACE_RUN', 'TERRACE_STEP',
           'carve_spine', 'widen', 'braid', 'terrace', 'build']

Cell = tuple

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FORWARD = (0, 1)
SIDEWAYS = ((1, 0), (-1, 0))

#: The longest run of cells a route may go sideways before it must advance
#: again.  The board leans one way; a route that ran across that lean for long
#: would have a marble nobody is steering pushed over the edge rather than along.
MAX_SIDEWAYS_RUN = 3

#: How many cells of progress share a terrace, and how far the next one drops.
#: The step is what the slope budget is spent on, so it stands alone here rather
#: than being scattered through the walk.
TERRACE_RUN = 3
TERRACE_STEP = 0.9

#: What a strand is for.  Decoration reads this and nothing else about a strand:
#: the spine is the way everyone goes, a shortcut is quicker and is where the
#: hazards live, and a scenic strand is the long way round and stays clear.
SPINE = 'spine'
SHORTCUT = 'shortcut'
SCENIC = 'scenic'


@dataclass
class Strand:
    """A run of cells that means something together.

    ``cells`` is the run in order; ``kind`` is one of :data:`SPINE`,
    :data:`SHORTCUT` or :data:`SCENIC`; ``rejoins`` is how many spine cells it
    bypasses, so a strand knows whether taking it saves anything.
    """
    cells: list
    kind: str = SPINE
    rejoins: int = 0

    @property
    def saves(self) -> int:
        """Cells saved by taking this strand; negative for the long way round."""
        return self.rejoins - (len(self.cells) - 1)


@dataclass
class Board:
    """A region of cells with a route across it and strands hanging on it."""
    cells: set = field(default_factory=set)
    heights: dict = field(default_factory=dict)
    spine: list = field(default_factory=list)
    strands: list = field(default_factory=list)
    start: tuple = (0, 0)
    finish: tuple = (0, 0)

    def strands_of(self, kind: Any) -> Any:
        return [strand for strand in self.strands if strand.kind == kind]

    def route_exists(self, blocked: Any=frozenset()) -> Any:
        """Whether start and finish are still joined avoiding ``blocked``."""
        return reachable(self.cells, self.start, blocked).issuperset({self.finish})


# -- reachability ---------------------------------------------------------

def reachable(cells: Any, origin: Any, blocked: Any=frozenset()) -> Any:
    """Every cell reachable from ``origin`` over ``cells``, avoiding ``blocked``."""
    if origin in blocked or origin not in cells:
        return set()
    seen = {origin}
    queue = deque([origin])
    while queue:
        col, row = queue.popleft()
        for dcol, drow in NEIGHBOURS:
            nxt = (col + dcol, row + drow)
            if nxt in cells and nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def distances(cells: Any, origin: Any) -> Any:
    """Steps from ``origin`` to every reachable cell, over ``cells``."""
    found = {origin: 0}
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        for dcol, drow in NEIGHBOURS:
            nxt = (cell[0] + dcol, cell[1] + drow)
            if nxt in cells and nxt not in found:
                found[nxt] = found[cell] + 1
                queue.append(nxt)
    return found


# -- the spine ------------------------------------------------------------

def carve_spine(rng: Any, length: Any) -> Any:
    """A forward-biased walk of ``length`` cells from ``(0, 0)``.

    Forward is +row, which is the way the board leans, so rolling and
    progressing are the same thing.  Sideways moves wander the route without
    letting it run across the lean: after
    :data:`MAX_SIDEWAYS_RUN` of them in a row the next move must advance, and a
    sideways move never reverses the one before it, which would only jitter.
    """
    cell = (0, 0)
    spine = [cell]
    seen = {cell}
    sideways = 0
    last_side = None
    while len(spine) < length:
        # The first move and the last are forward, so a board is left and
        # entered facing downhill rather than across it.
        if len(spine) == 1 or len(spine) == length - 1:
            step = FORWARD
        else:
            step = _spine_step(rng, sideways, last_side)
        nxt = (cell[0] + step[0], cell[1] + step[1])
        if nxt in seen:
            step, nxt = FORWARD, (cell[0], cell[1] + 1)
            if nxt in seen:                     # cannot happen: forward is new
                break                           # pragma: no cover
        seen.add(nxt)
        spine.append(nxt)
        cell = nxt
        if step in SIDEWAYS:
            sideways += 1
            last_side = step
        else:
            sideways = 0
            last_side = None
    return spine


def _spine_step(rng: Any, sideways: Any, last_side: Any) -> Any:
    """Forward, or a turn — never a reversal, and never a long sideways run."""
    if sideways >= MAX_SIDEWAYS_RUN:
        return FORWARD
    choices = [FORWARD, FORWARD, FORWARD, (1, 0), (-1, 0)]
    if last_side is not None:
        opposite = (-last_side[0], -last_side[1])
        choices = [step for step in choices if step != opposite]
    return rng.choice(choices)


# -- width ----------------------------------------------------------------

def widen(run: Any, radius: Any=1, plaza_every: Any=5, plaza_radius: Any=2) -> Any:
    """Every cell within ``radius`` of ``run``, opening into plazas at intervals.

    Room is measured **around** the run rather than across it, and that is the
    whole of why: a lane widened only sideways leaves the places the route goes
    sideways one cell tall, which is the one-cell corridor again with the board
    turned ninety degrees — and a corridor has a cell in it that every route has
    to pass through.  A radius has no direction to get wrong.

    ``radius`` 1 gives a lane three cells across.  Every ``plaza_every`` cells it
    opens to ``plaza_radius``: somewhere a rotating arm is a hazard with room to
    read it, and somewhere two strands can part and meet without a pinch.
    """
    cells = set()
    for index, (col, row) in enumerate(run):
        here = plaza_radius if index % plaza_every == plaza_every - 1 else radius
        for dcol in range(-here, here + 1):
            span = here - abs(dcol)
            for drow in range(-span, span + 1):
                cells.add((col + dcol, row + drow))
    return cells


# -- the ways round -------------------------------------------------------

def braid(rng: Any, spine: Any, count: int=2, reach: Any=(3, 7)) -> Any:
    """Strands that leave the spine and rejoin it, and the cells they need.

    Returns ``(strands, added)``: the strands built and the cells they put down.
    Two kinds, and the difference between them is the whole of why a board asks
    a question:

    A **shortcut** cuts a chord across a bend. The spine wanders to make a board
    worth crossing, and wherever it has wandered there is a straighter way from
    one side of the bend to the other — fewer cells, and the natural racing line.
    That is where the hazards go.

    A **scenic** strand bulges out and comes back: more cells than the spine it
    bypasses, and left clear. It is the way round for a player who would rather
    arrive.

    Either way it is a **loop**, and a loop is what makes a maze a choice rather
    than a solution.
    """
    strands: list[Any] = []
    added = set()
    used: set[Any] = set()

    for begin, end in _bends(spine, reach):
        if len(strands) >= count:
            break
        if used & set(range(begin, end + 1)):
            continue
        run = _chord(spine[begin], spine[end], rng)
        if run is None or (len(run) - 1) >= (end - begin):
            continue
        strands.append(Strand(cells=run, kind=SHORTCUT, rejoins=end - begin))
        added.update(run)
        used.update(range(begin, end + 1))

    for _ in range(count * 4):
        if len(strands) >= count + 1:
            break
        if len(spine) < reach[1] + 2:
            break
        begin = rng.randrange(1, len(spine) - reach[0] - 1)
        end = min(begin + rng.randint(*reach), len(spine) - 2)
        if end - begin < reach[0] or used & set(range(begin, end + 1)):
            continue
        run = _detour(spine[begin], spine[end], rng.choice((-1, 1)), rng)
        if run is None:
            continue
        strands.append(Strand(cells=run, kind=SCENIC, rejoins=end - begin))
        added.update(run)
        used.update(range(begin, end + 1))
    return strands, added


def _bends(spine: Any, reach: Any) -> Any:
    """Spans of the spine where a straighter way exists, slackest first.

    The slack is how many cells the spine spends over the shortest grid route
    between the two ends: no slack means the spine is already straight there and
    a chord would save nothing.
    """
    found = []
    for begin in range(1, len(spine) - reach[0] - 1):
        for span in range(reach[0], reach[1] + 1):
            end = begin + span
            if end >= len(spine) - 1:
                break
            direct = (abs(spine[end][0] - spine[begin][0])
                      + abs(spine[end][1] - spine[begin][1]))
            slack = span - direct
            if slack >= 2:
                found.append((slack, begin, end))
    found.sort(key=lambda entry: (-entry[0], entry[1]))
    return [(begin, end) for _, begin, end in found]


def _chord(begin: Any, finish: Any, rng: Any) -> Any:
    """The straight way between two cells: along one axis, then the other."""
    first_along_column = rng.random() < 0.5
    corner = (finish[0], begin[1]) if first_along_column else (begin[0], finish[1])
    run = [begin] + _leg(begin, corner) + _leg(corner, finish)
    return _tidy_run(run)


def _detour(begin: Any, finish: Any, side: Any, rng: Any) -> Any:
    """An L-shaped run from ``begin`` to ``finish`` bulging toward ``side``.

    Out to the side, along, and back in: three straight legs, which is the
    simplest run that is a genuinely different way round rather than a lane of
    the same corridor.  ``None`` when the two ends are not far enough apart in
    ``row`` for the run to be worth having.
    """
    if finish[1] - begin[1] < 2:
        return None
    # Far enough out that the widening around it does not merge it back into
    # the lane it left: a way round a metre to the side is the same journey.
    bulge = rng.randint(4, 6)
    lane = max(begin[0], finish[0]) + bulge if side > 0 \
        else min(begin[0], finish[0]) - bulge
    run = [begin]
    run += _leg(begin, (lane, begin[1]))
    run += _leg((lane, begin[1]), (lane, finish[1]))
    run += _leg((lane, finish[1]), finish)
    return _tidy_run(run)


def _tidy_run(run: Any) -> Any:
    """Drop the repeat where two legs meet at their shared corner."""
    tidy = [run[0]]
    for cell in run[1:]:
        if cell != tidy[-1]:
            tidy.append(cell)
    return tidy


def _leg(begin: Any, end: Any) -> Any:
    """The cells from ``begin`` to ``end`` along one axis, excluding ``begin``."""
    (col, row), (to_col, to_row) = begin, end
    if col != to_col:
        step = 1 if to_col > col else -1
        return [(c, row) for c in range(col + step, to_col + step, step)]
    step = 1 if to_row > row else -1
    return [(col, r) for r in range(row + step, to_row + step, step)]


def prune(cells: Any, protect: Any) -> Any:
    """Drop cells with fewer than two ways off them, except those protected.

    A tile reachable only from one side is a nub: somewhere a marble can go and
    then has to come back out of, which is a dead end wearing the shape of a
    board.  Pruning repeats, because taking one away can leave its neighbour
    just as lonely.
    """
    cells = set(cells)
    while True:
        lonely = {cell for cell in cells
                  if cell not in protect
                  and sum(((cell[0] + dc, cell[1] + dr) in cells)
                          for dc, dr in NEIGHBOURS) < 2}
        if not lonely:
            return cells
        cells -= lonely


# -- height ---------------------------------------------------------------

def terrace(cells: Any, spine: Any, run: Any=TERRACE_RUN, step: Any=TERRACE_STEP) -> Any:
    """Heights that fall with the rank a cell is on, in flat runs.

    Height depends on ``row`` alone — how far *forward* a cell is — which does
    three things at once. A rank is one terrace, so changing lane is never a
    step. Neighbours are one rank apart at most and therefore one terrace apart
    at most, so the slope budget holds by construction rather than by a check
    afterwards. And downhill and forward become the same direction, which is
    what the board's constant lean assumes.
    """
    datum = spine[0][1]
    return {cell: -step * ((cell[1] - datum) // run) for cell in cells}


# -- the whole thing ------------------------------------------------------

def build(rng: Any, length: Any=14, radius: Any=1, plaza_every: Any=5, difficulty: Any=2) -> Any:
    """A complete :class:`Board`: spine, room, braids and heights."""
    spine = carve_spine(rng, length)
    cells = widen(spine, radius=radius, plaza_every=plaza_every)
    strands, added = braid(rng, spine, count=1 + difficulty // 2)
    # A braid laid down after the widening leaves the cells it crossed one wide;
    # giving its own run the same room means a strand is a way round rather than
    # a thread through the middle of one.
    for strand in strands:
        cells |= widen(strand.cells, radius=radius,
                       plaza_every=len(strand.cells) + 1)
    cells |= added
    cells = prune(cells, protect=set(spine))
    strands = [Strand(cells=[c for c in strand.cells if c in cells],
                      kind=strand.kind, rejoins=strand.rejoins)
               for strand in strands]
    board = Board(cells=cells, spine=spine, start=spine[0], finish=spine[-1],
                  strands=[Strand(cells=list(spine), kind=SPINE)] + strands)
    board.heights = terrace(cells, spine)
    return board
