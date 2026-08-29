"""A story: chapters joined by which exit leads where.

A chain is a list, and a list cannot say what happens when a player misses the
turn-off — it can only end the run, which is the least interesting thing that
could happen. A story is a **graph**. Each chapter names a fragment from the
library and says where each of its exits goes, so a hairpin's ``ok`` leads to the
quick way on and its ``missed`` leads to the long way round, and the long way
rejoins. A player who gets it wrong is *behind*, not finished.

    >>> story = Story(name='two ways', start='a', chapters={
    ...     'a': Chapter(id='a', fragment='plateau', exits={'ok': 'turn'}),
    ...     'turn': Chapter(id='turn', fragment='hairpin',
    ...                     exits={'ok': 'quick', 'missed': 'long'}),
    ...     'quick': Chapter(id='quick', fragment='plateau', exits={'ok': 'end'}),
    ...     'long': Chapter(id='long', fragment='scatter', exits={'ok': 'end'}),
    ...     'end': Chapter(id='end', fragment='plateau', exits={})})
    >>> board = story.build(3)
    >>> sorted(board.placed)
    ['a', 'end', 'long', 'quick', 'turn']

**Placement is the hard part**, and it is where a graph differs from a list. The
main line is laid first, following ``ok`` from the start; then each branch is
laid from the exit it leaves by. A branch put down where the main line already is
would be a board with two floors in one place, so a branch that collides is
pushed sideways and tried again, and one that will not fit at all is reported
rather than quietly overlapped.

Where two ways meet again, the second to arrive is **joined** to the first by a
short run of cells rather than being laid on top of it: what makes a rejoin a
rejoin is that both ways reach the same place.
"""
from collections import deque
from dataclasses import dataclass, field

from . import fragments, pieces
from .pieces import Port

__all__ = ['Chapter', 'Story', 'Told', 'MAX_SHIFT']

#: How far sideways a branch may be pushed looking for room, in cells, before
#: the story is reported as one that will not fit.
MAX_SHIFT = 14


@dataclass
class Chapter:
    """One fragment of a story, and where each of its exits leads.

    ``exits`` maps an exit name to another chapter's id.  ``'ok'`` is the way on;
    anything else is where a player ends up having got this chapter wrong, and a
    chapter with no exits at all is the end of the story.
    """
    id: str
    fragment: str
    exits: dict = field(default_factory=dict)
    variant: str | None = None
    theme: str | None = None


@dataclass
class Told:
    """A story that has been laid out: the pieces, and where each one went."""
    story: object
    placed: dict
    cells: dict
    start: tuple
    finish: tuple

    def level(self, **named):
        """This story as a playable level.

        ``self.cells`` rather than the pieces' own, because a story lays runs of
        board that belong to no piece -- the connectors that join a branch back
        on after it was shifted sideways to find room.  Rebuilt from the pieces
        alone, every one of those became a one-cell hole and stranded whatever
        was past it.
        """
        named.setdefault('name', getattr(self.story, 'name', 'story'))
        return pieces._level(list(self.placed.values()), self.start, self.finish,
                             extra_cells=self.cells, **named)

    def rules(self):
        """What this story asks of a player, in the order it asks it."""
        return [(chapter, piece.rule)
                for chapter, piece in self.placed.items() if piece.rule]


@dataclass
class Story:
    """A graph of chapters, and the board it lays out."""
    name: str
    start: str
    chapters: dict = field(default_factory=dict)

    # -- checking -------------------------------------------------------
    def check(self):
        """Raise if this story cannot be laid out at all.

        Every complaint names the thing that is wrong: a story is written by
        hand or by a generator, and "it did not work" is no use to either.
        """
        if self.start not in self.chapters:
            raise ValueError('story %r starts at %r, which is nowhere: it has %s'
                             % (self.name, self.start,
                                ', '.join(sorted(self.chapters)) or 'no chapters'))
        for chapter in self.chapters.values():
            for exit_name, target in chapter.exits.items():
                if target not in self.chapters:
                    raise ValueError(
                        'chapter %r leads by %r to %r, which is missing'
                        % (chapter.id, exit_name, target))

    # -- laying it out --------------------------------------------------
    def build(self, seed=0, entry=None):
        """Lay the story out into a :class:`Told`."""
        import random
        self.check()
        rng = random.Random(seed)
        where = entry or Port(cell=(0, 0), facing=(0, 1), height=0.0,
                              width=pieces.LANE)
        placed: dict = {}
        cells: dict = {}
        # The main line first: it is the spine everything else is placed around.
        pending = deque([(self.start, where, None)])
        while pending:
            name, port, joining = pending.popleft()
            if name in placed:
                # Two ways meeting again.  The first to arrive laid the chapter;
                # the second is joined to it rather than laying a second copy.
                _rejoin(cells, port, placed[name].entry)
                continue
            piece = self._lay(rng, name, port, cells)
            placed[name] = piece
            cells.update(piece.cells)
            chapter = self.chapters[name]
            for exit_name in _in_order(chapter.exits):
                target = chapter.exits[exit_name]
                leaving = piece.exits.get(exit_name)
                if leaving is None:
                    # The fragment does not offer that exit, so the way it names
                    # leaves from the way on instead: a story may ask for a
                    # failure exit of a fragment that has none, and the answer
                    # is a slower route rather than an error.
                    leaving = piece.exit
                pending.append((target, leaving, name))
        finish = _furthest(cells, placed[self.start].entry.cell)
        return Told(story=self, placed=placed, cells=cells,
                    start=placed[self.start].entry.cell, finish=finish)

    def _lay(self, rng, name, port, taken):
        """Build one chapter at ``port``, shifted sideways until it fits."""
        chapter = self.chapters[name]
        across = port.across()
        for shift in _shifts(_emptier_side(taken, port)):
            at = Port(cell=(port.cell[0] + across[0] * shift,
                            port.cell[1] + across[1] * shift),
                      facing=port.facing, height=port.height, width=port.width)
            piece = fragments.build(chapter.fragment, rng, at,
                                    variant=chapter.variant)
            if chapter.theme is not None:
                piece.theme = chapter.theme
            # A piece is entered *at* the cell the last one was left by, so its
            # own mouth stands on board that is already there.  That is the join,
            # not a collision -- `pieces.chain` builds the same way -- and
            # counting it as one shifted every chapter sideways to get clear of
            # the piece it was supposed to be joined to.  Measured before this
            # was excluded: seven chapters of a board laid 3, 3, 3, 3, 5, 7 and 8
            # cells across from the exit they followed, and the board walked
            # thirty-two cells sideways while descending sixty-one.
            joint = set(at.cells()) | set(at.ahead(-1).cells())
            if not (set(piece.cells) - joint) & set(taken):
                if shift:
                    _rejoin(taken, port, piece.entry)
                return piece
        raise ValueError(
            'chapter %r will not fit beside what is already down: nothing free '
            'within %d cells either way of %r' % (name, MAX_SHIFT, port.cell))


def _shifts(prefer=1):
    """Offsets to try, nearest first and alternating sides.

    ``prefer`` is which side gets the first try at each distance.  It matters
    more than it looks: a port facing down the board has ``across()`` pointing
    *west*, so a generator that always tried ``+1`` first put every chapter that
    would not fit one cell west, and the boards marched that way -- measured over
    ten of them, 265 cells of board west of the start against 44 east, and nine
    finishing 25 to 35 cells west.  A board that drifts is a board a player
    spends the run steering across the lean rather than down it.
    """
    yield 0
    for step in range(1, MAX_SHIFT + 1):
        yield step * prefer
        yield -step * prefer


def _emptier_side(taken, port):
    """Which way across ``port`` the board has less of: ``+1`` or ``-1``.

    Trying that side first is what keeps a board from drifting.  It is measured
    against the port rather than the whole board so that a story which has
    genuinely turned a corner goes on turning, rather than being pulled back
    towards a start it has left behind.
    """
    across = port.across()
    here = port.cell
    lead = sum(1 for cell in taken
               if (cell[0] - here[0]) * across[0] + (cell[1] - here[1]) * across[1] > 0)
    other = sum(1 for cell in taken
                if (cell[0] - here[0]) * across[0] + (cell[1] - here[1]) * across[1] < 0)
    return 1 if lead <= other else -1


def _in_order(exits):
    """``ok`` first, so the main line is laid before anything hangs off it."""
    return (['ok'] if 'ok' in exits else []) + sorted(
        name for name in exits if name != 'ok')


def _rejoin(cells, leaving, arriving):
    """Fill the cells between two ports so both ways reach the same place.

    An L of single cells at the height of the port being left.  It is a
    connector rather than a piece: what makes a rejoin a rejoin is only that the
    two ways meet, and anything more elaborate is a fragment somebody should
    have written into the story.
    """
    (col, row), (to_col, to_row) = leaving.cell, arriving.cell
    height = leaving.height
    step = 1 if to_col >= col else -1
    for at in range(col, to_col + step, step):
        cells.setdefault((at, row), height)
    step = 1 if to_row >= row else -1
    for at in range(row, to_row + step, step):
        cells.setdefault((to_col, at), height)


def _furthest(cells, origin):
    """The cell furthest from ``origin`` over the board, which is the end of it.

    Worked out rather than declared: a story's last chapter is whichever one no
    exit leads out of, and on a graph with a rejoin that is not the one laid
    last.  The furthest cell is the one a player is trying to reach.
    """
    seen = {origin: 0}
    queue = deque([origin])
    furthest, distance = origin, 0
    while queue:
        cell = queue.popleft()
        for dcol, drow in pieces.NEIGHBOURS:
            nxt = (cell[0] + dcol, cell[1] + drow)
            if nxt in cells and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                if seen[nxt] > distance:
                    furthest, distance = nxt, seen[nxt]
                queue.append(nxt)
    return furthest
