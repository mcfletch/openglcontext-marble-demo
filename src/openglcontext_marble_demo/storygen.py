"""Composing a whole story out of the library.

A board picked at random from the library would be as samey as one shape
repeated, because what a player notices is not which fragments are on a board but
the **rhythm** of them: somewhere to gather yourself, then something that asks a
question, then somewhere to land. Eight questions in a row is unreadable and
eight places in a row is a corridor.

So composing is a set of rules about shape rather than a shuffle:

**Places and questions alternate.** Every joiner is followed by somewhere to
arrive, and a board begins and ends in a place — a challenge on the first chapter
is one a player meets from a standstill, and one on the last is a board that ends
mid-sentence.

**The same question is not asked twice running.** Two hairpins in a row is one
idea and a repetition, and a long board asks at least two different kinds.

**It gets harder.** Fragments carry roughly what getting them wrong costs, and
the cheap ones are dealt into the first half. What a player meets first should
cost less than what they meet last, and ``difficulty`` moves the whole board.

**Something has a way round it.** Where a chosen fragment offers a failure exit,
it is wired to a slower way that rejoins — which is the whole point of a story
being a graph, and the difference between getting it wrong and being finished.

**A gate gets a run-up.** A piece cannot manufacture speed for the piece after
it: a marble driven down a descent settles at about five metres a second however
far it has fallen, because steering across the board's lean spends the pull. So
a fragment tagged :data:`GATE_TAG`, which asks to be arrived at fast, has a
:data:`RUN_UP_TAG` fragment dealt in front of it. Without one it is a wall that
happens to be shaped like a challenge.

    >>> story = compose(3, chapters=8)
    >>> board = story.build(3)
    >>> bool(board.cells)
    True

What comes out is a :class:`~openglcontext_marble_demo.stories.Story`, so it lays
out, saves and plays like one written by hand — and can be opened in the editor
and changed a tile at a time, which is what makes it a starting point rather than
somebody else's board.
"""
import random

from . import fragments
from .stories import Chapter, Story

__all__ = ['compose', 'PLACE_TAG', 'GATE_TAG', 'RUN_UP_TAG', 'DIFFICULTY']

#: The tag that marks somewhere to be rather than something to get past.
PLACE_TAG = 'place'

#: The tag on a fragment that asks to be arrived at fast, and the tag on one that
#: is somewhere the speed can come from.
GATE_TAG = 'gate'
RUN_UP_TAG = 'run-up'

#: How much of the library's cost range a board is allowed, per difficulty from
#: 1 to 5.  A difficulty is a *ceiling* rather than a target: an easy board is
#: one with nothing expensive on it, not one with cheap things forced onto it.
DIFFICULTY = {1: 2.0, 2: 3.5, 3: 5.0, 4: 6.5, 5: 99.0}


def compose(seed=0, chapters=8, difficulty=3, themes=None):
    """A story of about ``chapters`` chapters, composed from the library.

    ``chapters`` is the length of the main line; a board may come out longer,
    because a way round something is chapters too.  ``difficulty`` caps how
    expensive a fragment may be.  ``themes`` is a list to draw from, or None to
    use whatever each fragment prefers.
    """
    rng = random.Random(seed)
    library = fragments.library()
    places = _by_kind(library, place=True)
    joiners = _by_kind(library, place=False, ceiling=DIFFICULTY.get(difficulty,
                                                                   5.0))
    if not places:
        raise ValueError('the library has nowhere to be: no fragment tagged %r'
                         % (PLACE_TAG,))
    line = _main_line(rng, library, places, joiners, chapters)
    story = Story(name='seed-%d-c%d-d%d' % (seed, chapters, difficulty),
                  start=line[0][0], chapters={})
    _write(story, line, rng, themes)
    _add_a_way_round(story, line, library, rng, themes, places, joiners)
    return story


# -- choosing ---------------------------------------------------------------

def _by_kind(library, place, ceiling=None):
    """The fragments that are places, or the ones that are not."""
    found = []
    for name, entry in sorted(library.items()):
        is_place = not entry.rule
        if is_place != place:
            continue
        if ceiling is not None and entry.cost > ceiling:
            continue
        found.append(name)
    return found


def _main_line(rng, library, places, joiners, chapters):
    """The spine, as ``(id, fragment)`` pairs: place, joiner, place, joiner...

    Joiners are dealt cheapest-first into the first half and dearest into the
    second, so a board gets harder as it goes rather than by luck.
    """
    wanted = max(3, int(chapters))
    picks = _pick_joiners(rng, library, joiners, wanted // 2)

    # The run-ups come out of the chapter budget rather than on top of it.  A
    # caller asking for eight chapters is sizing a board -- how long it takes to
    # play, how far the pilot has to get -- and one that grew by a third because
    # of a rule the caller cannot see is one they cannot size.
    #
    # Which is why the joiners that will not fit are dropped here rather than
    # the line being trimmed afterwards: they are dealt cheap-first so that a
    # board gets harder as it goes, and cutting the tail off the line would cut
    # exactly the hard end of it.  Dropping from the middle leaves both ends.
    while len(picks) > 1 and _length_with_run_ups(library, picks) > wanted:
        picks.pop(len(picks) // 2)

    line: list = []
    for pick in picks:
        line.append(('c%d' % len(line), _unlike(rng, places, line)))
        line.append(('c%d' % len(line), pick))
    line.append(('c%d' % len(line), _unlike(rng, places, line)))
    return _give_the_gates_a_run_up(library, line)


def _length_with_run_ups(library, picks):
    """How many chapters ``picks`` becomes, once places and run-ups are in.

    A place before each joiner and one to end on, plus a run-up for every joiner
    that asks to be arrived at fast.
    """
    gates = sum(1 for name in picks if GATE_TAG in library[name].tags)
    return 2 * len(picks) + 1 + gates


def _pick_joiners(rng, library, joiners, count):
    """``count`` joiners, cheap ones first, never the same one twice running.

    Sorted by cost after choosing rather than while: choosing by cost would
    give every board the same order, and the point of the ordering is the ramp,
    not which fragments are in it.
    """
    if not joiners:
        return []
    chosen = []
    while len(chosen) < count:
        offered = [name for name in joiners
                   if not chosen or name != chosen[-1]] or list(joiners)
        # Prefer one whose kind is not already on the board, so a long board
        # asks more than one sort of question.
        asked = {tag for name in chosen for tag in library[name].tags}
        fresh = [name for name in offered
                 if set(library[name].tags) - asked - {PLACE_TAG}]
        chosen.append(rng.choice(fresh or offered))
    chosen.sort(key=lambda name: library[name].cost)
    return _unrepeat(chosen)


def _give_the_gates_a_run_up(library, line):
    """Deal a run-up in front of every gate that has not got one.

    The run-up goes *immediately* before the gate, displacing nothing: a place
    between the two would spend the speed the run-up exists to give.  A gate
    that already follows a run-up is left as it is, and if the library holds no
    run-up at all the line comes back unchanged -- a generator that refused to
    compose because one fragment was missing would be no use to anybody adding
    fragments one at a time.
    """
    run_ups = sorted(name for name, entry in library.items()
                     if RUN_UP_TAG in entry.tags)
    if not run_ups:
        return line
    out = []
    for index, (chapter, fragment) in enumerate(line):
        if (GATE_TAG in library[fragment].tags
                and not (out and RUN_UP_TAG in library[out[-1][1]].tags)):
            out.append(('runup%d' % index, run_ups[index % len(run_ups)]))
        out.append((chapter, fragment))
    return out


def _unrepeat(names):
    """Break up any pair the cost sort put next to each other."""
    for index in range(1, len(names)):
        if names[index] == names[index - 1]:
            for later in range(index + 1, len(names)):
                if names[later] != names[index]:
                    names[index], names[later] = names[later], names[index]
                    break
    return names


def _unlike(rng, offered, line):
    """One of ``offered``, avoiding whatever went down last."""
    last = line[-1][1] if line else None
    choices = [name for name in offered if name != last] or list(offered)
    return rng.choice(choices)


# -- writing it out ---------------------------------------------------------

def _write(story, line, rng, themes):
    """Put the main line into the story, each chapter leading to the next."""
    library = fragments.library()
    for index, (name, fragment) in enumerate(line):
        following = line[index + 1][0] if index + 1 < len(line) else None
        story.chapters[name] = Chapter(
            id=name, fragment=fragment,
            variant=rng.choice(library[fragment].variants),
            theme=rng.choice(themes) if themes else None,
            exits={'ok': following} if following else {})


def _add_a_way_round(story, line, library, rng, themes, places, joiners):
    """Wire one failure exit to a slower way that rejoins the main line.

    Only where the board has a joiner with somewhere to rejoin *after* it: a way
    round that led to the finish would be a shortcut, and one that led back to
    where it came from would be a loop a player could not leave.
    """
    candidates = [index for index, (_, fragment) in enumerate(line)
                  if library[fragment].rule and index + 2 < len(line)]
    if not candidates:
        return
    at = rng.choice(candidates)
    missed_id, rejoin_id = 'round%d' % at, line[at + 2][0]
    # Never a gate: a way round is the forgiving route, and one that asked to be
    # arrived at fast would be a second challenge for a player who has just
    # arrived slowly from getting the first one wrong.
    offered = [name for name in (joiners or places)
               if GATE_TAG not in library[name].tags] or places
    detour = rng.choice(offered)
    story.chapters[missed_id] = Chapter(
        id=missed_id, fragment=detour,
        variant=rng.choice(library[detour].variants),
        theme=rng.choice(themes) if themes else None,
        exits={'ok': rejoin_id})
    story.chapters[line[at][0]].exits['missed'] = missed_id
