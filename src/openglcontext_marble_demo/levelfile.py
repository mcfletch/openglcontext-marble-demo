"""Levels as files: what a designer saves, and what the game reads back.

A generated level is a seed; an authored one is a file. The format is JSON,
small enough that a person can read it and fix it by hand, and versioned so a
game that meets a newer file says so rather than half-loading a level it does
not understand.

A board is laid out **a cell to a line and a mechanism to a line**, because the
promise that a person can read it is only kept if they can: a hundred cells
spread over five lines each is five hundred lines of column, and nobody reads
that or corrects a height in it.

Two things make it cheap to extend. A **feature writes itself**: every mechanism
in :mod:`~openglcontext_marble_demo.level` is a dataclass, so its fields are the
document, and adding a mechanism means adding one name to :data:`FEATURES`
(which a test holds you to). And the **cells are a list, not an object**: JSON
keys are strings, and a level's keys are pairs of integers that would have to be
spelled and re-parsed.

    >>> from openglcontext_marble_demo import generator
    >>> board = generator.generate(seed=3, difficulty=2)
    >>> from_json(to_json(board)).cells == board.cells
    True
"""
import contextlib
import dataclasses
import json
import os
import tempfile
import typing
from typing import Any

from .level import Bumper, Elevator, Finish, Gate, Level, Ramp, RotatingArm, SpringTrap, Wall

__all__ = ['VERSION', 'GENERATOR', 'FEATURES', 'SUFFIX',
           'to_json', 'from_json', 'dumps', 'save', 'load']

#: The format's version.  Raise it when a change would stop an older reader
#: understanding a file; :func:`from_json` refuses anything above what it knows.
VERSION = 1

#: What wrote the file, so a stray document is identifiable at a glance.
GENERATOR = 'openglcontext-marble'

#: The conventional extension for a board.
SUFFIX = '.marble'

#: The name each mechanism is written under.  A mechanism missing from here
#: cannot be saved, which ``test_levelfile`` asserts against the module rather
#: than trusting anyone to remember.
FEATURES = {
    'finish': Finish,
    'gate': Gate,
    'ramp': Ramp,
    'wall': Wall,
    'bumper': Bumper,
    'spring': SpringTrap,
    'elevator': Elevator,
    'arm': RotatingArm,
}
_KIND_OF = {cls: kind for kind, cls in FEATURES.items()}

#: Fields of :class:`~openglcontext_marble_demo.level.Level` that are written
#: as they stand.  ``cells``, ``cell_surfaces`` and ``features`` are not: each
#: is keyed or typed in a way JSON has no spelling for.
_PLAIN = ('name', 'time_limit', 'cell_size', 'kill_y', 'respawn_delay',
          'surface', 'seed', 'difficulty')


def _cell(value: Any) -> Any:
    """A ``(col, row)`` pair from a JSON list, as integers."""
    col, row = value
    return (int(col), int(row))


def to_json(level: Any) -> Any:
    """``level`` as a JSON-safe document."""
    return {
        'generator': GENERATOR,
        'version': VERSION,
        **{name: getattr(level, name) for name in _PLAIN},
        'start_cell': list(level.start_cell),
        'finish_cell': list(level.finish_cell),
        # [col, row, height] per cell, with the surface only where one is set,
        # so the common case reads as three numbers on a line.
        'cells': [[col, row, height, level.cell_surfaces[(col, row)]]
                  if (col, row) in level.cell_surfaces else [col, row, height]
                  for (col, row), height in sorted(level.cells.items())],
        'features': [_feature_to_json(feature) for feature in level.features],
    }


def from_json(document: Any) -> Any:
    """A :class:`~openglcontext_marble_demo.level.Level` from ``document``.

    Raises :exc:`ValueError` for a file from a newer writer, for a mechanism
    this game has no class for, and for a document that is not a level at all —
    in every case rather than returning something partly built, because a level
    silently missing its ramps is worse than one that refuses to open.
    """
    version = int(document.get('version', 0))
    if version > VERSION:
        raise ValueError(
            'this board was saved by a newer marble (file version %d, this one '
            'reads %d)' % (version, VERSION))
    if 'cells' not in document or 'start_cell' not in document:
        raise ValueError('not a marble board: no cells in it')

    cells = {}
    surfaces = {}
    for entry in document['cells']:
        col, row, height = entry[0], entry[1], entry[2]
        cells[(int(col), int(row))] = float(height)
        if len(entry) > 3 and entry[3]:
            surfaces[(int(col), int(row))] = str(entry[3])

    plain = {name: document[name] for name in _PLAIN if name in document}
    return Level(
        cells=cells,
        cell_surfaces=surfaces,
        start_cell=_cell(document['start_cell']),
        finish_cell=_cell(document['finish_cell']),
        features=[_feature_from_json(entry)
                  for entry in document.get('features', ())],
        **plain)


# -- features ------------------------------------------------------------

def _feature_to_json(feature: Any) -> Any:
    kind = _KIND_OF.get(type(feature))
    if kind is None:
        raise ValueError('%s has no name in levelfile.FEATURES, so it cannot '
                         'be saved' % type(feature).__name__)
    document: dict[str, Any] = {'kind': kind}
    for field in dataclasses.fields(feature):
        value = getattr(feature, field.name)
        document[field.name] = list(value) if isinstance(value, tuple) else value
    return document


def _load_mechanisms() -> None:
    """Import the mechanisms package, which is what registers what it holds.

    Registering happens on import, and a process that only reads a board file
    has imported nothing: without this, loading a board with a peg board on it
    says the game has no such mechanism, in a game that does.  Imported here
    rather than at the top of the module because ``mechanisms`` imports this one
    to register into it.
    """
    from . import mechanisms
    mechanisms.registry()


def _feature_from_json(entry: Any) -> Any:
    kind = entry.get('kind')
    if kind not in FEATURES:
        _load_mechanisms()
    factory = FEATURES.get(kind)
    if factory is None:
        raise ValueError('this board wants a %r, which this game has no '
                         'mechanism for' % (kind,))
    hints = typing.get_type_hints(factory)
    named = {}
    for field in dataclasses.fields(factory):
        if field.name not in entry:
            continue                       # absent: the dataclass default holds
        value = entry[field.name]
        if typing.get_origin(hints.get(field.name)) is tuple:
            value = tuple(value)
        named[field.name] = value
    return factory(**named)


# -- files ---------------------------------------------------------------

#: The keys written before the two long lists, in the order they read best.
_HEAD = ('generator', 'version', 'name', 'time_limit', 'cell_size', 'kill_y',
         'respawn_delay', 'surface', 'seed', 'difficulty', 'start_cell',
         'finish_cell')


def dumps(document: Any) -> Any:
    """``document`` as text, with each cell and each mechanism on one line.

    Assembled rather than handed to ``json.dumps(indent=2)``, which puts every
    number of every cell on a line of its own.  Every piece still goes through
    ``json.dumps``, so the escaping and the number formatting are the library's
    rather than this module's guesses about them.
    """
    def one(value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False)

    lines = ['{']
    for key in _HEAD:
        if key in document:
            lines.append('  %s: %s,' % (one(key), one(document[key])))
    for key in ('cells', 'features'):
        entries = document.get(key, [])
        if not entries:
            lines.append('  %s: [],' % one(key))
            continue
        lines.append('  %s: [' % one(key))
        lines.extend('    %s,' % one(entry) for entry in entries)
        lines[-1] = lines[-1][:-1]              # no comma after the last entry
        lines.append('  ],')
    lines[-1] = lines[-1][:-1]                  # nor after the last key
    lines.append('}')
    return '\n'.join(lines) + '\n'


def save(level: Any, path: str) -> Any:
    """Write ``level`` to ``path``; return the path.

    Written beside the target and moved onto it, because a board file is the
    only copy of what a designer drew: a write that truncated the file first
    would leave nothing at all if the disk filled or the machine went down half
    way through.  ``os.replace`` is atomic on every platform this runs on, so
    the file is either the old board or the new one.
    """
    document = dumps(to_json(level))
    beside = os.path.dirname(os.path.abspath(path))
    handle, temporary = tempfile.mkstemp(dir=beside, suffix='.marble-new')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as writing:
            writing.write(document)
        os.replace(temporary, path)
    except BaseException:
        # Nothing half-written left beside the designer's file, whatever went
        # wrong -- including an interrupt.
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return path


def load(path: str) -> Any:
    """Read a level from ``path``."""
    with open(path, encoding='utf-8') as handle:
        return from_json(json.load(handle))
