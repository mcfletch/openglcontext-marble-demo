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
import dataclasses
import json
import typing
from typing import Any

from OpenGLContext import atomicfiles
from OpenGLContext.loaders.documentvalues import (
    DocumentError,
    JSONObject,
    parse_object,
    require_array,
    require_number,
    require_object,
    require_text,
    require_whole,
)
from OpenGLContext.loaders.resolver import contained_source

from .level import Bumper, Elevator, Finish, Gate, Level, Ramp, RotatingArm, SpringTrap, Wall

__all__ = ['VERSION', 'GENERATOR', 'FEATURES', 'SUFFIX', 'register_feature',
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


def register_feature(kind: str, cls: type) -> None:
    """Save and load the dataclass ``cls`` under the name ``kind``."""
    FEATURES[kind] = cls
    _KIND_OF[cls] = kind

#: Fields of :class:`~openglcontext_marble_demo.level.Level` that are written
#: as they stand.  ``cells``, ``cell_surfaces`` and ``features`` are not: each
#: is keyed or typed in a way JSON has no spelling for.
_PLAIN = ('name', 'time_limit', 'cell_size', 'kill_y', 'respawn_delay',
          'surface', 'seed', 'difficulty')


def _seed(value: object, what: str) -> int | None:
    return None if value is None else require_whole(value, what)


#: How each of :data:`_PLAIN` is read, since each is the type Level declares.
_READ: dict[str, Any] = {
    'name': require_text, 'time_limit': require_number, 'cell_size': require_number,
    'kill_y': require_number, 'respawn_delay': require_number,
    'surface': require_text, 'seed': _seed, 'difficulty': require_whole,
}


def _cell(value: object, what: str) -> tuple[int, int]:
    """A ``(col, row)`` pair from a JSON list, as integers."""
    pair = require_array(value, what)
    if len(pair) != 2:
        raise DocumentError('%s is %r, which is not a column and a row' % (what, value))
    return (require_whole(pair[0], '%s column' % (what,)),
            require_whole(pair[1], '%s row' % (what,)))


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


def from_json(document: JSONObject) -> Any:
    """A :class:`~openglcontext_marble_demo.level.Level` from ``document``.

    Raises :exc:`ValueError` for a file from a newer writer or with a version
    that is not a whole number, for a mechanism this game has no class for,
    and for a document that is not a level at all —
    in every case rather than returning something partly built, because a level
    silently missing its ramps is worse than one that refuses to open. A value
    of the wrong kind is a :class:`~OpenGLContext.loaders.documentvalues.DocumentError`
    (a ``ValueError``) naming it.
    """
    version = document.get('version', 0)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError('not a marble board: its file version is %r, not a whole '
                         'number' % (version,))
    if version > VERSION:
        raise ValueError(
            'this board was saved by a newer marble (file version %d, this one '
            'reads %d)' % (version, VERSION))
    if 'cells' not in document or 'start_cell' not in document:
        raise ValueError('not a marble board: no cells in it')

    cells = {}
    surfaces = {}
    for raw in require_array(document['cells'], 'cells'):
        entry = require_array(raw, 'a cell')
        if len(entry) < 3:
            raise DocumentError('a cell is %r, which is not a column, a row and a '
                                'height' % (raw,))
        where = (require_whole(entry[0], 'cell column'), require_whole(entry[1], 'cell row'))
        cells[where] = require_number(entry[2], 'cell height')
        if len(entry) > 3 and entry[3]:
            surfaces[where] = require_text(entry[3], 'cell surface')

    plain = {name: _READ[name](document[name], name)
             for name in _PLAIN if name in document}
    return Level(
        cells=cells,
        cell_surfaces=surfaces,
        start_cell=_cell(document['start_cell'], 'start_cell'),
        finish_cell=_cell(document.get('finish_cell'), 'finish_cell'),
        features=[_feature_from_json(require_object(entry, 'a feature'))
                  for entry in require_array(document.get('features', ()), 'features')],
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
    from . import mechanisms  # noqa: PLC0415 mechanisms imports levelfile to register into it
    mechanisms.registry()


def _feature_from_json(entry: JSONObject) -> Any:
    kind = require_text(entry.get('kind'), 'feature kind')
    if kind not in FEATURES:
        _load_mechanisms()
    factory = FEATURES.get(kind)
    if factory is None:
        raise ValueError('this board wants a %r, which this game has no '
                         'mechanism for' % (kind,))
    hints = typing.get_type_hints(factory)
    named: dict[str, object] = {}
    for field in dataclasses.fields(factory):
        if field.name not in entry:
            continue                       # absent: the dataclass default holds
        value = entry[field.name]
        if typing.get_origin(hints.get(field.name)) is tuple:
            value = tuple(require_array(value, '%s %s' % (kind, field.name)))
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

    Written whole or not at all (:func:`OpenGLContext.atomicfiles.write_text`),
    because a board file is the only copy of what a designer drew: a write that
    truncated the file first would leave nothing at all if the disk filled or
    the machine went down half way through.
    """
    return atomicfiles.write_text(path, dumps(to_json(level)))


def load(path: str) -> Any:
    """Read a level from ``path``."""
    with open(contained_source(path), 'rb') as handle:
        return from_json(parse_object(handle.read(), path))
