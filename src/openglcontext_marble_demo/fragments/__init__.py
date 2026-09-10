"""The story library: one chapter of a board per module, discovered not listed.

A fragment is one chapter — a piece with an entry, one or more exits, a rule it
imposes and a set of variants — and it lives in **its own module in this
directory**, registering itself with :func:`fragment`.

That arrangement is a merge property before it is anything else. Two fragments
never touch the same file, so two people building fragments never conflict; and
because the library is found by scanning this directory rather than written down
somewhere, adding one edits nothing that anybody else is editing.

    >>> import random
    >>> from openglcontext_marble_demo.pieces import Port
    >>> name = sorted(library())[0]
    >>> piece = build(name, random.Random(1), Port(cell=(0, 0)))
    >>> 'ok' in piece.exits
    True

A fragment declares:

``tags``
    What kind of thing it is, so a generator can ask for "something about speed"
    rather than for a name.  ``speed``, ``aim``, ``brake``, ``luck``, ``place``,
    ``gate``.
``variants``
    Named bundles of material, layout and effect.  A fragment with one variant
    reads the same way every time it appears, so every fragment offers more.
``rule``
    The one line it asks of a player, empty for a place.
"""
import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..pieces import DESIGN_TILT

__all__ = ['Entry', 'fragment', 'library', 'build', 'tagged', 'TAGS',
           'DESIGN_TILT']

#: The kinds a generator chooses between.  A fragment may carry several.
TAGS = ('place', 'speed', 'brake', 'aim', 'luck', 'gate', 'hazard')

_LIBRARY: dict = {}
_LOADED = False


@dataclass(frozen=True)
class Entry:
    """One fragment, and what a generator needs to know to choose it."""
    name: str
    build: Callable[..., Any]
    tags: tuple = ()
    rule: str = ''
    variants: tuple = ('plain',)
    #: What it costs a player who gets it wrong, in seconds of the clock, as a
    #: rough figure a generator can balance a board with.
    cost: float = 0.0


def fragment(name: str, tags: Any=('place',), rule: Any='', variants: Any=('plain',), cost: Any=0.0) -> Any:
    """Register the decorated builder as a fragment of the library.

    The builder is ``f(rng, entry, variant=None, **named) -> Piece``, and it must
    put its cells down relative to ``entry`` and answer with a piece whose
    ``exits`` has at least an ``'ok'``.
    """
    def register(builder: Any) -> Any:
        _LIBRARY[name] = Entry(name=name, build=builder, tags=tuple(tags),
                               rule=rule, variants=tuple(variants), cost=cost)
        return builder
    return register


def _load() -> None:
    """Import every module beside this one, which is what registers them."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for found in pkgutil.iter_modules(__path__):
        if not found.name.startswith('_'):
            importlib.import_module('%s.%s' % (__name__, found.name))


def library() -> Any:
    """Every fragment, by name."""
    _load()
    return dict(_LIBRARY)


def tagged(*tags: Any) -> Any:
    """The fragments carrying any of ``tags`` — how a generator asks for a kind."""
    wanted = set(tags)
    return {name: entry for name, entry in library().items()
            if wanted & set(entry.tags)}


def build(name: str, rng: Any, entry: Any, variant: Any=None, **named: Any) -> Any:
    """Build fragment ``name`` at ``entry``; raise :exc:`KeyError` if unknown.

    An unknown variant is an error rather than a fallback to the plain one: a
    story that asked for the icy version and quietly got the ordinary one is a
    story nobody can debug.
    """
    found = library().get(name)
    if found is None:
        raise KeyError('no fragment called %r' % (name,))
    if variant is not None and variant not in found.variants:
        raise KeyError('fragment %r has no variant %r (it has %s)'
                       % (name, variant, ', '.join(found.variants)))
    return found.build(rng, entry, variant=variant or found.variants[0], **named)
