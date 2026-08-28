"""Mechanisms that are not part of ``level.py``: one per module, self-registering.

A mechanism is a dataclass with ``owned_cells()`` and ``build(scene, level,
index, result)``, exactly like the ones in
:mod:`~openglcontext_marble_demo.level`.  What this package adds is that a new
one is a **new file** rather than an edit to a file everyone else is editing —
:func:`mechanism` registers it, and registering also gives it the name the file
format saves it under, so a mechanism cannot exist and be unsaveable.

    >>> registry() is not None
    True
"""
import importlib
import pkgutil

__all__ = ['mechanism', 'registry']

_REGISTRY: dict = {}
_LOADED = False


def mechanism(name):
    """Register the decorated dataclass as a mechanism called ``name``.

    Registering is also what teaches
    :mod:`~openglcontext_marble_demo.levelfile` to read and write it, so there
    is one act rather than two and no way to do half of it.
    """
    def register(cls):
        from openglcontext_marble_demo import levelfile
        _REGISTRY[name] = cls
        levelfile.FEATURES[name] = cls
        levelfile._KIND_OF[cls] = name
        return cls
    return register


def _load():
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for found in pkgutil.iter_modules(__path__):
        if not found.name.startswith('_'):
            importlib.import_module('%s.%s' % (__name__, found.name))


def registry():
    """Every registered mechanism, by the name it saves under."""
    _load()
    return dict(_REGISTRY)
