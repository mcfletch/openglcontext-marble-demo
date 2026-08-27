"""Marble Madness re-interpretation for OpenGLContext.

An aerial-view, time-trial marble game built on the OpenGLContext PBR/IBL renderer
and the ``omi_physics`` rigid-body engine.  The player rolls a reflective marble
through a **procedurally generated** track, steering by imparting spin with the
arrow keys, and races a countdown to the finish.

This package holds the *game*: the material table, the level pieces and the
procedural generator that assembles them, the marble controller (steering,
fall/respawn), the HUD, and the run loop.  The reusable simulation and rendering
live in ``OpenGLContext`` and ``omi_physics`` — see
``plans/MARBLE-MADNESS-DEMO.md`` in the engine repo for the original design, and
``plans/STANDALONE-GAME.md`` here for where it is going.

Run it with the ``oglc-marble`` console script (``--help`` lists the knobs), or
``python -m openglcontext_marble_demo``.
"""

__version__ = "0.1.0"
