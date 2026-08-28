"""Procedural boards: a seed becomes somewhere to play.

The *shape* of a board — where the route goes, how wide it is, which ways round
there are, how it terraces down — is :mod:`~openglcontext_marble_demo.boards`.
This module decorates that shape and turns it into a
:class:`~openglcontext_marble_demo.level.Level`: ramps, rails, mechanisms,
surface patches, a finish and a clock.

Decoration is done **by strand**, which is what makes the choice of route a
choice worth making:

* the **shortcut** strands — the ways round that save cells — are where the
  hazards go, so the quick line is the one that asks something of the player;
* the **scenic** strands and the spine stay largely clear, so there is always a
  way through for a player who would rather arrive than gamble;
* and every hazard placed is checked against the board: a hazard that would
  leave no clean route from start to finish is not placed at all. Barriers
  block some passages, never all of them.

Everything derives from ``random.Random`` seeded from ``(seed, difficulty)``, so
a pair is a shareable, reproducible board.

    >>> board = generate(seed=3, difficulty=2)
    >>> board.start_cell in board.cells and board.finish_cell in board.cells
    True
"""
import random

from . import boards
from .level import (
    CELL_SIZE,
    Bumper,
    Elevator,
    Finish,
    Level,
    Ramp,
    RotatingArm,
    SpringTrap,
    Wall,
    _world_direction,
)

_SIDE_OF = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}

#: Largest single-step height change between adjacent cells.  The terracing
#: never spends more than this, so the board is always rollable.
MAX_STEP = boards.TERRACE_STEP

#: How long a board's spine is, and how often it opens into a plaza, per
#: difficulty.  Boards get *shorter* as they get harder rather than longer: what
#: makes a board hard is the decisions in it, and a long easy board is only a
#: long one.  What tightens is the room -- plazas thin out.
SPINE_LENGTH = (16, 15, 14, 13, 12)
PLAZA_EVERY = (3, 4, 5, 6, 7)

#: Seconds of clock per cell of the route, plus a constant.  A run is meant to
#: be short enough that another go is cheap.
SECONDS_PER_CELL = 1.15
SECONDS_SPARE = 8.0

#: Mechanisms that a marble cannot simply roll across.  What makes them fair is
#: that one is never placed where it would leave no clean way through.
_HAZARDS = (Bumper, SpringTrap, RotatingArm)

#: Surfaces used for the occasional patch.
_PATCH_SURFACES = ("ice_sheet", "metal", "rubber_pad")


def _tier(difficulty, ladder):
    return ladder[max(0, min(int(difficulty) - 1, len(ladder) - 1))]


def generate(seed=0, difficulty=1, cell_size=CELL_SIZE):
    """Return a reproducible :class:`~openglcontext_marble_demo.level.Level`."""
    rng = random.Random(seed * 1000 + difficulty)
    board = boards.build(rng, length=_tier(difficulty, SPINE_LENGTH),
                         plaza_every=_tier(difficulty, PLAZA_EVERY),
                         difficulty=difficulty)

    features = _decorate(rng, board, difficulty)
    features.append(Finish(board.finish))
    surfaces = _surface_patches(rng, board, difficulty)

    # The clock is set by how far the marble has to travel, not by how many
    # tiles were laid: a wide board is not a longer one.
    route = boards.distances(board.cells, board.start)[board.finish]
    time_limit = round(route * SECONDS_PER_CELL + SECONDS_SPARE, 1)

    return Level(
        name=f"seed-{seed}-d{difficulty}",
        cells=dict(board.heights),
        start_cell=board.start,
        finish_cell=board.finish,
        time_limit=time_limit,
        features=features,
        cell_surfaces=surfaces,
        cell_size=cell_size,
        seed=seed,
        difficulty=difficulty,
    )


# -- decoration ----------------------------------------------------------

def _decorate(rng, board, difficulty):
    """Ramps and mechanisms, placed strand by strand and checked as they go."""
    features = []
    blocked = set()
    spared = {board.start, board.finish}
    # The cells next to the start and the finish are spared too: a hazard the
    # marble meets before it is moving, or on the pad, is not a decision.
    for cell in (board.start, board.finish):
        spared.update((cell[0] + dc, cell[1] + dr)
                      for dc, dr in boards.NEIGHBOURS)

    for strand in board.strands:
        density = _density(strand, difficulty)
        cooldown = 0
        for index in range(1, len(strand.cells) - 1):
            cell = strand.cells[index]
            if cell in spared or cell not in board.cells:
                continue
            heading = _step(cell, strand.cells[index + 1])
            if cooldown:
                cooldown -= 1
            elif rng.random() < density.hazard:
                mechanism = _mechanism(rng, cell, heading, difficulty)
                if _keeps_a_clean_line(board, blocked, cell):
                    blocked.add(cell)
                    features.append(mechanism)
                    cooldown = _breathing_room(difficulty)
            elif rng.random() < density.ramp:
                features.append(_ramp(rng, board, cell, heading, difficulty))
                cooldown = 2
        features.extend(_rails(rng, board, strand))
    return features


class _Density:
    """How thickly a strand is decorated: hazards, and speed ramps."""

    def __init__(self, hazard, ramp):
        self.hazard = hazard
        self.ramp = ramp


def _density(strand, difficulty):
    """A shortcut earns its saving with hazards; everything else stays passable.

    This is the whole risk/reward statement, and it is one line: the strand that
    saves cells is the strand that costs something to take.
    """
    if strand.kind == boards.SHORTCUT:
        return _Density(hazard=0.28 + 0.08 * difficulty, ramp=0.20)
    if strand.kind == boards.SCENIC:
        return _Density(hazard=0.0, ramp=0.16)
    return _Density(hazard=0.02 + 0.10 * difficulty, ramp=0.12 + 0.02 * difficulty)


def _breathing_room(difficulty):
    """Cells left alone after a hazard, so an easy board is not a gauntlet."""
    return max(1, 4 - difficulty // 2)


def _keeps_a_clean_line(board, blocked, cell):
    """Whether blocking ``cell`` too still leaves a hazard-free way through."""
    return board.route_exists(blocked | {cell})


def _ramp(rng, board, cell, heading, difficulty):
    """A speed ramp, or a launch ramp where there is board ahead to land on."""
    ahead = (cell[0] + heading[0] * 2, cell[1] + heading[1] * 2)
    launch = ahead in board.cells and rng.random() < 0.25 + 0.03 * difficulty
    return Ramp(cell=cell, direction=heading, launch=launch,
                rise=1.0 if launch else 0.45,
                boost_speed=7.0 if launch else 9.0)


def _mechanism(rng, cell, heading, difficulty):
    """Pick a mechanism, weighted toward the ones there is room to dodge."""
    kind = rng.choices(("bumper", "spring", "arm", "elevator"),
                       weights=(3, 2, 2, 1))[0]
    if kind == "bumper":
        return Bumper(cell=cell)
    if kind == "spring":
        wdir = _world_direction(heading)
        return SpringTrap(cell=cell,
                          impulse=(wdir[0] * 4.0, 8.0, wdir[2] * 4.0))
    if kind == "arm":
        return RotatingArm(cell=cell, rpm=15 + 5 * difficulty)
    return Elevator(cell=cell, travel=2.0 + 0.3 * difficulty, period=3.0)


def _rails(rng, board, strand):
    """Low walls along the odd cell edge that faces the void.

    A rail is not an obstacle: it faces outward, so it keeps a marble on the
    board rather than standing in its way — which is why these are placed
    without asking whether a clean line survives.
    """
    rails = []
    for cell in strand.cells:
        if rng.random() >= 0.12:
            continue
        side = _void_side(cell, board.cells, rng)
        if side is not None:
            rails.append(Wall(cell=cell, side=side))
    return rails


def _surface_patches(rng, board, difficulty):
    """Short runs of ice/metal/rubber along the strands, for grip and for looks."""
    surfaces = {}
    chance = 0.10 + 0.03 * difficulty
    for strand in board.strands:
        index = 2
        while index < len(strand.cells) - 3:
            if rng.random() < chance:
                surface = rng.choice(_PATCH_SURFACES)
                length = rng.randint(2, 4)
                for cell in strand.cells[index:index + length]:
                    if cell not in (board.start, board.finish):
                        surfaces[cell] = surface
                index += length + 1
            else:
                index += 1
    return surfaces


def _step(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _void_side(cell, cells, rng):
    """A cardinal side of ``cell`` facing the void, or ``None`` if fully enclosed."""
    col, row = cell
    sides = [d for d in _SIDE_OF if (col + d[0], row + d[1]) not in cells]
    return _SIDE_OF[rng.choice(sides)] if sides else None
