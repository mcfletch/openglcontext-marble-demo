"""Procedural level generator: a seed becomes a track.

The generator carves a **connected path** of cells from a start to a finish with a
seeded random walk, biased to make forward progress while wandering left/right, and
steps the surface height downward within a slope budget so the track has the
descending, terraced Marble-Madness feel.  Because each new cell is placed adjacent
to the previous one, the path is connected by construction — there is always a way
from start to finish (asserted in the tests).

Everything derives from ``random.Random((seed, difficulty))``, so a ``(seed,
difficulty)`` pair is a shareable, reproducible level.  This is intentionally the
simplest generator that gives varied, solvable tracks (grid random-walk, per the
plan's resolved decision); richer shaping (branches, gaps, decoration) layers on in
later phases.
"""
import random

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

# Largest single-step height change between adjacent path cells.
MAX_STEP = 1.4

# Candidate steps for each new cell: mostly flat (so the track is terraced into flat,
# grouted runs) with the occasional bigger drop, which then gets a connecting ramp.
_HEIGHT_STEPS = (0.0, 0.0, 0.0, 0.0, -0.8, -1.4)

# Grid moves as (dcol, drow).  "Forward" is +row (away from the camera/start).
_FORWARD = (0, 1)
_LEFT = (-1, 0)
_RIGHT = (1, 0)


def generate(seed=0, difficulty=1, cell_size=CELL_SIZE):
    """Return a reproducible :class:`~openglcontext_marble_demo.level.Level`."""
    # Fold both knobs into one integer seed so (seed, difficulty) is reproducible.
    rng = random.Random(seed * 1000 + difficulty)
    target_length = 10 + 5 * difficulty

    cells, path = _carve_path(rng, target_length)
    start_cell = path[0]
    finish_cell = path[-1]

    # Connect height offsets between consecutive cells with ramps, then decorate the
    # remaining (flat) cells with boost ramps / mechanisms.
    transitions = _transition_ramps(cells, path)
    features = list(transitions.values())
    features += _decorate(rng, cells, path, difficulty, skip=set(transitions))
    features.append(Finish(finish_cell))
    cell_surfaces = _surface_patches(rng, path, difficulty)

    # Roughly a second and a bit of budget per cell, scaled so easy levels are
    # forgiving; enough to reward a clean line without being trivial.
    time_limit = round(len(cells) * (2.4 - 0.15 * difficulty) + 15.0, 1)

    return Level(
        name=f"seed-{seed}-d{difficulty}",
        cells=cells,
        start_cell=start_cell,
        finish_cell=finish_cell,
        time_limit=time_limit,
        features=features,
        cell_surfaces=cell_surfaces,
        cell_size=cell_size,
        seed=seed,
        difficulty=difficulty,
    )


# Surfaces used for the occasional patch, and how likely a patch is to start.
_PATCH_SURFACES = ("ice_sheet", "metal", "rubber_pad")


def _surface_patches(rng, path, difficulty):
    """Assign short runs of a different surface along the path, for variety.

    Interior cells (never the start/finish) occasionally begin a 2–4 cell patch of
    ice/metal/rubber, which changes both the look and the grip.  Deterministic from
    the same RNG stream, so a seed reproduces the whole level.
    """
    surfaces = {}
    patch_chance = 0.12 + 0.03 * difficulty
    i = 2
    while i < len(path) - 3:
        if rng.random() < patch_chance:
            surface = rng.choice(_PATCH_SURFACES)
            length = rng.randint(2, 4)
            for cell in path[i:i + length]:
                surfaces[cell] = surface
            i += length + 1
        else:
            i += 1
    return surfaces


def _transition_ramps(cells, path):
    """A ramp on every interior cell whose next cell is at a different height.

    Rolling from one level to the next, the marble takes a real slope instead of a
    hard step: the tile is tilted so its far edge (toward the next cell) meets that
    cell's height.  A down-step becomes a ramp down, an up-step a ramp up.  The
    tilted box is both the render surface and the physics collider, so the marble
    physically rolls the slope.
    """
    ramps = {}
    for i in range(1, len(path) - 1):        # never the start or the finish cell
        cell, nxt = path[i], path[i + 1]
        rise = cells[nxt] - cells[cell]
        if abs(rise) > 0.05:
            ramps[cell] = Ramp(cell=cell, direction=_step(cell, nxt),
                               rise=rise, boost_speed=5.0, launch=False)
    return ramps


def _decorate(rng, cells, path, difficulty, skip=frozenset()):
    """Sprinkle speed ramps (some launch ramps) on straights, plus a few rails.

    Ramps sit only on straight *flat* runs (in-direction == out-direction, and not
    already a transition-ramp cell) so they point where the marble is going, spaced
    out with a cooldown.  Because every cell stays in ``cells``, the track remains
    contiguous and solvable — a launch ramp just sends the marble over the next tiles
    in an arc rather than a gap.
    """
    features = []
    ramp_chance = 0.14 + 0.03 * difficulty
    mech_chance = 0.08 + 0.03 * difficulty
    cooldown = 0
    for i in range(1, len(path) - 2):
        cell = path[i]
        if cell in skip:
            continue
        in_dir = _step(path[i - 1], path[i])
        out_dir = _step(path[i], path[i + 1])
        straight = in_dir == out_dir
        if cooldown == 0 and straight and rng.random() < ramp_chance:
            launch = rng.random() < 0.30
            features.append(Ramp(cell=cell, direction=out_dir, launch=launch,
                                 rise=1.0 if launch else 0.45,
                                 boost_speed=7.0 if launch else 9.0))
            cooldown = 3
        elif cooldown == 0 and rng.random() < mech_chance:
            features.append(_mechanism(rng, cell, out_dir, difficulty))
            cooldown = 2
        elif rng.random() < 0.06:
            side = _void_side(cell, cells, rng)
            if side is not None:
                features.append(Wall(cell=cell, side=side))
        cooldown = max(0, cooldown - 1)
    return features


def _mechanism(rng, cell, out_dir, difficulty):
    """Pick a mechanism for ``cell``, weighted toward the gentler ones."""
    kind = rng.choices(
        ("bumper", "spring", "arm", "elevator"), weights=(3, 2, 2, 1))[0]
    if kind == "bumper":
        return Bumper(cell=cell)
    if kind == "spring":
        wdir = _world_direction(out_dir)
        impulse = (wdir[0] * 4.0, 8.0, wdir[2] * 4.0)   # a forward-and-up hop
        return SpringTrap(cell=cell, impulse=impulse)
    if kind == "arm":
        return RotatingArm(cell=cell, rpm=15 + 5 * difficulty)
    return Elevator(cell=cell, travel=2.0 + 0.3 * difficulty, period=3.0)


def _step(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _void_side(cell, cells, rng):
    """A cardinal side of ``cell`` that faces the void, or ``None`` if fully walled in."""
    col, row = cell
    sides = [d for d in _SIDE_OF if (col + d[0], row + d[1]) not in cells]
    return _SIDE_OF[rng.choice(sides)] if sides else None


def _carve_path(rng, target_length):
    """Random-walk a connected, mostly-forward, descending path of cells."""
    col, row, height = 0, 0, 0.0
    cells = {(col, row): height}
    path = [(col, row)]
    last_side = None

    attempts = 0
    max_attempts = target_length * 20
    while len(path) < target_length and attempts < max_attempts:
        attempts += 1
        dcol, drow = _choose_step(rng, last_side)
        nxt = (col + dcol, row + drow)
        if nxt in cells:                       # don't cross ourselves; try again
            continue
        height = max(height + rng.choice(_HEIGHT_STEPS), -difficulty_floor(target_length))
        cells[nxt] = height
        path.append(nxt)
        col, row = nxt
        last_side = (dcol, drow) if (dcol, drow) in (_LEFT, _RIGHT) else None
    return cells, path


def _choose_step(rng, last_side):
    """Pick the next grid move: usually forward, sometimes a turn.

    Forward is weighted twice as heavily as either side so the track advances but
    still winds.  We never step backward (would fold the path onto itself), and we
    avoid reversing the previous sideways move (no immediate zig-zag jitter).
    """
    choices = [_FORWARD, _FORWARD, _LEFT, _RIGHT]
    if last_side == _LEFT:
        choices = [c for c in choices if c != _RIGHT]
    elif last_side == _RIGHT:
        choices = [c for c in choices if c != _LEFT]
    return rng.choice(choices)


def difficulty_floor(target_length):
    """Lowest the track is allowed to descend to — scales with its length."""
    return target_length * MAX_STEP
