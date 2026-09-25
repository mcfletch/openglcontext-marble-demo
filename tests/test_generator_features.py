"""The generator decorates tracks with ramps/rails — deterministically and safely."""
from openglcontext_marble_demo import generator
from openglcontext_marble_demo.game import PLAYING, MarbleGame
from openglcontext_marble_demo.level import (
    Bumper,
    Elevator,
    Finish,
    Ramp,
    RotatingArm,
    SpringTrap,
    Wall,
)

_MECHANISMS = (Bumper, SpringTrap, Elevator, RotatingArm)


def _kinds(level):
    return [type(f).__name__ for f in level.features]


def test_generated_levels_have_a_finish_last():
    level = generator.generate(seed=4, difficulty=3)
    assert isinstance(level.features[-1], Finish)


def test_higher_difficulty_tracks_get_some_ramps():
    # Across a spread of seeds a hard difficulty should place at least some ramps.
    ramps = sum(sum(isinstance(f, Ramp) for f in generator.generate(s, 4).features)
                for s in range(10))
    assert ramps > 0


def test_decoration_is_deterministic():
    a = generator.generate(seed=8, difficulty=3)
    b = generator.generate(seed=8, difficulty=3)
    assert _kinds(a) == _kinds(b)
    # Ramp cells and directions match too.
    a_ramps = [(f.cell, f.direction, f.launch) for f in a.features if isinstance(f, Ramp)]
    b_ramps = [(f.cell, f.direction, f.launch) for f in b.features if isinstance(f, Ramp)]
    assert a_ramps == b_ramps


def test_generated_levels_include_mechanisms():
    mechanisms = sum(
        sum(isinstance(f, _MECHANISMS) for f in generator.generate(s, 4).features)
        for s in range(12))
    assert mechanisms > 0


def test_generated_level_builds_and_is_playable_headlessly():
    """A generated level with mechanisms builds into a game and advances cleanly."""
    level = generator.generate(seed=4, difficulty=4)
    game = MarbleGame(level)
    for _ in range(120):
        game.advance(1 / 60.0)
    assert game.state in (PLAYING, "won", "lost")   # no crash; animators/effects ran


def test_ramps_sit_on_track_cells():
    level = generator.generate(seed=2, difficulty=4)
    for feature in level.features:
        if isinstance(feature, Ramp):
            assert feature.cell in level.cells


def test_rails_face_the_void():
    level = generator.generate(seed=6, difficulty=4)
    for feature in level.features:
        if isinstance(feature, Wall):
            col, row = feature.cell
            dx, dz = Wall.OFFSET[feature.side]
            assert (col + dx, row + dz) not in level.cells   # rail faces empty space
