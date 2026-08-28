"""Levels as files: what a designer saves, and what the game reads back.

A generated level is a seed. An *authored* one is a file, and this is its
format: JSON, small enough to read, and versioned so an older game says so
rather than half-loading a level it does not understand.

The rule the round-trip tests hold is that a level written and read back is the
same level — every cell, every height, every surface, and every field of every
feature — because anything that quietly does not survive the trip is work a
designer loses without being told.
"""
import json
import math

import pytest

from openglcontext_marble_demo import generator, levelfile
from openglcontext_marble_demo.level import (
    Bumper,
    Elevator,
    Finish,
    Level,
    Ramp,
    RotatingArm,
    SpringTrap,
    Wall,
)


def _level(**named):
    cells = {(0, 0): 0.0, (1, 0): -0.4, (1, 1): -0.4, (2, 1): -1.2}
    named.setdefault('name', 'hand-made')
    named.setdefault('cells', cells)
    named.setdefault('start_cell', (0, 0))
    named.setdefault('finish_cell', (2, 1))
    named.setdefault('time_limit', 33.5)
    named.setdefault('features', [Finish((2, 1))])
    return Level(**named)


def _round_trip(level):
    return levelfile.from_json(json.loads(json.dumps(levelfile.to_json(level))))


# -- the level itself ----------------------------------------------------------

def test_a_level_survives_the_round_trip():
    before = _level()
    after = _round_trip(before)
    assert after.name == before.name
    assert after.cells == before.cells
    assert after.start_cell == before.start_cell
    assert after.finish_cell == before.finish_cell
    assert after.time_limit == before.time_limit


def test_cell_coordinates_come_back_as_tuples_of_ints():
    """JSON has no tuples and no integer keys; a cell is both."""
    after = _round_trip(_level())
    for cell in after.cells:
        assert isinstance(cell, tuple) and len(cell) == 2
        assert all(isinstance(part, int) for part in cell)


def test_surfaces_survive_per_cell():
    before = _level(cell_surfaces={(1, 0): 'ice_sheet', (1, 1): 'rubber_pad'})
    after = _round_trip(before)
    assert after.cell_surfaces == before.cell_surfaces


def test_the_board_shape_survives():
    before = _level(cell_size=6.0, kill_y=-12.0, respawn_delay=0.5, surface='metal')
    after = _round_trip(before)
    assert after.cell_size == 6.0
    assert after.kill_y == -12.0
    assert after.respawn_delay == 0.5
    assert after.surface == 'metal'


def test_negative_coordinates_and_heights_survive():
    before = _level(cells={(-3, -2): -4.5, (0, 0): 0.0}, start_cell=(-3, -2),
                    finish_cell=(0, 0))
    assert _round_trip(before).cells == before.cells


# -- the features --------------------------------------------------------------

def test_every_feature_kind_survives_with_every_field():
    features = [
        Finish((2, 1)),
        Ramp(cell=(1, 0), direction=(0, 1), rise=1.25, boost_speed=6.5,
             launch=True, launch_up=4.25),
        Wall(cell=(1, 1), side='W', height=2.5, thickness=0.4, material='metal'),
        Bumper(cell=(0, 0), radius=0.55, height=1.1),
        SpringTrap(cell=(1, 1), impulse=(1.5, 9.0, -2.5), rearm=2.25),
        Elevator(cell=(2, 1), travel=4.5, period=2.25, thickness=0.6),
        RotatingArm(cell=(1, 0), length=3.0, rpm=42.0, thickness=0.35,
                    clearance=0.45),
    ]
    after = _round_trip(_level(features=features))
    assert after.features == features        # dataclass equality: every field


def test_a_feature_keeps_its_tuple_fields_as_tuples():
    after = _round_trip(_level(features=[SpringTrap(cell=(1, 1),
                                                    impulse=(1.0, 8.0, 2.0))]))
    trap = after.features[0]
    assert isinstance(trap.cell, tuple)
    assert isinstance(trap.impulse, tuple)


def test_an_unknown_feature_kind_is_refused_by_name():
    document = levelfile.to_json(_level())
    document['features'].append({'kind': 'trampoline', 'cell': [0, 0]})
    with pytest.raises(ValueError, match='trampoline'):
        levelfile.from_json(document)


def test_every_feature_class_the_game_has_can_be_written():
    """A mechanism added to level.py without a name here would not save.

    A subset rather than an equality: mechanisms also arrive from the
    ``mechanisms/`` package, where registering one *is* what puts it in
    ``FEATURES``, so the registry is properly larger than what ``level.py``
    defines.  That direction is held by
    ``test_registry.test_every_mechanism_is_a_feature_the_file_format_can_write``;
    what is asserted here is the one a person can still get wrong, which is
    writing a feature class into ``level.py`` and forgetting to name it.
    """
    import dataclasses

    from openglcontext_marble_demo import level as level_module
    authorable = {
        obj for obj in vars(level_module).values()
        if dataclasses.is_dataclass(obj) and isinstance(obj, type)
        and hasattr(obj, 'build') and hasattr(obj, 'owned_cells')}
    missing = authorable - set(levelfile.FEATURES.values())
    assert not missing, 'unsaveable: %s' % ', '.join(
        sorted(cls.__name__ for cls in missing))


# -- version and provenance ----------------------------------------------------

def test_the_document_says_what_wrote_it_and_which_version():
    document = levelfile.to_json(_level())
    assert document['generator'] == 'openglcontext-marble'
    assert document['version'] == levelfile.VERSION


def test_a_newer_file_is_refused_rather_than_half_read():
    document = levelfile.to_json(_level())
    document['version'] = levelfile.VERSION + 1
    with pytest.raises(ValueError, match='newer'):
        levelfile.from_json(document)


def test_a_seeded_level_records_the_seed_it_came_from():
    before = generator.generate(seed=12, difficulty=2)
    after = _round_trip(before)
    assert after.seed == 12
    assert after.difficulty == 2


# -- generated levels round-trip too -------------------------------------------

def test_a_generated_level_survives_unchanged():
    """Whatever the generator makes, the editor can open and the game reload."""
    for seed in range(6):
        before = generator.generate(seed=seed, difficulty=2)
        after = _round_trip(before)
        assert after.cells == before.cells
        assert after.features == before.features
        assert after.cell_surfaces == before.cell_surfaces


# -- files ---------------------------------------------------------------------

def test_save_then_load_gives_the_same_level(tmp_path):
    path = tmp_path / 'board.marble'
    before = generator.generate(seed=3, difficulty=2)
    levelfile.save(before, str(path))
    after = levelfile.load(str(path))
    assert after.cells == before.cells
    assert after.features == before.features


def test_the_file_is_text_a_person_can_read(tmp_path):
    path = tmp_path / 'board.marble'
    levelfile.save(_level(), str(path))
    text = path.read_text(encoding='utf-8')
    assert 'hand-made' in text
    assert text.endswith('\n')
    assert '\n  ' in text                       # indented, not one long line


def test_a_failed_write_leaves_the_previous_file_alone(tmp_path, monkeypatch):
    """A level file is a designer's only copy; a half-written one is worse than
    an unwritten one."""
    path = tmp_path / 'board.marble'
    levelfile.save(_level(name='original'), str(path))

    def explode(*args, **named):
        raise OSError('disk full')
    monkeypatch.setattr(levelfile.os, 'replace', explode)
    with pytest.raises(OSError):
        levelfile.save(_level(name='replacement'), str(path))
    assert 'original' in path.read_text(encoding='utf-8')
    assert list(tmp_path.iterdir()) == [path]   # nothing half-written left behind


def test_loading_something_that_is_not_a_level_says_so(tmp_path):
    path = tmp_path / 'not-a-board.marble'
    path.write_text('{"hello": "world"}', encoding='utf-8')
    with pytest.raises(ValueError):
        levelfile.load(str(path))


def test_a_level_loaded_from_a_file_builds_and_plays(tmp_path):
    """The point of the format: the game can play what the editor wrote."""
    from openglcontext_marble_demo.game import PLAYING, MarbleGame
    path = tmp_path / 'board.marble'
    levelfile.save(generator.generate(seed=4, difficulty=2), str(path))
    game = MarbleGame(levelfile.load(str(path)))
    for _ in range(120):
        game.lean(0.0, 1.0)
        game.advance(1 / 120.0)
    assert game.state == PLAYING
    assert math.isfinite(game.controller.speed)
