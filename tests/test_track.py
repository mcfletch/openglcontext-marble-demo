"""Tests for :class:`TrackMap` — the grid of occupied track cells.

The track map answers the questions the controller needs every frame: which cell
is the marble over, is that cell part of the track, and how high is its surface.
Cell ``(col, row)`` is centred at world ``(col*size, ·, row*size)``.
"""
from openglcontext_marble_demo.track import TrackMap


def _flat_track():
    # A 3-cell straight run at height 0, cell size 4.
    return TrackMap({(0, 0): 0.0, (1, 0): 0.0, (2, 0): 0.0}, cell_size=4.0)


def test_cell_of_rounds_to_nearest_cell_centre():
    t = _flat_track()
    assert t.cell_of(0.0, 0.0) == (0, 0)
    assert t.cell_of(3.9, 0.1) == (1, 0)      # nearer the cell centred at x=4
    assert t.cell_of(-1.5, 0.0) == (0, 0)


def test_is_track_true_only_for_occupied_cells():
    t = _flat_track()
    assert t.is_track(0.0, 0.0)
    assert t.is_track(8.0, 0.0)               # cell (2,0)
    assert not t.is_track(0.0, 8.0)           # cell (0,2) — not in the map
    assert not t.is_track(20.0, 0.0)          # off the end


def test_surface_height_returns_none_over_void():
    t = TrackMap({(0, 0): 1.5, (1, 0): 2.0}, cell_size=4.0)
    assert t.surface_height(0.0, 0.0) == 1.5
    assert t.surface_height(4.0, 0.0) == 2.0
    assert t.surface_height(0.0, 8.0) is None


def test_cell_center_maps_back_to_world():
    t = _flat_track()
    assert t.cell_center(2, 0) == (8.0, 0.0)
    assert t.cell_center(0, 1) == (0.0, 4.0)
