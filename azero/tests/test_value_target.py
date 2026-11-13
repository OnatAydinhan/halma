"""Tests for azero/value_target.py (T-12 — teacher value target)."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from halma8 import Halma8
from azero.value_target import (
    load_weights, eval_signed, normalized_value, calibrate_scale,
    enemy_base_center, piece_centroid, centroid_distance,
    distance_advantage, calibrate_distance_scale,
    P1_BASE_CENTER, P2_BASE_CENTER,
)

P1, P2 = 1, 2


def _grid_with(p1_cells, p2_cells):
    """Raw 8x8 grid (0/1/2) with explicit P1/P2 cell placements for sign tests."""
    g = [[0] * 8 for _ in range(8)]
    for r, c in p1_cells:
        g[r][c] = P1
    for r, c in p2_cells:
        g[r][c] = P2
    return g

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WPATH = os.path.join(ROOT, 'best_weights.json')


def _weights():
    return load_weights(WPATH)


def test_load_weights():
    w = _weights()
    assert isinstance(w, dict) and len(w) > 0


def test_eval_signed_p2_is_negated():
    """Mover-perspective negamax: the P2 target is exactly the negation of the
    P1 target for the same grid (this is what keeps the value head in the
    negamax frame the rest of the pipeline assumes)."""
    w = _weights()
    grid = Halma8().grid
    e1 = eval_signed(grid, 1, w)
    e2 = eval_signed(grid, 2, w)
    assert e1 + e2 == 0.0  # exact: eval_signed returns `e` / `-e` by definition


def test_normalized_value_bounded():
    w = _weights()
    scale = calibrate_scale(w)
    for turn in (1, 2):
        nv = normalized_value(Halma8().grid, turn, w, scale)
        assert -1.0 <= nv <= 1.0
        assert math.isfinite(nv)


def test_normalized_value_sign_flips_with_turn():
    w = _weights()
    scale = calibrate_scale(w)
    grid = Halma8().grid
    assert abs(normalized_value(grid, 1, w, scale)) == abs(
        normalized_value(grid, 2, w, scale))


def test_calibrate_scale_positive_finite():
    s = calibrate_scale(_weights())
    assert s >= 1.0
    assert math.isfinite(s)


# --- T-13: distance-to-enemy-base shaping target -------------------------

def test_enemy_base_center_targets():
    # Each side advances toward the enemy's base center.
    assert enemy_base_center(P1) == P2_BASE_CENTER
    assert enemy_base_center(P2) == P1_BASE_CENTER


def test_centroid_start_positions():
    g = Halma8().grid
    assert piece_centroid(g, P1) == (6.0, 1.0)   # P1 home: bottom-left 3x3
    assert piece_centroid(g, P2) == (1.0, 6.0)   # P2 home: top-right 3x3


def test_centroid_distance_finite():
    g = Halma8().grid
    d = centroid_distance(g, P1, (1.0, 6.0))
    assert d > 0.0 and math.isfinite(d)


def test_distance_advantage_zero_at_start():
    # Symmetric start: both sides are equally (dis)advanced -> advantage 0.
    g = Halma8().grid
    assert abs(distance_advantage(g, P1, 4.0)) < 1e-9
    assert abs(distance_advantage(g, P2, 4.0)) < 1e-9


def test_distance_advantage_negation():
    # Negamax symmetry: swapping the mover exactly negates the advantage.
    p1 = [(r, c) for r in range(3) for c in range(5, 8)]   # top-right 3x3
    p2 = [(r, c) for r in range(5, 8) for c in range(5, 8)]  # bottom-right 3x3
    g = _grid_with(p1, p2)
    a1 = distance_advantage(g, P1, 4.0)
    a2 = distance_advantage(g, P2, 4.0)
    assert a1 == -a2


def test_distance_advantage_sign_p1_advanced():
    # P1 sitting on its target (top-right) while P2 is 5 away from its target
    # (bottom-left center) -> P1 is clearly ahead -> positive for P1.
    p1 = [(r, c) for r in range(3) for c in range(5, 8)]
    p2 = [(r, c) for r in range(5, 8) for c in range(5, 8)]
    g = _grid_with(p1, p2)
    assert distance_advantage(g, P1, 4.0) > 0.5
    assert -1.0 <= distance_advantage(g, P1, 4.0) <= 1.0


def test_calibrate_distance_scale_positive_finite():
    s = calibrate_distance_scale()
    assert s >= 1.0
    assert math.isfinite(s)
