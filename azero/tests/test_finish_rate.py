"""Tests for azero/finish_rate.py (T-14 gate).

finish_rate is the one metric that can SEE the finish the full-game Elo is
blind to: the fraction of fresh near-win positions a net converts to a true
win. Gated: it returns a float in [0,1], and on a trivially-winnable set it is
> 0 (every synthesized position is winnable-in-one, so MCTS finds the win).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from azero.model import HalmaNet
from azero.finish_rate import finish_rate


def _model(seed=1234):
    torch.manual_seed(seed)
    m = HalmaNet()
    m.eval()
    return m


def test_returns_float_in_range():
    m = _model()
    fr = finish_rate(m, n_positions=6, full=8, max_moves=6, n_sims=12,
                     device="cpu", seed=20261110)
    assert isinstance(fr, float)
    assert 0.0 <= fr <= 1.0


def test_trivially_winnable_is_positive():
    """Every synthesized position is winnable-in-one; MCTS (no noise) finds a
    1-move win among the ~dozen legal steps within a few tens of sims, so
    across a small set at least one must convert even for a random net."""
    m = _model()
    fr = finish_rate(m, n_positions=24, full=8, max_moves=6, n_sims=40,
                     device="cpu", seed=20261110)
    assert fr > 0.0, f"expected at least one near-win converted, got {fr}"
