"""Tests for azero/selfplay.py (T-4)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from halma8 import Halma8, P1
from azero.model import HalmaNet
from azero.selfplay import play_one_game, self_play_batch, selfplay_worker
from azero.value_target import load_weights, calibrate_scale


def _weights_and_scale():
    w = load_weights(os.path.join(os.path.dirname(__file__), '..', '..', 'best_weights.json'))
    return w, calibrate_scale(w)


def _fresh_model(seed=1234):
    torch.manual_seed(seed)
    m = HalmaNet()
    m.eval()
    return m


def _forced_win_state():
    """P1 wins in one move: P2's base is 8/9 full (last empty (1,5)) and P1
    has a piece at (1,4) that can step into it. (Same position as test_mcts.)"""
    s = Halma8()
    s.grid = [[0] * 8 for _ in range(8)]
    for (r, c) in [(0, 5), (0, 6), (0, 7), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)]:
        s.grid[r][c] = 2
    s.grid[1][4] = P1
    s.turn = P1
    s.winner = None
    s.chain_active = False
    s.chain_banned = set()
    s.chain_starter = None
    s.chain_current = None
    return s


def test_play_one_game_completes():
    model = _fresh_model()
    result = play_one_game(
        model, device="cpu", n_sims=10, max_moves=30, seed=7
    )
    assert isinstance(result, list)


def test_tuple_shapes():
    model = _fresh_model()
    result = play_one_game(
        model, device="cpu", n_sims=10, max_moves=60, seed=7
    )
    for enc, policy, value in result:
        assert enc.shape == (7, 8, 8)
        assert policy.shape == (417,)
        # Value is +1/-1 (won game) or a dense MCTS estimate (no-winner game).
        assert -1.0 <= value <= 1.0


def test_no_winner_yields_dense_data():
    """A short game from the start position cannot produce a winner (filling a
    9-square enemy base with an attacker piece is impossible in 8 moves), so it
    exercises the no-winner path. It must STILL return training tuples, with at
    least one value that is a dense (non-+/-1) MCTS estimate. This is what makes
    T-8's long run trainable: weak-model games are no longer discarded."""
    model = _fresh_model()
    result = play_one_game(
        model, device="cpu", n_sims=10, max_moves=8, seed=7
    )
    assert len(result) >= 1
    values = [v for _e, _p, v in result]
    assert all(-1.0 <= v <= 1.0 for v in values)
    assert any(-1.0 < v < 1.0 for v in values)


def test_self_play_batch_completes():
    model = _fresh_model()
    result = self_play_batch(
        model, n_games=3, device="cpu", n_sims=5, max_moves=20, seed=7
    )
    assert isinstance(result, list)


def test_self_play_batch_start_states_true_outcome():
    """T-14 gate: self_play_batch(start_states=[...]) plays exactly those games,
    and from a forced-win start the value targets hit +1 (true outcome, not the
    dense-MCTS ~0)."""
    model = _fresh_model()
    states = [_forced_win_state(), _forced_win_state()]
    result = self_play_batch(
        model, n_games=0, device="cpu", n_sims=50, max_moves=6,
        add_noise=False, seed=7, start_states=states,
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    for _enc, policy, value in result:
        assert policy.shape == (417,)
        assert value == 1.0, f"expected +1 (mover won) from a forced-win start, got {value}"


def test_self_play_batch_start_states_off_is_original():
    """T-14 gate: start_states=None reproduces the original fresh-board batch
    exactly (byte-identical for a fixed seed) — no hidden behavior change."""
    model = _fresh_model()
    a = self_play_batch(model, n_games=3, device="cpu", n_sims=5, max_moves=20, seed=7)
    b = self_play_batch(model, n_games=3, device="cpu", n_sims=5, max_moves=20,
                        seed=7, start_states=None)
    assert len(a) == len(b)
    for (e1, p1, v1), (e2, p2, v2) in zip(a, b):
        assert torch.equal(e1, e2)
        assert (p1 == p2).all()
        assert v1 == v2


def test_policy_sums_to_one():
    model = _fresh_model()
    result = play_one_game(
        model, device="cpu", n_sims=10, max_moves=60, seed=7
    )
    for _enc, policy, _value in result:
        assert abs(policy.sum() - 1.0) < 1e-4
        assert (policy >= 0).all()


def test_no_nan_inf():
    model = _fresh_model()
    result = play_one_game(
        model, device="cpu", n_sims=10, max_moves=60, seed=7
    )
    for enc, policy, value in result:
        assert torch.isfinite(enc.float()).all()
        assert np.isfinite(policy).all()
        assert np.isfinite(value)


def test_value_fill_forced_win():
    """Deterministic end-to-end check of the value-fill path: P1 to move in a
    forced-win position. MCTS finds the winning move, the game ends on the
    first record, and that record's value must be +1.0 (P1 wins)."""
    model = _fresh_model()
    state = _forced_win_state()
    assert state.winner is None
    result = play_one_game(
        model,
        device="cpu",
        n_sims=50,
        max_moves=5,
        add_noise=False,
        seed=7,
        start_state=state,
    )
    assert len(result) >= 1
    # Every recorded mover won (P1), so every value must be +1.0.
    for _enc, policy, value in result:
        assert value == 1.0
        assert policy.shape == (417,)


def test_selfplay_worker():
    """T-9 gate: the module-level worker (run by a spawned process) rebuilds the
    net from a plain state dict and returns a non-empty list of correctly-shaped
    tuples. Called directly here (no process spawn) to check its contract."""
    model = _fresh_model()
    args = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_games": 2,
        "n_sims": 5,
        "max_moves": 20,
        "device": "cpu",
        "seed": 7,
        "add_noise": True,
    }
    result = selfplay_worker(args)
    assert isinstance(result, list)
    assert len(result) >= 1, "worker produced no training tuples"
    for enc, policy, value in result:
        assert enc.shape == (7, 8, 8)
        assert policy.shape == (417,)
        assert abs(policy.sum() - 1.0) < 1e-4
        assert -1.0 <= value <= 1.0
        assert torch.isfinite(enc.float()).all()
        assert np.isfinite(policy).all()


def test_collect_selfplay_workers_spawn():
    """T-9 gate: the fan-out path actually spawns worker processes and merges
    their output into one well-formed data list (exercises the spawn import +
    pickling round-trip that the in-process worker test cannot cover)."""
    from azero.pipeline import _collect_selfplay

    model = _fresh_model()
    data = _collect_selfplay(
        model,
        n_games=2,
        device="cpu",
        n_sims=3,
        max_moves=8,
        workers=2,
        seed=7,
    )
    assert isinstance(data, list)
    assert len(data) >= 1, "multi-process self-play produced no training tuples"
    for enc, policy, value in data:
        assert enc.shape == (7, 8, 8)
        assert policy.shape == (417,)
        assert -1.0 <= value <= 1.0


def test_value_teacher_blends_informative_target():
    """T-12 gate: with value_teacher=1.0 the value target must be a real,
    informative signal, not the ~0 self-play target. From the start position a
    short game has no winner, so the W=0 baseline is the near-constant dense
    MCTS value (~0 for a weak net); the W=1.0 target distills the eval and
    must have a materially larger |v| spread while staying within [-1, 1]."""
    model = _fresh_model()
    w, scale = _weights_and_scale()
    common = dict(device="cpu", n_sims=10, max_moves=20, seed=7,
                  teacher_weights=w, teacher_scale=scale)
    base = play_one_game(model, value_teacher=0.0, **common)
    taught = play_one_game(model, value_teacher=1.0, **common)

    vb = [v for _e, _p, v in base]
    vt = [v for _e, _p, v in taught]
    assert len(vb) == len(vt) >= 1
    assert all(-1.0 <= v <= 1.0 for v in vt)

    mb = float(np.mean(np.abs(vb)))
    mt = float(np.mean(np.abs(vt)))
    assert mt > 0.1, f"teacher value target not informative: mean|v|={mt:.4f}"
    assert mt >= mb, (f"teacher target should beat the ~0 self-play baseline "
                      f"(mt={mt:.4f}, mb={mb:.4f})")


def test_value_teacher_off_is_original_behavior():
    """T-12 gate: value_teacher=0.0 (or no weights) must reproduce the exact
    pre-teacher self-play value target — no hidden behavior change."""
    model = _fresh_model()
    w, scale = _weights_and_scale()
    a = play_one_game(model, device="cpu", n_sims=10, max_moves=20, seed=7)
    b = play_one_game(model, device="cpu", n_sims=10, max_moves=20, seed=7,
                      value_teacher=0.0, teacher_weights=w, teacher_scale=scale)
    va = [v for _e, _p, v in a]
    vb = [v for _e, _p, v in b]
    assert len(va) == len(vb)
    for x, y in zip(va, vb):
        assert abs(x - y) < 1e-9


def test_dist_teacher_blends_informative_target():
    """T-13 gate: with dist_teacher=1.0 the value target must be a real,
    informative signal (focused advancement advantage), not the near-constant
    self-play target. Measured reality (seeded, n_sims=10, max_moves=20): the
    weak net's pieces barely advance within the cap, so the target is SMALL
    in magnitude (mean|v| ~0.03) but ~2x the baseline's spread — it carries a
    directional (sign) signal the ~0 MCTS value lacks. Gate: spread must be
    materially larger than the baseline and non-zero in absolute terms."""
    model = _fresh_model()
    common = dict(device="cpu", n_sims=10, max_moves=20, seed=7)
    base = play_one_game(model, dist_teacher=0.0, **common)
    taught = play_one_game(model, dist_teacher=1.0, **common)

    vb = [v for _e, _p, v in base]
    vt = [v for _e, _p, v in taught]
    assert len(vb) == len(vt) >= 1
    assert all(-1.0 <= v <= 1.0 for v in vt)

    mb = float(np.mean(np.abs(vb)))
    mt = float(np.mean(np.abs(vt)))
    assert mt > 0.01, f"dist value target not informative: mean|v|={mt:.4f}"
    assert mt > 1.5 * mb, (f"dist target should spread well beyond the "
                           f"near-constant self-play baseline "
                           f"(mt={mt:.4f}, mb={mb:.4f})")


def test_dist_teacher_off_is_original_behavior():
    """T-13 gate: dist_teacher=0.0 must reproduce the exact pre-T-13 self-play
    value target — no hidden behavior change."""
    model = _fresh_model()
    a = play_one_game(model, device="cpu", n_sims=10, max_moves=20, seed=7)
    b = play_one_game(model, device="cpu", n_sims=10, max_moves=20, seed=7,
                      dist_teacher=0.0, dist_scale=4.0)
    va = [v for _e, _p, v in a]
    vb = [v for _e, _p, v in b]
    assert len(va) == len(vb)
    for x, y in zip(va, vb):
        assert abs(x - y) < 1e-9


def test_both_teachers_blend_is_convex_and_informative():
    """T-13 gate: with BOTH teachers at W=1.0 the per-position value is
    (eval + dist) / 2, so it must stay in [-1, 1] and cannot exceed the max
    of the two individual teacher targets in magnitude (a convex combination
    stays in their convex hull). The blend must still be informative."""
    model = _fresh_model()
    w, scale = _weights_and_scale()
    common = dict(device="cpu", n_sims=10, max_moves=20, seed=7,
                  teacher_weights=w, teacher_scale=scale)
    ev = [v for _e, _p, v in play_one_game(model, value_teacher=1.0, **common)]
    di = [v for _e, _p, v in play_one_game(model, dist_teacher=1.0, **common)]
    both = [v for _e, _p, v in
            play_one_game(model, value_teacher=1.0, dist_teacher=1.0, **common)]
    assert len(ev) == len(di) == len(both) >= 1
    for e, d, b in zip(ev, di, both):
        assert -1.0 <= b <= 1.0
        assert abs(b) <= max(abs(e), abs(d)) + 1e-9
    assert float(np.mean(np.abs(both))) > 0.05
