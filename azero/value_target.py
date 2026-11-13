"""Value-target bootstrap for the AlphaZero NN (Part 3).

A *teacher* value target derived from the hand-crafted 6-term eval
(`ai/crosscheck_eval.py::py_evaluate`, the 448/540 weights). It is used only
to set training targets so the value head gets an informative gradient (it
starts at ~0 in pure self-play because every game is a draw). It never plays,
is not part of the trained net, and is faded to weight 0 for pure self-play.
"""

import json
import math
import random

from halma8 import Halma8
from azero.board import ACTION_LIST, action_index, decode_action, legal_action_indices

P1 = 1
P2 = 2


def load_weights(path):
    with open(path) as f:
        return json.load(f)


def eval_signed(grid, turn, weights):
    """World score from the *mover's* (turn) perspective.

    `py_evaluate` returns a P1-perspective score; negate it for P2 so the
    target is always in the mover's frame (negamax convention).
    """
    import ai.crosscheck_eval as _ce  # local import keeps `azero` import-light
    e = _ce.py_evaluate(grid, weights)
    return e if turn == P1 else -e


def normalized_value(grid, turn, weights, scale):
    """Mover-perspective eval squashed to [-1, 1] via tanh."""
    return math.tanh(eval_signed(grid, turn, weights) / max(1e-6, scale))


def _random_walk_positions(n_games, plies, seed=0):
    rng = random.Random(seed)
    pos = []
    for _ in range(n_games):
        st = Halma8()
        for _ in range(plies):
            pos.append((st.grid, st.turn))
            idxs = legal_action_indices(st)
            if not idxs:
                break
            a = ACTION_LIST[rng.choice(idxs)]
            if a[0] == "stop":
                st.end_turn()
            else:
                st.apply_move(decode_action(action_index(a)))
            if st.winner is not None:
                break
    return pos


def calibrate_scale(weights, n_games=40, plies=14, seed=0):
    """Divisor so `tanh(eval/scale)` maps typical positions into [-1, 1].

    Uses the ~85th percentile of |eval| over random-walk positions, so the
    middle of the distribution lands near tanh(1)=0.76 and only extremes
    saturate. Pure CPU, no GPU, sub-second.
    """
    vals = sorted(abs(eval_signed(g, t, weights)) for g, t in
                  _random_walk_positions(n_games, plies, seed))
    if not vals:
        return 1.0
    p = vals[min(len(vals) - 1, int(0.85 * len(vals)))]
    return max(1.0, float(p))


# --- T-13: distance-to-enemy-base shaping target -------------------------
#
# The 6-term eval is a "who is better right now" snapshot; advancement toward
# the win is only one diluted term. This adds a *focused* "am I getting closer
# to the win" gradient: each side's 9-piece centroid vs the enemy base center
# it is advancing toward. It is a value-target bootstrap (same fade-to-0
# pattern as the eval teacher), never a played feature.

_SIZE = 8
P1_BASE_CENTER = (6.0, 1.0)   # P1 (Red) base: rows 5-7, cols 0-2
P2_BASE_CENTER = (1.0, 6.0)   # P2 (Blue) base: rows 0-2, cols 5-7


def enemy_base_center(mover):
    """The enemy base center the `mover` is advancing toward."""
    return P2_BASE_CENTER if mover == P1 else P1_BASE_CENTER


def piece_centroid(grid, player):
    """Average (r, c) of all of `player`'s pieces, or None if it has none."""
    sr = sc = n = 0
    for r in range(_SIZE):
        row = grid[r]
        for c in range(_SIZE):
            if row[c] == player:
                sr += r
                sc += c
                n += 1
    if n == 0:
        return None
    return (sr / n, sc / n)


def centroid_distance(grid, player, target):
    """Euclidean distance from `player`'s piece centroid to `target`."""
    cent = piece_centroid(grid, player)
    if cent is None:
        return 0.0
    tr, tc = target
    cr, cc = cent
    return math.sqrt((cr - tr) ** 2 + (cc - tc) ** 2)


def distance_advantage(grid, mover, scale=4.0):
    """Mover-perspective signed advancement advantage in [-1, 1].

    `+1` = the mover's mass is markedly closer to the enemy base than the
    defender's is to the mover's base; `-1` = the reverse. Computed as
    `tanh((d_defender - d_mover) / scale)`, each `d` being that side's 9-piece
    centroid distance to the base it advances toward. Symmetric and
    negamax-consistent (one number covers both players).
    """
    other = P2 if mover == P1 else P1
    d_mine = centroid_distance(grid, mover, enemy_base_center(mover))
    d_their = centroid_distance(grid, other, enemy_base_center(other))
    adv = d_their - d_mine
    return math.tanh(adv / max(1e-6, scale))


def calibrate_distance_scale(n_games=40, plies=14, seed=0):
    """Divisor so `tanh(adv/scale)` maps typical positions into [-1, 1].

    Mirrors `calibrate_scale` for the eval: p85 of the raw |advantage| over
    random-walk positions. Pure CPU, sub-second.
    """
    raw = []
    for grid, turn in _random_walk_positions(n_games, plies, seed):
        other = P2 if turn == P1 else P1
        d_mine = centroid_distance(grid, turn, enemy_base_center(turn))
        d_their = centroid_distance(grid, other, enemy_base_center(other))
        raw.append(abs(d_their - d_mine))
    if not raw:
        return 4.0
    raw.sort()
    p = raw[min(len(raw) - 1, int(0.85 * len(raw)))]
    return max(1.0, float(p))
