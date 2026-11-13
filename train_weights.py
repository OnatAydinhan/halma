#!/usr/bin/env python
"""T-10 — Weight training, pure standard library (no numpy/pandas/sklearn).

Reads a self-play CSV (produced by `ai/target/release/selfplay.exe`) and learns
the six named eval-term `Weights` via a standardized L2-regularized logistic
regression fit of `(mover eval features) -> did the mover eventually win?`.
Writes a flat `weights.json` with the six keys, matching what Rust's
`evaluate.rs::Weights::load` expects (integer values; no intercept — the search
does not apply a constant term).

Why pure stdlib: the plan's intended pandas/sklearn deps are not installed on
this box, and installing them needs network + write perms and exceeds the
1-minute cap. A hand-rolled gradient-descent logistic fit is dependency-free,
portable, and fully reproducible given a `--seed`.

Robustness (T-10 decision, 2026-11-01): self-play games almost never reach a
winner (a check_halma race is long; the default 120-move cap makes every game
"ongoing"). When a CSV contains **no decisive rows** (no `win` or no `loss`
class) the binary fit has no signal, so we fall back to the Rust default weight
set, write it, and warn loudly — the file is still valid and loads cleanly, and
the real T-10 gate (the Rust<->Python eval round-trip) does not depend on fit
quality.

(T-10.5) The training *label* is selectable:
  --label progress  (default) — mid-game geometric base-race proxy, defined for every
                    game (weight-independent: read straight from the board at the
                    reference ply `--mark`, default 50). This is what actually gives
                    the fit a non-degenerate signal, because self-play rarely decides.
  --label final           — the T-10 "did the mover win the full game?" label, kept as
                    the A/B control (structurally degenerate when games mostly go
                    "ongoing").

Usage:
  py train_weights.py selfplay.csv -o weights.json
  py train_weights.py selfplay.csv --label progress --mark 50 -o weights.json
  py train_weights.py selfplay.csv --label final -o weights.json   # A/B control
"""

import argparse
import csv
import json
import math
import random
import sys

# The six named eval terms, in the exact order the CSV columns appear in and
# the exact order Rust composes them in `side_score` (evaluate.rs).
TERMS = ["progress", "base_presence", "defense", "spread", "mobility", "threat"]

# The Rust `Weights::default()` values — used as the fallback when there is no
# learnable (win vs loss) signal in the CSV, so the output stays valid and
# matches the engine's built-in prior.
DEFAULT_WEIGHTS = {
    "progress": 6,
    "base_presence": 30,
    "defense": 25,
    "spread": 4,
    "mobility": 3,
    "threat": 55,
}

# Loud-warning threshold from the plan: any |weight| above this is a sign of a
# degenerate fit (we still write the file).
WEIGHT_SANITY_BOUND = 1000

# (T-10.5) The two 3x3 target bases, as board-string indices (board is 64 chars,
# index i = r*8 + c, row-major). Layout from the wire format / halma8.py:
#   P1 starts at rows 5-7, cols 0-2 (P1's base = P2's target).
#   P2 starts at rows 0-2, cols 5-7 (P2's base = P1's target).
# progress_1 measures P1's intrusion into P2's base; progress_2 into P1's base.
P2_BASE_CELLS = [r * 8 + c for r in range(3) for c in range(5, 8)]
P1_BASE_CELLS = [r * 8 + c for r in range(5, 8) for c in range(3)]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_rows(path):
    """Return (X, y) lists; only decisive rows (win/loss) are usable.

    X: list of 6-int feature vectors (the mover's own eval terms).
    y: 1 if the mover eventually won the game, 0 otherwise.
    `ongoing` rows are skipped for the binary fit, per the plan.
    """
    X, y, n_ongoing, n_bad = [], [], 0, 0
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            label = (r.get("result_from_movers_view") or "").strip()
            if label == "ongoing":
                n_ongoing += 1
                continue
            if label not in ("win", "loss"):
                n_bad += 1
                continue
            try:
                feats = [int(r[t]) for t in TERMS]
            except (KeyError, ValueError, TypeError):
                n_bad += 1
                continue
            X.append(feats)
            y.append(1 if label == "win" else 0)
    return X, y, n_ongoing, n_bad


# (T-10.5) The `progress` label: a mid-game geometric base-race proxy, defined
# for every game (not just the rare decisive ones), so the fit stops degenerating
# when self-play almost never produces a winner.

def _occupied_in_base(board, cells):
    """Number of occupied squares in a 3x3 base (9 - empty), i.e. how many
    pieces currently sit in that target base."""
    return 9 - sum(1 for i in cells if board[i] == ".")


def _board_of(position_string):
    """The 64-char board part of a `position` string = `<board> <turn> <chain>`."""
    parts = position_string.split()
    if not parts or len(parts[0]) < 64:
        raise SystemExit("train_weights: malformed position_string (board part missing)")
    return parts[0][:64]


def load_rows_progress(path, mark):
    """(T-10.5) Derive a mid-game geometric label for every row.

    For each `game_id` group, take the reference row nearest `ply == mark`.
    The mark is **clamped to the game's actual length**: a game that finished
    before `mark` (an early win) is referenced at its final position, so the
    label is defined for every game and never hard-errors (T-11 unblock,
    2026-11-01). Plies are contiguous `0..L-1`, so the nearest row to
    `min(mark, L-1)` is an exact match. At that position
    read the base-race geometry purely from the board:
      progress_1 = occupied squares in P2's base (P1's target)
      progress_2 = occupied squares in P1's base (P2's target)
    Then, for every row of the game with mover M:
      label = +1 if M's progress > opponent's, -1 if < , 0 if tied.
    A row whose derived label is 0 is skipped (tied reference position); the rest
    become y = 1 / 0 (M was ahead / behind in the race at the reference point).
    Weight-independent: derived from the board geometry only, so there is no
    circular dependency on the weights being fitted.
    """
    games = {}
    order = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            gid_raw = r.get("game_id")
            if gid_raw is None:
                continue
            gid = int(gid_raw)
            if gid not in games:
                games[gid] = []
                order.append(gid)
            games[gid].append(r)

    X, y = [], []
    n_tied = 0
    n_bad = 0
    for gid in order:
        rows = games[gid]
        # Reference ply = `mark` clamped to the game's last ply, so a game that
        # ends before the mark is referenced at its final position (never far from
        # any row; plies are contiguous, so the nearest row is an exact match).
        plies = []
        for r in rows:
            try:
                plies.append(int(r["ply"]))
            except (KeyError, ValueError, TypeError):
                pass
        if not plies:
            n_bad += len(rows)
            continue
        ref_target = min(mark, max(plies))
        best, best_gap = None, None
        for r in rows:
            try:
                gap = abs(int(r["ply"]) - ref_target)
            except (KeyError, ValueError, TypeError):
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best = gap, r
        if best is None:
            n_bad += len(rows)
            continue
        board = _board_of(best["position_string"])
        progress_1 = _occupied_in_base(board, P2_BASE_CELLS)
        progress_2 = _occupied_in_base(board, P1_BASE_CELLS)
        if progress_1 == progress_2:
            n_tied += len(rows)
            continue
        p1_ahead = progress_1 > progress_2
        for r in rows:
            try:
                mover = int(r["mover"])
                feats = [int(r[t]) for t in TERMS]
            except (KeyError, ValueError, TypeError):
                n_bad += 1
                continue
            y.append(1 if (mover == 1) == p1_ahead else 0)
            X.append(feats)
    return X, y, n_tied, n_bad


# ---------------------------------------------------------------------------
# logistic regression (stdlib)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _standardize(X, train_idx):
    """Per-feature mean/std computed on the training split only."""
    mean, std = [0.0] * 6, [0.0] * 6
    n = len(train_idx)
    for t in range(6):
        s = sum(X[i][t] for i in train_idx)
        m = s / n
        ss = sum((X[i][t] - m) ** 2 for i in train_idx)
        sd = math.sqrt(ss / n)
        mean[t] = m
        std[t] = sd if sd > 0 else 1.0
    return mean, std


def _fit(X, y, train_idx, mean, std, max_iter, lr, lam, seed):
    """L2-regularized logistic regression on standardized features.

    Returns (w_std, b) in the *standardized* space; caller maps back to raw
    scale. Deterministic given the fixed data order (no per-iteration shuffling
    is needed for a small, repeated small-batch objective — full gradient each
    step, which is exact and reproducible).
    """
    Xs = [[(X[i][t] - mean[t]) / std[t] for t in range(6)] for i in train_idx]
    ys = [y[i] for i in train_idx]
    n = len(train_idx)
    w = [0.0] * 6
    b = 0.0
    for _ in range(max_iter):
        gw = [0.0] * 6
        gb = 0.0
        for zi, yi in zip(Xs, ys):
            z = b
            for t in range(6):
                z += w[t] * zi[t]
            err = _sigmoid(z) - yi
            for t in range(6):
                gw[t] += err * zi[t]
            gb += err
        for t in range(6):
            w[t] -= lr * (gw[t] / n + lam * w[t])
        b -= lr * (gb / n)
    return w, b


def _predict(w_std, b, X, mean, std):
    out = []
    for i in range(len(X)):
        z = b
        for t in range(6):
            z += w_std[t] * ((X[i][t] - mean[t]) / std[t])
        out.append(1 if z > 0 else 0)
    return out


def _accuracy(y_true, y_pred):
    ok = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return ok / len(y_true) if y_true else 0.0


def _f1(y_true, y_pred):
    tp = fp = fn = 0
    for a, b in zip(y_true, y_pred):
        if b == 1 and a == 1:
            tp += 1
        elif b == 1 and a == 0:
            fp += 1
        elif b == 0 and a == 1:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _split_indices(n, frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(round(n * frac)))
    return idx[:k], idx[k:]


def to_raw_weights(w_std, std):
    """Map standardized coefficients back to the raw-feature scale Rust expects.

    Rust scores `w_raw * raw_term`; we trained on `(raw - mean)/std`, so the
    raw weight is `w_std / std`. (T-14) Emitted as a float rounded to 4 dp —
    `Weights::load` parses `f64`, so a "CP budget" shared across features that
    differ 10–30× in raw scale survives the round, instead of being forced to 0
    by the old integer rounding (the root cause of the flat, defense-only fits).
    """
    return {
        TERMS[t]: round(w_std[t] / std[t], 4)
        for t in range(6)
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="path to the self-play CSV (selfplay.exe output)")
    ap.add_argument("-o", "--out", default="weights.json")
    ap.add_argument(
        "--label",
        choices=["final", "progress"],
        default="progress",
        help=(
            "training label: 'progress' = mid-game geometric base-race proxy "
            "(T-10.5 default, defined for every game); 'final' = game-end winner "
            "(T-10; structurally degenerate because self-play rarely decides) — kept "
            "as the A/B control"
        ),
    )
    ap.add_argument(
        "--mark",
        type=int,
        default=50,
        help="ply at which the 'progress' label's reference position is read (default 50)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=5000, help="logistic fit iterations")
    ap.add_argument("--lrm", type=float, default=0.5, help="learning rate")
    ap.add_argument("--lam", type=float, default=1e-3, help="L2 regularization")
    ap.add_argument("--holdout", type=float, default=0.2, help="test split fraction")
    args = ap.parse_args(argv)

    if args.label == "progress":
        X, y, n_tied, n_bad = load_rows_progress(args.csv, args.mark)
        print(
            f"train_weights: {args.csv}: label=progress mark={args.mark}: "
            f"{len(X)} labelled rows, {n_tied} tied (skipped), {n_bad} bad (skipped)"
        )
    else:
        X, y, n_ongoing, n_bad = load_rows(args.csv)
        print(
            f"train_weights: {args.csv}: label=final: {len(X)} decisive rows, "
            f"{n_ongoing} ongoing (skipped), {n_bad} bad (skipped)"
        )
    n = len(X)

    classes = sorted(set(y)) if y else []
    weights = None
    if 0 not in classes or 1 not in classes or n == 0:
        # No usable binary signal -> fall back to the engine's built-in prior.
        if args.label == "progress":
            print("WARN: no learnable base-race signal (need BOTH an ahead and a behind row).")
            print("       If the reference positions are almost always tied at this --mark,")
            print("       raise/ lower --mark so the base is under pressure, or add games")
            print("       (or lengthen --maxmove) to get non-tied reference rows.")
        else:
            print("WARN: no learnable win/loss signal (need BOTH a `win` and a `loss` row).")
            print("       Self-play games rarely finish within the move cap, so this is expected")
            print("       unless the CSV was produced with a high --maxmove and enough games.")
        print("       Falling back to the Rust default weight set (still a valid weights.json).")
    else:
        train_idx, test_idx = _split_indices(n, 1 - args.holdout, args.seed)
        mean, std = _standardize(X, train_idx)
        w_std, b = _fit(X, y, train_idx, mean, std, args.max_iter, args.lrm, args.lam, args.seed)
        learned = to_raw_weights(w_std, std)

        acc = f1 = None
        if test_idx:
            pred = _predict(w_std, b, X, mean, std)
            acc = _accuracy([y[i] for i in test_idx], [pred[i] for i in test_idx])
            f1 = _f1([y[i] for i in test_idx], [pred[i] for i in test_idx])
            print(f"train_weights: held-out (n={len(test_idx)}) accuracy={acc:.4f} f1={f1:.4f}")
        else:
            print("train_weights: held-out split empty; skipped accuracy/f1")

        # Degenerate-fit guard (T-14): the float fit can no longer collapse to
        # all-zero *by rounding*, so the only remaining no-signal check is the
        # held-out accuracy — if the fit sits at/below chance, fall back to the
        # known-good Rust defaults so the AI keeps a meaningful (non-constant)
        # evaluation rather than a constant one.
        at_chance = acc is not None and acc <= 0.5 + 1e-9
        if at_chance:
            print("WARN: fit is degenerate (held-out accuracy at/below chance); its learned")
            print("       weights are not better than the engine's built-in prior. Falling back")
            print("       to the Rust default weight set so the AI keeps a meaningful")
            print("       (non-constant) evaluation.")
        else:
            weights = learned

    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)

    # Sanity bound from the plan.
    for k, v in weights.items():
        if abs(v) > WEIGHT_SANITY_BOUND:
            print(f"WARN: |{k}| = {v} exceeds sanity bound {WEIGHT_SANITY_BOUND} (likely degenerate fit).")

    out = {t: weights[t] for t in TERMS}
    with open(args.out, "w") as f:
        f.write(json.dumps(out))
    print(f"train_weights: wrote {args.out}: " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
