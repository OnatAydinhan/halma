#!/usr/bin/env python
"""T-15 — tools_tune.py: selfplay → train → re-selfplay, Elo vs fixed anchor
(decoupled-seed loop).

The T-11 loop regenerated its training CSV at a single fixed seed, so every
"iteration" fit the same data and the Elo curve was a flat line plotted N×.
T-15 decouples the seeds so the curve can actually move:

  • Training seed   for iteration i = seed_base + i     (fresh data each iter)
  • Measurement seed for iteration i = seed_base + 1000 + i  (different data)
  • h2h seed family stays distinct from every training seed.

Orchestrates the full tuning loop:
  • weights_v0.json = Rust defaults  (anchor, R = 1000 by construction)
  • Iteration i ≥ 1:
      1. train  : py train_weights.py selfplay.csv --label progress --seed (base+i) -o weights_v{i}.json
                  (selfplay.csv was generated at seed base+i — fresh per iter)
      2. measure: selfplay --weights-a weights_v{i} --weights-b weights_v0 --games N_elo --depth D --seed (base+1000+i)
                   → (W, L, D) from P1's (weights_v{i}) view
                   → S = (W + 0·5D) / N   [clip 0.01–0.99]
                   → R_i = 1000 + 400·log10(S/(1-S))
      3. log    : append a row to tune_log.txt (with train_seed + measure_seed)
      4. regen  : selfplay --games N --depth D --seed (base+i+1) --out selfplay.csv
  • Final: write elo_curve.svg

Pure standard-library (no numpy/pandas/matplotlib) — matches the box's Python
3.14 stdlib-only constraint.

Windows cp1252 guard applied before any user-facing output.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

# --- Windows cp1252 stdout guard (project memory mandate) ------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths --------------------------------------------------------------------
SELFPLAY_EXE = os.path.join("ai", "target", "release", "selfplay.exe")
TRAIN_PY     = "train_weights.py"

DEFAULT_WEIGHTS = {
    "progress": 6, "base_presence": 30, "defense": 25,
    "spread": 4, "mobility": 3, "threat": 55,
}

ANCHOR_R = 1000


# --- subprocess helpers --------------------------------------------------------

def run_cmd(args, label=None):
    """
    Run `args` (list of strings), print output, raise SystemExit on failure.
    Returns the subprocess.CompletedProcess so the caller can inspect stdout.
    """
    label = label or "$ " + " ".join(args)
    print(label)
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: exit={r.returncode}")
        print(f"  stdout: {r.stdout.strip()}")
        print(f"  stderr: {r.stderr.strip()}")
        raise SystemExit(label)
    for line in (r.stdout or "").strip().splitlines():
        print(f"  {line}")
    return r


# --- game / eval helpers -------------------------------------------------------

def ensure_weights_v0(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w") as f:
        f.write(json.dumps(DEFAULT_WEIGHTS))
    print(f"  wrote {path}: {json.dumps(DEFAULT_WEIGHTS)}")


def run_selfplay(games, depth, seed, out_csv, weights_a=None, weights_b=None,
                 workers=1, maxmove=None):
    args = [SELFPLAY_EXE, "--games", str(games), "--depth", str(depth),
            "--seed", str(seed), "--out", out_csv]
    if weights_a:
        args += ["--weights-a", weights_a]
    if weights_b:
        args += ["--weights-b", weights_b]
    # T-16 (h2h measurement): per-game move cap. Without it selfplay defaults to
    # 120, at which ~95% of games end 'ongoing' (noise floor). Default is applied
    # by the caller (main) — only appended when set.
    if maxmove is not None:
        args += ["--maxmove", str(maxmove)]
    # T-12: worker count. selfplay.exe defaults to 1 (serial) when omitted.
    if workers and workers > 1:
        args += ["--workers", str(workers)]
    return run_cmd(args)


def train(csv_path, out_weights, seed):
    args = ["py", TRAIN_PY, csv_path,
            "--label", "progress", "--mark", "50",
            "--seed", str(seed),
            "-o", out_weights]
    return run_cmd(args)


def parse_selfplay_result(stdout):
    """
    Parse a selfplay stdout block of the form:
    'selfplay: N games, M rows, P1 wins=X, P2 wins=Y, ongoing=Z'
    plus 'selfplay: median move ms: D'
    Returns (W, L, D_ongoing, median_ms).
    """
    m = re.search(r"P1 wins=(\d+), P2 wins=(\d+), ongoing=(\d+)", stdout)
    if not m:
        raise SystemExit(f"Cannot parse selfplay wins from:\n{stdout}")
    W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
    mm = re.search(r"median move ms:\s*(\d+)", stdout)
    median_ms = int(mm.group(1)) if mm else 0
    return W, L, D, median_ms


def compute_elo(W, L, D, anchor=ANCHOR_R):
    N = W + L + D
    S = (W + 0.5 * D) / N if N > 0 else 0.5
    S = max(0.01, min(0.99, S))
    R = anchor + 400 * math.log10(S / (1 - S))
    return S, R


# --- SVG writer -----------------------------------------------------------------

def write_elo_svg(points, out_svg,
                  title="check_halma — AI Elo over self-play iterations"):
    """
    points: [(iteration_int, rating_float), ...]  e.g. [(0,1000),(1,1042),(2,978),(3,1011)]
    Writes a self-contained, pure-inline SVG (no external resources).
    """
    W, H = 720, 360
    ML, MR, MT, MB = 60, 20, 35, 55
    pw, ph = W - ML - MR, H - MT - MB

    vals = [r for _, r in points]
    if not vals:
        vals = [ANCHOR_R]
    vmax, vmin = max(vals), min(vals)
    # guarantee anchor is in Y range AND at least 100 padding each side
    lo = min(vmin - 100, ANCHOR_R - 100)
    hi = max(vmax + 100, ANCHOR_R + 100)

    def x(i):
        n = len(points)
        if n == 1:
            return ML + pw / 2
        return ML + i / (n - 1) * pw

    def y(r):
        frac = (r - lo) / (hi - lo) if hi != lo else 0.5
        return MT + (1 - frac) * ph

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    p.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    p.append(f'<text x="{W//2}" y="22" text-anchor="middle" '
             f'font-family="sans-serif" font-size="14" font-weight="bold">'
             f'{esc(title)}</text>')

    # 5 horizontal grid lines + Y labels
    for k in range(5):
        r_val = lo + (k + 0.5) / 5 * (hi - lo)
        yy = y(r_val)
        p.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" '
                f'stroke="#e0e0e0" stroke-width="1"/>')
        p.append(f'<text x="{ML-4}" y="{yy+3:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="10" fill="#555">'
                f'{r_val:.0f}</text>')

    # X ticks: vertical line + label per iteration
    for i, _ in points:
        xx = x(i)
        p.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{H-MB}" '
                f'stroke="#f0f0f0" stroke-width="1"/>')
        p.append(f'<text x="{xx:.1f}" y="{H-MB+14}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" fill="#555">i={i}</text>')

    # Anchor dashed line — always drawn at ANCHOR_R even when not in points
    ay = y(ANCHOR_R)
    p.append(f'<line x1="{ML}" y1="{ay:.1f}" x2="{W-MR}" y2="{ay:.1f}" '
            f'stroke="#aaa" stroke-width="1.5" stroke-dasharray="7,5"/>')
    p.append(f'<text x="{W-MR-4}" y="{ay-5:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10" fill="#888">'
            f'Rust defaults (anchor = {ANCHOR_R})</text>')

    # Polyline through all points (anchor first)
    coords = " ".join(f"{x(i):.1f},{y(r):.1f}" for i, r in points)
    p.append(f'<polyline points="{coords}" fill="none" stroke="#446688" stroke-width="2.5"/>')

    # 4 px circles + value labels
    for i, r in points:
        xx, yy = x(i), y(r)
        p.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="#446688"/>')
        delta = r - ANCHOR_R
        sign = "+" if delta >= 0 else ""
        p.append(f'<text x="{xx:.1f}" y="{yy-8:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" font-weight="bold" '
                f'fill="#446688">{r:.0f} ({sign}{delta:.0f})</text>')

    p.append('</svg>')
    text = "\n".join(p) + "\n"
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(text)
    return out_svg


# --- gate sanity check ---------------------------------------------------------

def gate_check(W, L, D, S, R, acc, i, iterations, anchor=ANCHOR_R):
    problems = []
    # gate 2: S in [0.35, 0.65]
    if not (0.35 <= S <= 0.65):
        problems.append(f"S={S:.4f} outside [0.35, 0.65] — h2h may be buggy")
    # gate 7: R_1 != 1000 (at least one R_i differs by > 50)
    return problems


# --- main ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="T-11 — selfplay→train→Elo loop, SVG curve, baseline")
    ap.add_argument("--iterations", type=int, default=3,
                    help="number of measured iterations (i≥1); i=0 anchor is always included")
    ap.add_argument("--games", type=int, default=50,
                    help="selfplay games per training CSV (plain self-play, default weights)")
    ap.add_argument("--depth", type=int, default=6,
                    help="search depth for all selfplay runs")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="T-15 base seed: training seed for iter i = base+i; "
                         "measurement seed for iter i = base+1000+i")
    ap.add_argument("--elo-games", type=int, default=30,
                    help="head-to-head games per Elo measurement — PER ORDER "
                         "(double-blind runs both orders, so total = 2× here)")
    ap.add_argument("--maxmove", type=int, default=480,
                    help="h2h/measure per-game move cap. 120 (selfplay default) "
                         "lets ~95%% of games end 'ongoing' → noise floor; ~480 is "
                         "the first cap where most games resolve to a winner")
    ap.add_argument("--workers", type=int, default=12,
                    help="T-12 — selfplay worker threads (default 12 fits this 24-core box)")
    ap.add_argument("--out-dir", default=".",
                    help="directory for weights_vN.json, selfplay.csv, tune_log.txt, elo_curve.svg")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    w0_path   = os.path.join(out_dir, "weights_v0.json")
    csv_path  = os.path.join(out_dir, "selfplay.csv")
    tune_log  = os.path.join(out_dir, "tune_log.txt")
    elo_svg   = os.path.join(out_dir, "elo_curve.svg")

    print(f"\n=== tools_tune.py (T-15 decoupled-seed loop) ===")
    print(f"  iterations={args.iterations}  games={args.games}  depth={args.depth}")
    print(f"  seed_base={args.seed_base}  elo_games={args.elo_games}(per order)  "
          f"maxmove={args.maxmove}  double-blind h2h  workers={args.workers}")
    print(f"    train   seeds: {[args.seed_base + i for i in range(1, args.iterations + 1)]}")
    print(f"    measure seeds: {[args.seed_base + 1000 + i for i in range(1, args.iterations + 1)]}")
    print(f"  out_dir={os.path.abspath(out_dir)}\n")

    # ---- anchor (i=0) -----------------------------------------------------------
    print("--- i=0 (anchor = Rust defaults = 1000) ---")
    ensure_weights_v0(w0_path)
    points = [(0, ANCHOR_R)]

    # write tune_log header (T-15: train_seed + measure_seed columns)
    with open(tune_log, "w") as f:
        f.write("iteration,train_seed,measure_seed,games,depth,"
                "S_score,R_elo,test_accuracy,median_move_ms\n")

    # generate initial training CSV (iteration 1 trains on seed_base + 1)
    init_train_seed = args.seed_base + 1
    print(f"\n--- Generating initial selfplay.csv (training data, "
          f"seed_base+1={init_train_seed}) ---")
    run_selfplay(args.games, args.depth, init_train_seed, csv_path,
                 workers=args.workers)

    # ---- iterations 1..N ---------------------------------------------------------
    all_results = []
    for i in range(1, args.iterations + 1):
        print(f"\n--- Iteration {i} ---")
        # T-15: decoupled seeds (train family ≠ measure family)
        train_seed   = args.seed_base + i
        measure_seed = args.seed_base + 1000 + i
        w_i_path = os.path.join(out_dir, f"weights_v{i}.json")
        elo_csv_a = os.path.join(out_dir, f"elo_v{i}_A.csv")
        elo_csv_b = os.path.join(out_dir, f"elo_v{i}_B.csv")

        # step 1: train (selfplay.csv was generated at train_seed for this iter)
        print(f"  [step 1] train: selfplay.csv → weights_v{i}.json "
              f"(train_seed={train_seed})")
        r_train = train(csv_path, w_i_path, train_seed)
        acc_m = re.search(r"accuracy=([\d.]+)", r_train.stdout)
        acc = float(acc_m.group(1)) if acc_m else 0.0

        # step 2: MEASURE Elo with a DOUBLE-BLIND head-to-head.
        # halma has a first-mover (P1) advantage — measured 2026-11-02 as ~+58
        # Elo when both sides run identical weights (17/7 of finished games to
        # P1 at cap 480). The old loop put the candidate on P1 and the anchor on
        # P2, so "being P1" was conflated with "better weights". Double-blind
        # plays BOTH orders and combines them so the side bias cancels:
        #   order A: candidate = P1 (A), anchor = P2 (B)
        #   order B: candidate = P2 (B), anchor = P1 (A)
        # candidate total: W = W_A + L_B ; L = L_A + W_B ; D = D_A + D_B
        print(f"  [step 2] double-blind h2h: weights_v{i} vs weights_v0, "
              f"{args.elo_games} games/order, depth {args.depth}, "
              f"maxmove={args.maxmove}, seeds {measure_seed}/{measure_seed + 50000}")
        r_A = run_selfplay(args.elo_games, args.depth, measure_seed, elo_csv_a,
                            weights_a=w_i_path, weights_b=w0_path,
                            workers=args.workers, maxmove=args.maxmove)
        r_B = run_selfplay(args.elo_games, args.depth, measure_seed + 50000, elo_csv_b,
                            weights_a=w0_path, weights_b=w_i_path,
                            workers=args.workers, maxmove=args.maxmove)
        W_A, L_A, D_A, ms_A = parse_selfplay_result(r_A.stdout)
        W_B, L_B, D_B, ms_B = parse_selfplay_result(r_B.stdout)
        W = W_A + L_B
        L = L_A + W_B
        D = D_A + D_B
        median_ms = int(round((ms_A + ms_B) / 2)) if (ms_A or ms_B) else 0
        S, R = compute_elo(W, L, D)
        print(f"    order A (cand=P1): W={W_A} L={L_A} D={D_A}  median={ms_A}ms")
        print(f"    order B (cand=P2): W={W_B} L={L_B} D={D_B}  median={ms_B}ms")
        print(f"  results: cand W={W} L={L} D={D}  "
              f"S={S:.5f}  R={R:.2f}  median_ms={median_ms}  acc={acc:.4f}")

        # live gate warnings
        if not (0.35 <= S <= 0.65):
            print(f"  ⚠ S={S:.4f} outside normal [0.35, 0.65] range — investigate h2h path")
        if abs(R - ANCHOR_R) < 1:
            print(f"  ⚠ R={R:.2f} is within 1.0 of anchor (1000) — the fit may not be changing behaviour")

        # step 3: log (T-15: train_seed + measure_seed columns)
        log_line = (f"{i},{train_seed},{measure_seed},{args.games},{args.depth},"
                    f"{S:.5f},{R:.2f},{acc:.4f},{median_ms}")
        with open(tune_log, "a") as f:
            f.write(log_line + "\n")
        print(f"  [step 3] logged → {log_line}")

        # append to points (for SVG)
        all_results.append((i, R, S, acc))

        # step 4: regenerate training CSV for next iteration at the NEXT train seed
        if i < args.iterations:
            next_train_seed = args.seed_base + (i + 1)
            print(f"  [step 4] regenerating selfplay.csv (train_seed={next_train_seed}, "
                  f"{args.games} games, depth {args.depth})")
            run_selfplay(args.games, args.depth, next_train_seed, csv_path,
                         workers=args.workers)
        # (no regen needed after the last iteration)

        points.append((i, R))

    # ---- final -----------------------------------------------------------------
    svg_path = write_elo_svg(points, elo_svg)
    print(f"\n  wrote {svg_path}")

    # summary
    print(f"\n=== SUMMARY ===")
    print(f"  tune_log.txt:  {os.path.abspath(tune_log)}")
    print(f"  elo_curve.svg: {os.path.abspath(elo_svg)}")
    print(f"\n  {'iter':<6}  {'R':>7}  {'Δ':>7}  {'S':>7}  {'acc':>6}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    print(f"  {'i=0':<6}  {ANCHOR_R:>7.1f}  {'0.0':>7}  {'—':>7}  {'—':>6}")
    for i, R, S, acc in all_results:
        delta = R - ANCHOR_R
        print(f"  {'i='+str(i):<6}  {R:>7.1f}  {delta:>+7.1f}  {S:>7.5f}  {acc:>6.4f}")
    all_near_anchor = all(abs(R - ANCHOR_R) < 50 for _, R, _, _ in all_results)
    if all_near_anchor:
        print(f"\n  ⚠ All R_i within 50 of anchor (1000). Fit may still be degenerate.")
        print(f"    Check tune_log.txt and the per-iteration training output for clues.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
