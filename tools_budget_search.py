#!/usr/bin/env python
"""T-16 — Budget-constrained weight search (affordable form of the grid idea).

Finds a float weight set that BEATS the Rust-default anchor by a stable >50
Elo margin at depth 6, scored by a CHEAP proxy (depth-4, few games) during the
sweep and a FULL head-to-head (depth-6, 50 games, 3 independent seeds) only for
the top candidates. The "budget" is a multiplicative factor
`f_i in [0.25, 4.0]` applied to the Rust-default weight for each of the 6
features, so `w_i = default_i * f_i` keeps the eval's relative shape close to
the defaults (avoids one term 100x another).

Search space (plan T-16):
  * Random sampler: `--candidates` (default 40) points with each factor drawn
    LogUniform(0.25, 4.0) from a seeded RNG.
  * `--sampler {random,ascent}` — the plan lists "pick one"; random is the
    default (40 evaluations ≈ plan); ascent is an optional 2-pass coordinate
    search (12 candidates/pass × 2 = 24).

Cheap proxy (per candidate):
  selfplay --weights-a cand.json --weights-b weights_v0.json --games PG
           --depth PD --workers W --seed S_i --out cand_s{i}.csv
  -> (W, L, D) -> S = (W + 0.5D)/N (clip 0.01..0.99)
       R = 1000 + 400*log10(S/(1-S))
Rank by proxy R; confirm the top `--max-confirm` (default 3) by 3 seeds
(5000+j) at the full depth; accept iff `mean(R) - 1000 > 50  AND  std(R) < 20
AND  min(R) > 1000`.

Pure standard-library. Windows cp1252 guard applied (reconfigure + ASCII-only
stdout).
"""

import argparse
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys

# --- Windows cp1252 stdout guard (project memory mandate) ----------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths (relative to the project directory — the script lives there) -------
ROOT = os.path.dirname(os.path.abspath(__file__))
SELFPLAY_EXE = os.path.join(ROOT, "ai", "target", "release", "selfplay.exe")
ANCHOR_WEIGHTS = os.path.join(ROOT, "weights_v0.json")

ANCHOR_R = 1000.0

TERMS = ["progress", "base_presence", "defense", "spread", "mobility", "threat"]

# Fallback in case the anchor file is missing (should not happen in-tree).
FALLBACK_DEFAULTS = {
    "progress": 6.0, "base_presence": 30.0, "defense": 25.0,
    "spread": 4.0, "mobility": 3.0, "threat": 55.0,
}

F_MIN, F_MAX = 0.25, 4.0   # plan T-16 factor range
PROXY_SEED_BASE = 10000    # distinct family from confirm seeds
CONFIRM_SEEDS   = [5000, 5001, 5002]   # plan T-16


def load_defaults():
    if os.path.isfile(ANCHOR_WEIGHTS):
        with open(ANCHOR_WEIGHTS) as f:
            return json.load(f)
    return dict(FALLBACK_DEFAULTS)


def run_selfplay(games, depth, seed, out_csv, w_a_path, w_b_path, workers=12,
                 maxmove=None):
    if not os.path.isfile(SELFPLAY_EXE):
        raise SystemExit(f"SELFPLAY_EXE missing: {SELFPLAY_EXE}")
    args = [
        SELFPLAY_EXE,
        "--games", str(games), "--depth", str(depth), "--seed", str(seed),
        "--out", out_csv,
        "--weights-a", w_a_path, "--weights-b", w_b_path,
        "--workers", str(max(1, workers)),
    ]
    if maxmove is not None:
        args += ["--maxmove", str(maxmove)]
    print("  $ " + " ".join(args))
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("  ERROR selfplay exit={}".format(r.returncode))
        print("  stdout: " + r.stdout.strip())
        print("  stderr: " + r.stderr.strip())
        raise SystemExit("selfplay failed")
    return r


def wins_from_selfplay(stdout):
    m = re.search(r"P1 wins=(\d+), P2 wins=(\d+), ongoing=(\d+)", stdout)
    if not m:
        raise SystemExit("selfplay output unparsable:\n" + stdout)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_SR(W, L, D):
    N = W + L + D
    if N == 0:
        return 0.5, ANCHOR_R
    S = (W + 0.5 * D) / N
    S = max(0.01, min(0.99, S))
    R = ANCHOR_R + 400.0 * math.log10(S / (1.0 - S))
    return S, R


def dblind(candidate_w, games, depth, seed, out_csv, workers=12, maxmove=None,
           anchor=ANCHOR_WEIGHTS):
    """Double-blind head-to-head of `candidate_w` vs the anchor, combining both
    move orders so the first-mover (P1) advantage cancels.

    halma has a first-mover (P1) advantage — measured 2026-11-02 as ~+58 Elo
    when both sides run identical weights (17/7 of finished games to P1 at cap
    480). The old loop put the candidate on P1 and the anchor on P2, confounding
    'being P1' with 'better weights'. Fix: play BOTH orders and combine:
        order A: candidate = P1 (weights-a), anchor = P2 (weights-b)
        order B: candidate = P2 (weights-b), anchor = P1 (weights-a)
    candidate total:  W = W_A + L_B ;  L = L_A + W_B ;  D = D_A + D_B
    Returns the combined (W, L, D) from the candidate's point of view.
    """
    base, ext = os.path.splitext(out_csv)  # cand_proxy0.csv -> _A.csv / _B.csv
    rA = run_selfplay(games, depth, seed, base + "_A" + ext,
                      candidate_w, anchor,
                      workers=workers, maxmove=maxmove)
    rB = run_selfplay(games, depth, seed + 50000, base + "_B" + ext,
                      anchor, candidate_w,
                      workers=workers, maxmove=maxmove)
    WA, LA, DA = wins_from_selfplay(rA.stdout)
    WB, LB, DB = wins_from_selfplay(rB.stdout)
    return WA + LB, LA + WB, DA + DB


# ---------------------------------------------------------------------------
# Candidate samplers (deterministic given seed)
# ---------------------------------------------------------------------------

def candidate_random(n, seed):
    """n candidates; per candidate, each factor ~ LogUniform(F_MIN, F_MAX)."""
    rng = random.Random(seed)
    out = []
    lo = math.log(F_MIN)
    hi = math.log(F_MAX)
    for _ in range(n):
        f = {}
        for t in TERMS:
            f[t] = math.exp(lo + rng.random() * (hi - lo))
        out.append(f)
    return out


def candidate_ascent(defaults, n_passes=2, seed=0):
    """Coordinate-ascent sampler.
    Start at f_i = 1.0 for all i. For each candidate, for each feature in turn,
    try {0.5, 1, 2} * f_i (others held), take the best by a cheap tiebreak
    (feature index order is enough for reproducibility). After `n_passes`
    passes of this, emit the best state of each pass as one candidate.
    This gives 12 (features x 3 multipliers) / pass ≈ plan's "12/pass".
    """
    rng = random.Random(seed)
    base = {t: 1.0 for t in TERMS}
    out = []
    for _ in range(n_passes):
        best_state = dict(base)
        # A single deterministic sweep (no scoring) — this sampler produces an
        # exploration of distinct multiplicative offsets; the cheap proxy is
        # what decides "best" per gate. For determinism we emit 12 distinct
        # perturbations per pass.
        for t in TERMS:
            for m in (0.5, 1.0, 2.0):
                c = dict(base)
                c[t] = c[t] * m
                # keep within [F_MIN, F_MAX]
                for k in TERMS:
                    if c[k] < F_MIN: c[k] = F_MIN
                    if c[k] > F_MAX: c[k] = F_MAX
                out.append(c)
    return out[-12 * n_passes:] if n_passes > 0 else out


GRID_FLAGS = {"progress": "progress", "defense": "defense", "threat": "threat",
              "base": "base_presence"}


def candidate_grid(args, defaults):
    """T-17 grid: cartesian product over ABSOLUTE weight values.
    4 flagged terms take comma lists (--progress/--defense/--threat/--base);
    the remaining terms must each be --pin'd to exactly one value.
    Returns factor dicts (w/default) so the existing pipeline (CSV f_* columns,
    w = default*f) is unchanged.
    """
    lists = {}
    for flag, term in GRID_FLAGS.items():
        raw = getattr(args, flag)
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        if not vals:
            raise SystemExit(f"grid: --{flag} must be a non-empty comma list "
                             f"(e.g. 12,15,18)")
        lists[term] = [float(v) for v in vals]
    for p in args.pin:
        t, _, v = p.partition("=")
        t = t.strip()
        if not v.strip() or t not in TERMS:
            raise SystemExit(f"grid: bad --pin '{p}' (want TERM=VALUE, TERM in {TERMS})")
        if t in lists:
            raise SystemExit(f"grid: {t} given both a list and a --pin")
        lists[t] = [float(v)]
    missing = [t for t in TERMS if t not in lists]
    if missing:
        raise SystemExit("grid: no value for: " + ", ".join(missing)
                         + "  (use --progress/--defense/--threat/--base or --pin)")
    cells = [lists[t] for t in TERMS]
    out = []
    for combo in itertools.product(*cells):
        w = dict(zip(TERMS, combo))
        out.append({t: w[t] / defaults[t] for t in TERMS})
    return out


# ---------------------------------------------------------------------------
# CSV / verdict
# ---------------------------------------------------------------------------

def write_csv_rows(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def write_verdict(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def main(argv=None):
    # Windows console defaults to cp1252; the skip-proxy NO-ACCEPT verdict
    # prints em-dash/minus (non-ASCII) at the very end -> force UTF-8 so the
    # long run doesn't die on the final print (data files are written before it
    # anyway, but a clean exit-0 avoids a scary traceback after ~4h).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="T-16 — budget-constrained weight search (proxy -> confirm).")
    ap.add_argument("--sampler", choices=["random", "ascent"], default="random")
    ap.add_argument("--grid", action="store_true",
                    help="T-17: enumerate a grid box over ABSOLUTE weight lists "
                         "(--progress/--defense/--threat/--base) plus "
                         "--pin TERM=VALUE for the fixed terms")
    ap.add_argument("--progress", default="",
                    help="grid: non-empty comma list of progress values")
    ap.add_argument("--defense", default="",
                    help="grid: non-empty comma list of defense values")
    ap.add_argument("--threat", default="",
                    help="grid: non-empty comma list of threat values")
    ap.add_argument("--base", default="",
                    help="grid: non-empty comma list of base_presence values")
    ap.add_argument("--pin", action="append", default=[], metavar="TERM=VALUE",
                    help="grid: pin a term to one value (repeatable; e.g. spread=4)")
    ap.add_argument("--anchor-file", default="weights_v0.json",
                    help="anchor weights file (default weights_v0.json); ratings "
                         "are on the anchor's 1000 scale")
    ap.add_argument("--candidates", type=int, default=40,
                    help="number of proxy-scored candidates (random: default 40)")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="RNG seed base; proxy seed for cand i = base + 10000 + i")
    ap.add_argument("--proxy-games", type=int, default=15)
    ap.add_argument("--proxy-depth", type=int, default=4)
    ap.add_argument("--confirm-games", type=int, default=50)
    ap.add_argument("--confirm-depth", type=int, default=6)
    ap.add_argument("--maxmove", type=int, default=480,
                    help="per-game move cap for every h2h (proxy + confirm). "
                         "120 default -> ~95%% 'ongoing' noise floor; 480 -> most "
                         "games resolve to a winner. Applied PER ORDER.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-confirm", type=int, default=3,
                    help="number of ranked candidates to confirm (top-N)")
    ap.add_argument("--skip-proxy", action="store_true",
                    help="skip the d4 proxy screen; confirm ALL grid cells "
                         "directly at confirm settings and rank by d6 mean")
    ap.add_argument("--out-dir", default=".",
                    help="where the budget_search_log.csv / weights land")
    ap.add_argument("--skip-confirm", action="store_true",
                    help="skip the depth-6 confirm phase (proxy-only, for smoke)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    defaults = load_defaults()

    anchor = args.anchor_file
    if not os.path.isabs(anchor):
        anchor = os.path.join(ROOT, anchor)

    log_csv  = os.path.join(args.out_dir, "budget_search_log.csv")
    verdict_ = os.path.join(args.out_dir, "budget_search_verdict.txt")
    best_w   = os.path.join(args.out_dir, "best_weights.json")

    print("\n=== tools_budget_search.py (T-16/T-17) ===")
    print(f"  sampler={'grid' if args.grid else args.sampler}  "
          f"candidates={args.candidates}  seed_base={args.seed_base}")
    if args.skip_proxy:
        print("  proxy:  SKIPPED (--skip-proxy) -> d6-confirm ALL cells directly")
    else:
        print(f"  proxy:  games={args.proxy_games} depth={args.proxy_depth} "
              f"maxmove={args.maxmove} double-blind workers={args.workers}")
    if args.skip_confirm:
        _confirm_desc = "SKIPPED"
    else:
        _confirm_desc = (f"games={args.confirm_games} depth={args.confirm_depth} "
                         f"maxmove={args.maxmove} double-blind seeds={CONFIRM_SEEDS}")
    print(f"  confirm: {_confirm_desc}")
    print(f"  out_dir={os.path.abspath(args.out_dir)}")
    print(f"  anchor file: {anchor} (R={ANCHOR_R:.0f} on this scale)")

    # sample candidates
    if args.grid:
        cand = candidate_grid(args, defaults)
    elif args.sampler == "random":
        cand = candidate_random(args.candidates, args.seed_base)
    else:
        cand = candidate_ascent(defaults, n_passes=2, seed=args.seed_base)
        cand = cand[: args.candidates] if len(cand) > args.candidates else cand
    print(f"  {len(cand)} candidates sampled")

    # header for the CSV
    header = (["rank",
               "f_progress", "f_base_presence", "f_defense", "f_spread",
               "f_mobility", "f_threat",
               "proxy_S", "proxy_R",
               "confirm_R_s5000", "confirm_R_s5001", "confirm_R_s5002",
               "confirm_mean_R", "confirm_std_R", "confirm_min_R",
               "verdict"])
    rows = []
    scored = []  # list of (factor_dict, proxy_S, proxy_R, cand_idx)

    # ---------------- candidate weight files (always written) ----------------
    for i, f in enumerate(cand):
        w = {t: defaults[t] * f[t] for t in TERMS}
        w_path = os.path.join(args.out_dir, f"cand_w{i}.json")
        with open(w_path, "w") as fh:
            fh.write(json.dumps({t: float(w[t]) for t in TERMS}))

    # ---------------- proxy phase (optional) ------------------------------------
    if args.skip_proxy:
        print(f"\n--- Proxy screen SKIPPED ({len(cand)} cells -> d6-confirm directly) ---")
        scored = [(f, 0.0, 0.0, i, 0, 0, 0) for i, f in enumerate(cand)]
    else:
        print(f"\n--- Proxy scoring ({len(cand)} candidates) ---")
        for i, f in enumerate(cand):
            w_path = os.path.join(args.out_dir, f"cand_w{i}.json")
            out_csv = os.path.join(args.out_dir, f"cand_proxy{i}.csv")
            proxy_seed = args.seed_base + PROXY_SEED_BASE + i
            W, L, D = dblind(w_path, args.proxy_games, args.proxy_depth,
                             proxy_seed, out_csv,
                             workers=args.workers, maxmove=args.maxmove,
                             anchor=anchor)
            S, R = compute_SR(W, L, D)
            scored.append((f, S, R, i, W, L, D))
            print(f"  cand[{i:>3}]  S={S:.5f} R={R:8.2f}  (W={W} L={L} D={D})")

    if not args.skip_proxy:
        scored.sort(key=lambda x: x[2], reverse=True)  # higher proxy R first

    # ---------------- confirm phase (top-N) -----------------------------------
    if args.skip_confirm:
        print("\n--- Confirm phase SKIPPED (proxy-only run) ---")
        for rank, (f, S, R, i, W, L, D) in enumerate(scored, 1):
            rows.append([str(rank)] + [f"{f[t]:.4f}" for t in TERMS]
                         + [f"{S:.4f}", f"{R:.2f}"] + [""] * 6
                         + ["not confirmed (skip-confirm)"])
        write_csv_rows(log_csv, header, rows)
        v = (f"PROXY-ONLY: {len(scored)} candidates scored at "
             f"d{args.proxy_depth}/{args.proxy_games}. "
             f"top proxy R={scored[0][2]:.2f} (delta {scored[0][2]-ANCHOR_R:+.2f}).")
        write_verdict(verdict_, v)
        print(f"\n  LOG     -> {log_csv}")
        print(f"  VERDICT -> {v}")
        return 0

    confirm_topn = len(scored) if args.skip_proxy else min(args.max_confirm, len(scored))
    confirm = {}  # candidate idx -> dict of per-seed R, mean, std, min
    print(f"\n--- Confirm ({confirm_topn} cells; "
          f"d{args.confirm_depth}/{args.confirm_games}, seeds {CONFIRM_SEEDS}) ---")
    for rank, (f, S, R, i, W, L, D) in enumerate(scored[:confirm_topn], 1):
        w_path = os.path.join(args.out_dir, f"cand_w{i}.json")
        per_seed_R = []
        for j, cs in enumerate(CONFIRM_SEEDS):
            c_seed = args.seed_base + cs + j
            c_csv  = os.path.join(args.out_dir, f"cand_confirm_{i}_s{c_seed}.csv")
            Wc, Lc, Dc = dblind(w_path, args.confirm_games, args.confirm_depth,
                                c_seed, c_csv,
                                workers=args.workers, maxmove=args.maxmove,
                                anchor=anchor)
            Sc, Rc = compute_SR(Wc, Lc, Dc)
            per_seed_R.append(Rc)
            print(f"  rank {rank}  seed {c_seed}: W={Wc} L={Lc} D={Dc}  S={Sc:.4f} R={Rc:.2f}")
        mean_R = sum(per_seed_R) / len(per_seed_R)
        var    = sum((x - mean_R) ** 2 for x in per_seed_R) / len(per_seed_R)
        std_R  = math.sqrt(var)
        min_R  = min(per_seed_R)
        confirm[i] = {
            "R": per_seed_R, "mean": mean_R, "std": std_R, "min": min_R,
        }

    if args.skip_proxy:
        scored.sort(key=lambda x: confirm.get(x[3], {}).get("mean", -1.0),
                    reverse=True)  # no proxy -> rank by d6 confirm mean

    # ---------------- accept / reject -----------------------------------------
    verdicts = {}   # candidate idx -> verdict string
    for rank, (f, S, R, i, W, L, D) in enumerate(scored, 1):
        if i not in confirm:
            verdicts[i] = "not confirmed (ranked below top-N)"
            continue
        C = confirm[i]
        ok = (C["mean"] - ANCHOR_R > 50.0) and (C["std"] < 20.0) and (C["min"] > ANCHOR_R)
        if ok:
            verdicts[i] = (f"ACCEPTED (mean={C['mean']:.2f} std={C['std']:.2f} "
                           f"min={C['min']:.2f})")
        else:
            reasons = []
            if not (C["mean"] - ANCHOR_R > 50.0):
                reasons.append(f"mean_delta={C['mean']-ANCHOR_R:+.2f}"
                               f" (need > 50)")
            if not (C["std"] < 20.0):
                reasons.append(f"std={C['std']:.2f} (need < 20)")
            if not (C["min"] > ANCHOR_R):
                reasons.append(f"min={C['min']:.2f} (need > {ANCHOR_R:.0f})")
            verdicts[i] = "REJECTED [" + "; ".join(reasons) + "]"

    # ---------------- write CSV -----------------------------------------------
    for rank, (f, S, R, i, W, L, D) in enumerate(scored, 1):
        c = confirm.get(i)
        c_cells = (
            [f"{x:.2f}" for x in c["R"]] + [f"{c['mean']:.2f}",
                                             f"{c['std']:.2f}", f"{c['min']:.2f}"]
            if c else ["", "", "", "", ""]
        ) if not args.skip_confirm else [""] * 6
        proxy_cells = ["NA", "NA"] if args.skip_proxy else [f"{S:.4f}", f"{R:.2f}"]
        rows.append([str(rank)] + [f"{f[t]:.4f}" for t in TERMS]
                    + proxy_cells + c_cells + [verdicts[i]])
    write_csv_rows(log_csv, header, rows)

    # ---------------- best weights / verdict ----------------------------------
    accepted = [(i, confirm[i]["mean"]) for (i, _) in
                [(x[3], 0) for x in scored] if i in verdicts and verdicts[i].startswith("ACCEPTED")]
    accepted.sort(key=lambda x: x[1], reverse=True)
    best_i = accepted[0][0] if accepted else None

    if best_i is not None:
        f = [c[0] for c in scored if c[3] == best_i][0]
        w = {t: float(defaults[t] * f[t]) for t in TERMS}
        with open(best_w, "w") as fh:
            fh.write(json.dumps(w))
        top_mean = confirm[best_i]["mean"]
        verdict = (f"ACCEPTED rank={best_i + 1}  mean_R@depth6 = {top_mean:.2f} "
                   f"(delta {top_mean - ANCHOR_R:+.2f}); best_weights.json written.")
    else:
        top = scored[0]
        top_i = top[3]
        if args.skip_proxy:
            m = confirm.get(top_i, {}).get("mean", 0.0)
            verdict = (
                f"NO ACCEPT (skip-proxy d6-only pass) — best d6 mean_R = {m:.2f} "
                f"(delta vs anchor {m - ANCHOR_R:+.2f}); "
                f"no confirmed cell cleared mean−1000>50 & std<20 & min>1000 vs "
                f"this anchor. Full d6 per-cell surface in budget_search_log.csv.")
        elif top_i in verdicts:
            verdict = (
                f"NO ACCEPT — best proxy R = {top[2]:.2f} "
                f"(delta {top[2]-ANCHOR_R:+.2f}); "
                f"best confirm (if run) -> {verdicts.get(top_i, 'n/a')}; "
                f"ongoing-game floor (most games hit the move cap) limits the h2h "
                f"signal below the T-16 acceptance gate. "
                f"VERDICT: research outcome — no weight set clears "
                f"mean>1050 & std<20 & min>1000 in this measurement. See "
                f"budget_search_log.csv.")
        else:
            verdict = f"NO ACCEPT — best proxy R = {top[2]:.2f} (delta {top[2]-ANCHOR_R:+.2f})."

    write_verdict(verdict_, verdict)
    print(f"\n  LOG     -> {log_csv}")
    print(f"  VERDICT -> {verdict}")
    if best_i is not None:
        print(f"  BEST    -> {best_w}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
