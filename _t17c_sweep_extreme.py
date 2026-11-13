"""T-17 STEP 1c — wide d4/25 SHAPE probe of progress and base to (dis)prove an
interior optimum.

Fixes the 4 locked axes (threat 130, defense 6, spread 4, mobility 9) and
sweeps ONE open axis at a time over a deliberately WIDE, log-ranged set that
includes "very large" values, to see whether the proxy-R surface turns down
(a real interior peak -> the boundary is reachable at d4) or keeps rising to
the tested edge / collapses (d4 has no interior point -> only d6 can rank it).

Proxy tier ONLY: depth-4, ~25 games/point, double-blind, cap 480, anchor =
incumbent best_weights.json (R=1000 scale). Pure stdlib. Ends by writing
_t17c/summary.txt and STOPPING (staged rule: no d6 until the user goes ahead).
"""

import csv
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOL = "tools_budget_search.py"
ANCHOR = "best_weights.json"          # incumbent T-16 best; R=1000 on this scale
OUTDIR = "_t17c"

PGAMES = 25                            # shape probe: a touch above the jittery 15
PDEPTH = 4
MAXMOVE = 480
WORKERS = 12

DEFAULTS = json.load(open(os.path.join(ROOT, "weights_v0.json")))

# LOCKED terms (converged in Pass 1+2, sanity OK) — fixed for both sweeps.
LOCKED = {"threat": 130.0, "defense": 6.0, "spread": 4.0, "mobility": 9.0}

# Wide, log-ranged sweeps with deliberately large high ends.
SWEEPS = [
    ("progress",      [14, 28, 56, 112, 224, 448, 896, 1792, 3584]),
    ("base_presence", [24, 48, 90, 180, 360, 720, 1440]),
]

# Anchor value for the OTHER open axis when it is not the one being swept.
ANCHOR_PROGRESS = 24.0                 # prior best-so-far for progress
ANCHOR_BASE     = 90.0                 # prior best-so-far for base (greedy pt)


def argv_for(axis, values):
    """Grid argv for the single-axis sweep: `axis` varies, the other open axis and
    all 4 locked terms are held at their anchors. spread/mobility are --pin only."""
    a = [TOOL, "--grid", "--skip-confirm",
         "--proxy-games", str(PGAMES), "--proxy-depth", str(PDEPTH),
         "--maxmove", str(MAXMOVE), "--workers", str(WORKERS),
         "--anchor-file", ANCHOR]
    a += ["--progress", ",".join(str(int(x)) for x in values)] if axis == "progress" \
         else ["--progress", "{:g}".format(ANCHOR_PROGRESS)]
    a += ["--base", ",".join(str(int(x)) for x in values)] if axis == "base_presence" \
         else ["--base", "{:g}".format(ANCHOR_BASE)]
    a += ["--defense", "{:g}".format(LOCKED["defense"])]
    a += ["--threat",  "{:g}".format(LOCKED["threat"])]
    a += ["--pin", "spread={:g}".format(LOCKED["spread"])]
    a += ["--pin", "mobility={:g}".format(LOCKED["mobility"])]
    return a


def run_one(argv, outdir):
    r = subprocess.run(["py"] + argv + ["--out-dir", outdir],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("==== sub-run failed (out-dir={}) ====\n".format(outdir))
        sys.stderr.write(r.stdout[-2000:] + "\n----\n" + r.stderr[-2000:] + "\n")
        raise SystemExit("sub-run failed at out-dir {}".format(outdir))


def read_curve(axis, outdir):
    """Return sorted [(absolute_value, proxy_R)] for the swept axis."""
    path = os.path.join(outdir, "budget_search_log.csv")
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        ci = header.index("f_" + axis)
        cj = header.index("proxy_R")
        pts = []
        for row in rd:
            try:
                pts.append((float(row[ci]) * DEFAULTS[axis], float(row[cj])))
            except ValueError:
                continue
    pts.sort()
    return pts


def verdict(pts, tested_max):
    """Classify the shape: interior peak / still-rising / collapse."""
    best_i = max(range(len(pts)), key=lambda i: pts[i][1])
    hi = pts[-1][1]
    if hi < 900.0:
        tail = " COLLAPSE (top-of-range R < 900: single-term dominance is pathological)"
    else:
        tail = ""
    if best_i == len(pts) - 1:
        return "STILL RISING at tested max {:g} -> boundary NOT reached at d4{}".format(
            tested_max, tail), best_i
    if best_i == 0:
        return "FALLING from the start (peak below the tested range){}".format(tail), best_i
    return "INTERIOR PEAK at {:g} (boundary reached; d4 can rank this axis){}".format(
        pts[best_i][0], tail), best_i


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = [
        "T-17 STEP 1c — wide d4/{} SHAPE probe (interior-optimum test)".format(PGAMES),
        "anchor = {} (incumbent, R=1000); locked: threat130 defense6 spread4 mobility9".format(ANCHOR),
        "goal: confirm each open axis (progress, base) turns over somewhere, or prove",
        "it doesn't at d4 (still-rising / collapse) -> then only d6 can rank it.",
        "=" * 74,
    ]
    print("=== T-17 1c wide d4 shape probe ===")
    print("locked: threat130 defense6 spread4 mobility9; sweeping progress & base wide")

    results = {}
    for axis, values in SWEEPS:
        outdir = OUTDIR + "/" + axis
        argv = argv_for(axis, values)
        print("sweeping {} over {} ({} points)...".format(axis, values, len(values)))
        run_one(argv, outdir)
        pts = read_curve(axis, outdir)
        if not pts:
            print("  [!] no curve rows for {}".format(axis)); continue
        v, bi = verdict(pts, values[-1])
        results[axis] = pts
        lo, hi = pts[0], pts[-1]
        peak = pts[bi]
        print("  R: value {} -> {}  |  peak {} @ {}".format(lo[0], hi[0], peak[1], peak[0]))
        print("  verdict: {}".format(v))
        for j, (val, R) in enumerate(pts):
            mark = " *" if j == bi else ""
            report.append("{:<14} {:>7g}  R={:>8.2f}{}".format(axis, val, R, mark))
        report.append("-" * 74)
        report.append("  VERDICT[{}]: {}".format(axis, v))
        report.append("")

    with open(os.path.join(OUTDIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print()
    print("=== SHAPE PROBE DONE -> {} ===".format(os.path.abspath(os.path.join(OUTDIR, "summary.txt"))))
    print("STOP: d4 shape surfaces ready for review (staged rule — no d6 until go-ahead).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
