"""T-17 STEP 1 — d4 boundary-hunt: greedy 1D coordinate-ascent over all 6 params.

Holds the prior axes at their running-best and sweeps ONE axis at a time, in
importance order, over a range that extends PAST the tested frontier so we can
see whether each curve turns down (a real boundary) or hits the tested edge
("still rising -> widen in pass 2"). Proxy tier only (d4/15, double-blind,
cap 480, anchor = incumbent best_weights.json = T-16 best 1285.94, R=1000 scale).

Ends by writing a combined surface report to _t17b/summary.txt and STOPPING:
this is a review gate (staged rule — no d6 without the user's explicit go-ahead).
Pure stdlib; Windows cp1252 stdout guard applied.
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
ANCHOR = "best_weights.json"     # incumbent T-16 best; R=1000 on this scale
OUTDIR = "_t17b"

PGAMES = 15
PDEPTH = 4
MAXMOVE = 480
WORKERS = 12
SEED_BASE = 50000

# defaults (weights_v0.json); value = f_<term> col * default[term]
DEFAULTS = json.load(open(os.path.join(ROOT, "weights_v0.json")))

# current best frontier point (absolute weights) from Pass B: grid1 = progress24 x threat130
current = {
    "progress": 24.0,
    "base_presence": 90.0,
    "defense": 8.0,
    "spread": 4.0,
    "mobility": 9.0,
    "threat": 130.0,
}

# 4 axes the tool can list-sweep; spread/mobility are --pin-only -> swept by the
# driver issuing separate 1-cell runs (PINNABLE).
LISTFLAG = {"progress": "--progress", "defense": "--defense",
            "threat": "--threat", "base_presence": "--base"}
PINNABLE = {"spread", "mobility"}

# importance order: progress-dominant, then threat/base, defense weak, then the
# two never-mapped axes. Ranges extend OUTWARD past the frontier to catch the turn.
AXES = [
    ("progress",      [15, 18, 21, 24, 27, 30, 33, 36, 42]),
    ("threat",        [70, 90, 110, 130, 150, 170]),
    ("base_presence", [60, 75, 90, 105, 120, 135]),
    ("defense",       [4, 6, 8, 10, 12, 15, 18]),
    ("spread",        [2, 3, 4, 6, 8, 10, 12]),
    ("mobility",      [3, 5, 7, 9, 11, 13]),
]


def base_argv():
    return [TOOL, "--grid", "--skip-confirm",
            "--proxy-games", str(PGAMES), "--proxy-depth", str(PDEPTH),
            "--maxmove", str(MAXMOVE), "--workers", str(WORKERS),
            "--anchor-file", ANCHOR, "--seed-base", str(SEED_BASE)]


def hold_others(axis, argv):
    """Specify every term except `axis` in `argv`, each at its running-best value:
    list-axes as a 1-value list, pinnable axes via --pin."""
    for t, val in current.items():
        if t == axis:
            continue
        if t in PINNABLE:
            argv += ["--pin", "{}={:g}".format(t, val)]
        else:
            argv += [LISTFLAG[t], "{:g}".format(val)]


def run_one(argv, outdir):
    r = subprocess.run(["py"] + argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("==== sub-run failed (out-dir={}) ====\n".format(outdir))
        sys.stderr.write(r.stdout[-2000:] + "\n----\n" + r.stderr[-2000:] + "\n")
        raise SystemExit("sub-run failed at out-dir {}".format(outdir))


def read_curve(axis, outdir):
    """Return sorted [(value, proxy_R)] for the swept axis from the proxy-only CSV."""
    path = os.path.join(outdir, "budget_search_log.csv")
    pts = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        ci = header.index("f_" + axis)
        cj = header.index("proxy_R")
        for row in rd:
            try:
                fv = float(row[ci]); R = float(row[cj])
            except ValueError:
                continue
            pts.append((fv * DEFAULTS[axis], R))
    pts.sort()
    return pts


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report_lines = [
        "T-17 STEP 1 — d4 boundary-hunt (greedy 1D coordinate-ascent, importance order)",
        "anchor = {} (incumbent, R=1000 scale); proxy d{}/{} double-blind cap {}".format(
            ANCHOR, PDEPTH, PGAMES, MAXMOVE),
        "start point P* = " + "  ".join("{:g}".format(v) for _, v in current.items()),
        "=" * 72,
    ]
    print("\n=== T-17 STEP 1 boundary-hunt (d4 proxy) ===")
    print("anchor={}  start P*= progress24 threat130 base90 defense8 spread4 mobility9".format(ANCHOR))

    for axis, values in AXES:
        if axis in PINNABLE:
            # driver-side sweep: one 1-cell grid run per value
            curve = []
            for i, v in enumerate(values):
                outdir = "{}/{}_{}".format(OUTDIR, axis, i)
                argv = base_argv() + ["--out-dir", outdir,
                                      "--pin", "{}={:g}".format(axis, v)]
                hold_others(axis, argv)
                run_one(argv, outdir)
                c = read_curve(axis, outdir)
                if c:
                    curve.append(c[0])
                print("  {}={:g} -> R={}".format(axis, v, c[0][1] if c else float("nan")))
        else:
            outdir = OUTDIR + "/" + axis
            argv = base_argv() + ["--out-dir", outdir,
                                  LISTFLAG[axis], ",".join(str(int(x)) for x in values)]
            hold_others(axis, argv)
            run_one(argv, outdir)
            curve = read_curve(axis, outdir)

        best_val, best_R = max(curve, key=lambda p: p[1])
        edge = (best_val == min(values)) or (best_val == max(values))
        old = current[axis]
        current[axis] = best_val
        note = "  [STILL RISING — edge of tested range; widen in pass 2]" if edge else ""
        for v, R in curve:
            mark = " *" if v == best_val else ""
            report_lines.append("{:<14} {:>5g}  R={:>8.2f}{}".format(axis, v, R, mark))
        report_lines.append("-" * 72)
        print("[{}] best={:g} R={:.2f} (was {:g}){}".format(axis, best_val, best_R, old, note))

    report_lines.append("")
    report_lines.append("GREEDY POINT after this pass (absolute weights):")
    rep_b = {}
    for t in ["progress", "base_presence", "defense", "spread", "mobility", "threat"]:
        rep_b[t] = current[t]
        report_lines.append("  {:<14} {:g}".format(t, current[t]))
    report_lines.append("")
    report_lines.append("NOTE: 1 forward pass (coordinate-ascent). Axes marked 'still rising'")
    report_lines.append("hit the tested max; a pass 2 (wider) or the d6 neighborhood search")
    report_lines.append("should follow after the user reviews this surface.")
    with open(os.path.join(OUTDIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n=== BOUNDARY HUNT DONE ===")
    print("greedy point -> " + "  ".join("{}={:g}".format(t, current[t])
                                         for t in ["progress", "base_presence", "defense",
                                                    "spread", "mobility", "threat"]))
    print("summary -> " + os.path.abspath(os.path.join(OUTDIR, "summary.txt")))
    print("STOP: proxy surface ready for user review (staged rule — no d6 until go-ahead).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
