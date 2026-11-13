"""T-17 STEP 1b — pass 2: convergence re-hunt around the pass-1 greedy point.

Pass 1 finished with progress flagged STILL RISING at 42 and no anchor sanity
cell. Pass 2 (still proxy d4/15, double-blind, cap 480, anchor = incumbent) does:
  * re-hunts progress WIDENED (to 80) against the FINAL neighbor context,
  * re-confirms the other axes' interior maxima in that context,
  * runs ONE sanity cell at the incumbent's exact weights -> must read ~1000
    (proves the '1000 = incumbent' reference this run, as T-17a did).
Coordinates-ascent: each axis holds the prior axes at their running-best.
Different seed family from pass 1 (60000) to avoid seed-correlation.
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
ANCHOR = "best_weights.json"
PGAMES, PDEPTH, MAXMOVE, WORKERS = 15, 4, 480, 12
SEED = 60000
DEFAULTS = json.load(open(os.path.join(ROOT, "weights_v0.json")))

OS   = {"progress": "--progress", "defense": "--defense",
        "threat": "--threat", "base_presence": "--base"}
PIN  = {"spread", "mobility"}

# pass-1 greedy point = starting context
current = {"progress": 42.0, "base_presence": 120.0, "threat": 130.0,
           "defense": 6.0, "spread": 4.0, "mobility": 9.0}

# incumbent (T-16 best) exact weights, for the sanity cell
INCUMBENT = {"progress": 14.18, "base_presence": 91.49, "defense": 14.77,
             "spread": 7.56, "mobility": 9.07, "threat": 91.60}

# pass-2 ranges: progress widened; others re-confirmed around their pass-1 peaks
AXES = [
    ("progress",      [18, 24, 30, 36, 42, 48, 55, 65, 80]),
    ("base_presence", [90, 105, 120, 135, 150]),
    ("threat",        [110, 130, 150]),
    ("defense",       [4, 6, 8, 10]),
    ("spread",        [3, 4, 5, 6]),
    ("mobility",      [7, 9, 11]),
]


def base(argv):
    return [TOOL, "--grid", "--skip-confirm",
            "--proxy-games", str(PGAMES), "--proxy-depth", str(PDEPTH),
            "--maxmove", str(MAXMOVE), "--workers", str(WORKERS),
            "--anchor-file", ANCHOR, "--seed-base", str(SEED)]


def hold(axis, argv):
    for t, val in current.items():
        if t == axis:
            continue
        if t in PIN:
            argv += ["--pin", "{}={:g}".format(t, val)]
        else:
            argv += [OS[t], "{:g}".format(val)]


def run_one(argv, outdir):
    r = subprocess.run(["py"] + argv, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("== fail " + outdir + " ==\n" + r.stdout[-2000:] + "\n" + r.stderr[-2000:] + "\n")
        raise SystemExit("sub-run failed at " + outdir)


def curve(axis, outdir):
    path = os.path.join(outdir, "budget_search_log.csv")
    pts = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        h = next(rd)
        ci, cj = h.index("f_" + axis), h.index("proxy_R")
        for row in rd:
            try:
                pts.append((float(row[ci]) * DEFAULTS[axis], float(row[cj])))
            except ValueError:
                pass
    pts.sort()
    return pts


def main():
    out = "_t17b_pass2"
    os.makedirs(out, exist_ok=True)
    rep = ["T-17 STEP 1b — pass 2 (convergence) around progress42/base120/threat130/"
           "defense6/spread4/mobility9; anchor=%s (R=1000 scale); d%d/%d" % (ANCHOR, PDEPTH, PGAMES)]
    print("=== T-17 pass 2 (d4 proxy, seed 60000) ===")

    for axis, values in AXES:
        if axis in PIN:
            all_pts = []
            for i, v in enumerate(values):
                o = "%s/%s_%d" % (out, axis, i)
                a = base([]) + ["--out-dir", o, "--pin", "%s=%g" % (axis, v)]
                hold(axis, a); run_one(a, o)
                c = curve(axis, o)
                if c:
                    all_pts.append(c[0])
            cv = all_pts
        else:
            o = out + "/" + axis
            a = base([]) + ["--out-dir", o, OS[axis], ",".join(str(int(x)) for x in values)]
            hold(axis, a); run_one(a, o)
            cv = curve(axis, o)

        bv, br = max(cv, key=lambda p: p[1])
        edge = bv == min(values) or bv == max(values)
        old = current[axis]; current[axis] = bv
        for v, R in cv:
            rep.append("%-14s %6g  R=%9.2f%s" % (axis, v, R, " *" if v == bv else ""))
        tag = "  [STILL RISING -> cap/flag]" if edge else ""
        rep.append("-" * 72)
        print("[%s] best=%g R=%.2f (was %g)%s" % (axis, bv, br, old, tag))

    # ---- anchor sanity cell: run the incumbent's exact weights, expect ~1000 ----
    o = out + "/_sanity"
    a = base([]) + ["--out-dir", o,
                    OS["progress"], "{:g}".format(INCUMBENT["progress"]),
                    OS["defense"], "{:g}".format(INCUMBENT["defense"]),
                    OS["threat"], "{:g}".format(INCUMBENT["threat"]),
                    OS["base_presence"], "{:g}".format(INCUMBENT["base_presence"]),
                    "--pin", "spread=%g" % INCUMBENT["spread"],
                    "--pin", "mobility=%g" % INCUMBENT["mobility"]]
    run_one(a, o)
    r_sanity = curve("progress", o)[0][1]
    ok = abs(r_sanity - 1000.0) <= 25.0
    rep.append("")
    rep.append("SANITY: incumbent-exact weights -> R=%.2f  (expect ~1000)  %s" % (r_sanity,
        "OK" if ok else "OUT OF BAND — treat all R values with caution"))
    rep.append("")
    rep.append("PASS-2 GREEDY POINT (absolute): " +
        " ".join("%s=%g" % (t, current[t]) for t in
                 ["progress", "base_presence", "defense", "spread", "mobility", "threat"]))
    rep.append("(1 forward pass each; run a 3rd pass / d6 neighborhood to harden before accept)")
    with open(os.path.join(out, "summary_pass2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    print("\n=== PASS 2 DONE ===")
    print("sanity (incumbent) R=%.2f -> %s" % (r_sanity, "OK (~1000)" if ok else "OUT OF BAND"))
    print("greedy -> " + " ".join("%s=%g" % (t, current[t]) for t in
                                  ["progress", "base_presence", "defense", "spread", "mobility", "threat"]))
    print("summary -> %s" % os.path.abspath(os.path.join(out, "summary_pass2.txt")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
