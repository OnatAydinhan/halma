#!/usr/bin/env bash
# T-17 Pass A — single-axis main-effect scans (d4/15, cap 480, double-blind,
# anchor = incumbent best_weights.json, --skip-confirm).
# Per plan_ai.md T-17: hold other 3 axes at P*=(progress 15, defense 15, threat
# 90, base 90); sweep each axis alone, 7 values each, 4 separate runs.
# Spread/mobility pinned at 4/9 (T-17a box convention).

set -u
cd "$(dirname "$0")"

LOGDIR=./_t17_passA
mkdir -p "$LOGDIR"

COMMON=(--proxy-games 15 --proxy-depth 4 --maxmove 480 --workers 12
        --anchor-file best_weights.json --skip-confirm)

echo "=== T-17 Pass A — $(date) ==="

echo "--- A: progress scan ---"
py -u tools_budget_search.py --grid \
  --progress 6,9,12,15,18,21,24 \
  --defense 15 --threat 90 --base 90 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_sweepA \
  > "$LOGDIR/A_progress.log" 2>&1 || echo "A FAILED (exit $?)"

echo "--- B: defense scan ---"
py -u tools_budget_search.py --grid \
  --progress 15 --defense 4,8,12,15,18,22,26 --threat 90 --base 90 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_sweepB \
  > "$LOGDIR/B_defense.log" 2>&1 || echo "B FAILED (exit $?)"

echo "--- C: threat scan ---"
py -u tools_budget_search.py --grid \
  --progress 15 --defense 15 --threat 20,37,55,72,90,110,130 --base 90 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_sweepC \
  > "$LOGDIR/C_threat.log" 2>&1 || echo "C FAILED (exit $?)"

echo "--- D: base scan ---"
py -u tools_budget_search.py --grid \
  --progress 15 --defense 15 --threat 90 --base 30,45,60,75,90,105,120 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_sweepD \
  > "$LOGDIR/D_base.log" 2>&1 || echo "D FAILED (exit $?)"

echo "=== T-17 Pass A done — $(date) ==="
echo "Logs: $LOGDIR/{A_progress,B_defense,C_threat,D_base}.log"
echo "CSVs: _t17_sweepA..D/budget_search_log.csv"
