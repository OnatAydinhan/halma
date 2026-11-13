#!/usr/bin/env bash
# T-17 Pass B — 2D interaction grids over the strongest axis pairs (d4/15,
# cap 480, double-blind, anchor = incumbent best_weights.json, --skip-confirm).
# Per Pass A surface report + plan_ai.md T-17:
#   1) progress x threat  (progress 15,18,21,24 x threat 90,110,130),
#      defense=8 (Pass-A argmax) + base=90 (Pass-A argmax) fixed
#   2) progress x base    (progress 15,18,21,24 x base 75,90,105),
#      defense=8 + threat=110 (Pass-A argmax) fixed
# spread=4 / mobility=9 pinned throughout (T-17a box convention).
# NOTE: tools_budget_search.py's --grid requires ALL FOUR flag terms to be
# non-empty comma lists (even a 1-value "fixed" axis); --pin is reserved for
# the remaining two terms (spread/mobility).

set -u
cd "$(dirname "$0")"

mkdir -p ./_t17_passB
COMMON=(--proxy-games 15 --proxy-depth 4 --maxmove 480 --workers 12
        --anchor-file best_weights.json --skip-confirm)

echo "=== T-17 Pass B — $(date) ==="

echo "--- 1: progress x threat ---"
py -u tools_budget_search.py --grid \
  --progress 15,18,21,24 --threat 90,110,130 \
  --defense 8 --base 90 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_pair1 \
  > _t17_passB/pair1_progress_threat.log 2>&1 || echo "pair1 FAILED (exit $?)"

echo "--- 2: progress x base ---"
py -u tools_budget_search.py --grid \
  --progress 15,18,21,24 --base 75,90,105 \
  --defense 8 --threat 110 \
  --pin spread=4 --pin mobility=9 \
  "${COMMON[@]}" --out-dir _t17_pair2 \
  > _t17_passB/pair2_progress_base.log 2>&1 || echo "pair2 FAILED (exit $?)"

echo "=== T-17 Pass B done — $(date) ==="
echo "Logs: _t17_passB/{pair1_progress_threat,pair2_progress_base}.log"
echo "CSVs: _t17_pair1/budget_search_log.csv , _t17_pair2/budget_search_log.csv"
