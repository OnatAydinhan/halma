"""Bounded poller for _t17c_sweep_extreme.py: waits up to ~50 s, prints progress
each ~10 s, exits early when _t17c/summary.txt appears. Pure stdlib, ASCII only."""

import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(ROOT, "_t17c", "summary.txt")
LOG = os.path.join(ROOT, "C:\\Users\\onata\\.qwen\\tmp")  # fallback display only

def count_cell_csvs():
    n = 0
    base = os.path.join(ROOT, "_t17c")
    if not os.path.isdir(base):
        return n
    for d in os.listdir(base):
        p = os.path.join(base, d, "budget_search_log.csv")
        if os.path.isfile(p):
            try:
                n += sum(1 for _ in open(p)) - 1
            except Exception:
                pass
    return n

deadline = time.time() + 50.0
while time.time() < deadline:
    done = os.path.isfile(SUMMARY)
    print("poll: summary={} cell_rows={}".format("YES" if done else "no", count_cell_csvs()))
    if done:
        break
    time.sleep(10)
print("state=COMPLETED" if os.path.isfile(SUMMARY) else "state=STILL_RUNNING")
