"""End-to-end check of the T-14 deadlock fix through the REAL pipeline path.

Calls `_collect_selfplay` (the function the training round actually uses) with
endgame near-win games, and counts how many returned value targets are true
terminal +/-1 (|v| >= 0.99). Before the fix (noise on, low sims) these were ~0;
after (noise off, 200 sims) they should be ~+/-1 for the converted games.

Tests BOTH the sequential path (workers=1) and the production worker path
(workers>1, spawn).

Minutes. ASCII-only. Run:  py -u azero/_probe_evalue_fix.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from azero.model import HalmaNet
from azero.pipeline import _collect_selfplay

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def count_terminal(data):
    vals = [float(t[2]) for t in data]
    term = sum(1 for v in vals if abs(v) >= 0.99)
    plus = sum(1 for v in vals if v >= 0.99)
    minus = sum(1 for v in vals if v <= -0.99)
    near0 = sum(1 for v in vals if abs(v) < 0.05)
    return len(vals), term, plus, minus, near0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = HalmaNet()
    ck = torch.load(os.path.join(CKPT_DIR, "round_6.pt"), map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval(); m.to(dev)
    print("device=%s  ckpt=round_6.pt" % dev, flush=True)

    # --- Sequential path (workers=1) ---
    print("\n[sequential path, workers=1] endgame_n=20  endgame_n_sims=200  noise=OFF", flush=True)
    data = _collect_selfplay(
        m, n_games=0, device=dev, n_sims=100, max_moves=12, workers=1,
        seed=7, endgame_n=20, endgame_n_sims=200, endgame_add_noise=False,
    )
    n, term, plus, minus, near0 = count_terminal(data)
    print("  samples=%d  true_terminal(>=0.99)=%d  (+1=%d, -1=%d)  near_zero=%d"
          % (n, term, plus, minus, near0), flush=True)
    print("  -> %d/%d samples are genuine +/-1 (was ~0 before the fix)"
          % (term, n), flush=True)

    # --- Production worker path (workers>1, spawn) ---
    print("\n[worker path, workers=2] endgame_n=8  endgame_n_sims=200  noise=OFF", flush=True)
    data2 = _collect_selfplay(
        m, n_games=0, device=dev, n_sims=100, max_moves=12, workers=2,
        seed=7, endgame_n=8, endgame_n_sims=200, endgame_add_noise=False,
    )
    n2, term2, plus2, minus2, near02 = count_terminal(data2)
    print("  samples=%d  true_terminal(>=0.99)=%d  (+1=%d, -1=%d)  near_zero=%d"
          % (n2, term2, plus2, minus2, near02), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
