"""Cheap, MCTS-free probe: does the net's VALUE head recognize a near-win as a
win for the side to move, and does the endgame round (round_6) actually change
that vs round_5 (pre-endgame) and baseline (untrained)?

For each held-out near-win position (attacker to move, base full/8 + invader)
we do ONE forward pass and read:
  - value head output (expect ~+1 if the net knows it is about to win)
  - policy mass on the actual 1-move winning action (vs uniform 1/417)
and a fresh-board control (value should be ~0).

No self-play, no MCTS, no training -> a few seconds. ASCII-only stdout.
Run:  py -u azero/_probe_finish.py [ckpt1 ckpt2 ...]   (defaults: baseline, round_5, round_6)
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np

from halma8 import Halma8
from azero.board import (
    ACTION_LIST, action_index, decode_action, encode_board,
    halma_clone, legal_action_indices,
)
from azero.endgame import immediate_win_exists, make_endgame_positions
from azero.model import HalmaNet

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load(path, dev):
    m = HalmaNet()
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    m.to(dev)
    return m


def find_win_action(st):
    """Index of the single legal action that wins for st.turn, or None."""
    for idx in legal_action_indices(st):
        a = ACTION_LIST[idx]
        trial = halma_clone(st)
        if a[0] == "stop":
            trial.end_turn()
        else:
            trial.apply_move(decode_action(idx))
        if trial.winner is not None and trial.winner == st.turn:
            return idx
    return None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 20261110          # same held-out base finish_rate uses
    n = 24
    positions = make_endgame_positions(n, full=8, seed=seed)
    n_win1 = sum(1 for st in positions if immediate_win_exists(st))
    print("device=%s  held-out seed=%d  n=%d  winnable-in-1=%d/%d"
          % (dev, seed, n, n_win1, len(positions)), flush=True)

    encs = torch.stack([encode_board(st).squeeze(0) for st in positions]).to(dev)
    fresh = encode_board(Halma8()).to(dev)          # (1,7,8,8) control
    uniform = 1.0 / len(ACTION_LIST)

    ckpts = sys.argv[1:] or [
        os.path.join(CKPT_DIR, "baseline.pt"),
        os.path.join(CKPT_DIR, "round_5.pt"),
        os.path.join(CKPT_DIR, "round_6.pt"),
    ]

    last = {}
    for path in ckpts:
        if not os.path.exists(path):
            print("--- %s: MISSING" % os.path.basename(path), flush=True)
            continue
        tag = os.path.basename(path)
        m = load(path, dev)
        with torch.no_grad():
            logits, vals = m(encs)
            _, fval = m(fresh)
            probs = torch.softmax(logits, dim=-1)
        v = vals.cpu().numpy().ravel()
        p = probs.cpu().numpy()
        win_mass = []
        for i, st in enumerate(positions):
            wi = find_win_action(st)
            if wi is not None:
                win_mass.append(p[i, wi])
        wm = float(np.mean(win_mass)) if win_mass else float("nan")
        fv = float(fval.cpu().numpy().ravel()[0])
        last[tag] = (float(v.mean()), wm)
        print("--- %s ---" % tag, flush=True)
        print("  near-win value (attacker to move): mean=%+.4f  std=%.4f  frac>0.3=%.3f"
              % (v.mean(), v.std(), float((v > 0.3).mean())), flush=True)
        print("  fresh-board value (control):       %+.4f" % fv, flush=True)
        print("  policy mass on 1-move win act:     %.5f  (uniform=%.5f)  n=%d"
              % (wm, uniform, len(win_mass)), flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    tags = [os.path.basename(p) for p in ckpts if os.path.exists(p)]
    if len(tags) >= 2:
        a, b = last.get(tags[-2]), last.get(tags[-1])
        if a and b:
            print("", flush=True)
            print("== %s vs %s (near-win value mean, policy-win mass) ==" % (tags[-1], tags[-2]), flush=True)
            print("  value mean delta: %+.4f  (%+.4f -> %+.4f)" % (b[0] - a[0], a[0], b[0]), flush=True)
            print("  win-act mass delta: %+.5f  (%.5f -> %.5f)" % (b[1] - a[1], a[1], b[1]), flush=True)
    print("", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
