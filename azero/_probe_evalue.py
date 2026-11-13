"""Does an ENDGAME self-play game (near-win start) produce a true +1 value target?

Plays near-win positions through the REAL self_play path (play_one_game) with
the synthesized position actually passed in as `start_state`, and prints, per
(noise, n_sims) setting: how many convert to a genuine +1 target.

This is the T-14 curriculum gate: if the net can convert these (win-in-1)
positions, the endgame games are a real +/-1 signal and un-stick the value
head. If they do NOT convert (all value targets stay ~0), the chicken-and-egg
deadlock is still active even for the curriculum.

Also verifies each synthesized position is a genuine one-move win via
`immediate_win_exists`, so a failure is a search problem, not a bad position.

Seconds. ASCII-only. Run:  py -u azero/_probe_evalue.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from azero.endgame import immediate_win_exists, make_endgame_positions
from azero.model import HalmaNet
from azero.selfplay import play_one_game

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def _pick_ckpt():
    for name in ("round_6.pt", "round_5.pt", "final.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            return p
    raise SystemExit("no checkpoint found in %s" % CKPT_DIR)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _pick_ckpt()
    m = HalmaNet()
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    m.to(dev)
    print("device=%s  ckpt=%s" % (dev, os.path.basename(ckpt)), flush=True)

    pos = make_endgame_positions(10, full=8, seed=7)
    n_win1 = sum(1 for st in pos if immediate_win_exists(st))
    print("positions=%d  immediate_win_exists=%d/10  (must be 10/10 or the" % (len(pos), n_win1), flush=True)
    print("positions are the problem, not the search)", flush=True)

    for noisy in (False, True):
        for n_sims in (30, 60, 200):
            n_plus1 = 0
            n_multi = 0
            for i, st in enumerate(pos):
                tuples = play_one_game(m, device=dev, n_sims=n_sims, max_moves=10,
                                       add_noise=noisy, seed=100 + i, start_state=st)
                vals = [round(float(t[2]), 3) for t in tuples]
                if len(tuples) == 1 and abs(vals[0] - 1.0) < 1e-6:
                    n_plus1 += 1
                elif len(vals) > 1:
                    n_multi += 1
            print("  noise=%-5s sims=%3d : converted_to_+1=%d  multi=%d / %d"
                  % ("on" if noisy else "off", n_sims, n_plus1, n_multi, len(pos)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
