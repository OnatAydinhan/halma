"""Per-position diagnosis: why do some near-wins convert and others not?

For each of 10 synthesized near-win positions (noise OFF, the training-relevant
setting), report at several sim budgets:
  - the net's masked-softmax prior on a winning action (low prior -> starved)
  - MCTS best_action == a winning action?   (does search pick the finish)
  - the winning child's visit count         (0 = never explored)

This reconciles the single-position result (pos0 always converts) with the
all-10 result (conversion varies and worsens with sims on round_6).

Seconds. ASCII-only. Run:  py -u azero/_probe_conv.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from azero.board import (
    ACTION_LIST, action_index, decode_action, encode_board, halma_clone,
    legal_action_indices,
)
from azero.endgame import make_endgame_positions
from azero.model import HalmaNet
from azero.mcts import mcts_search

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def win_action_indices(st):
    out = []
    for idx in legal_action_indices(st):
        a = ACTION_LIST[idx]
        trial = halma_clone(st)
        if a[0] == "stop":
            trial.end_turn()
        else:
            trial.apply_move(decode_action(action_index(a)))
        if trial.winner is not None and trial.winner == st.turn:
            out.append(idx)
    return out


def net_prior_on_win(model, st, dev):
    enc = encode_board(st).squeeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits, values = model(enc)
    legal = legal_action_indices(st)
    mask = torch.full((logits.shape[-1],), float("-inf"), device=logits.device)
    for i in legal:
        mask[i] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)[0]
    wins = win_action_indices(st)
    p = max(float(probs[i].item()) for i in wins) if wins else 0.0
    return p, float(values[0].item())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = HalmaNet()
    ck = torch.load(os.path.join(CKPT_DIR, "round_6.pt"), map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval(); m.to(dev)
    print("device=%s  ckpt=round_6.pt" % dev, flush=True)

    pos = make_endgame_positions(10, full=8, seed=7)
    priors = [net_prior_on_win(m, st, dev) for st in pos]

    for noisy in (False, True):
        for n_sims in (30, 60, 200):
            conv = 0
            visits = []
            for st in pos:
                best, _pol = mcts_search(halma_clone(st), m, n_sims=n_sims, device=dev,
                                        add_noise=noisy, seed=123, return_root=False)
                # best is the argmax-visit action; check it is a win.
                wi_set = set(win_action_indices(st))
                best_i = action_index(best) if best is not None else -1
                ok = best_i in wi_set
                if ok:
                    conv += 1
            print("  noise=%-5s sims=%3d : best_is_win=%d/10"
                  % ("on" if noisy else "off", n_sims, conv), flush=True)

    print("\n  net prior on the winning action (off-noise) per position:", flush=True)
    for i, (p, v) in enumerate(priors):
        print("    pos%2d: prior_on_win=%.4f  net_value=%+.3f  n_win_actions=%d"
              % (i, p, v, len(win_action_indices(pos[i]))), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
