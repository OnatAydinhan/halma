"""Decisive cheap isolation: is the finish even LEARNABLE by the value head?

Trains the value head directly on endgame positions (value target +1 for the
attacker-to-move, which immediate_win_exists confirms is a real one-move win),
with NO fresh-board dilution and NO self-play. If the held-out value jumps to
~+1, the signal is learnable and the real run's failure is dilution/batching.
If it barely moves, there is a deeper bug.

Before/after, train vs held-out. Seconds, CPU or CUDA. ASCII-only.
Run:  py -u azero/_probe_learn.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np

from azero.board import encode_board
from azero.endgame import immediate_win_exists, make_endgame_positions
from azero.model import HalmaNet, value_loss

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = os.path.join(CKPT_DIR, "round_5.pt")
    print("source=%s  device=%s" % (os.path.basename(src), dev), flush=True)

    train = make_endgame_positions(24, full=8, seed=7)
    test = make_endgame_positions(24, full=8, seed=20261110)  # held-out
    train_ok = sum(1 for st in train if immediate_win_exists(st))
    test_ok = sum(1 for st in test if immediate_win_exists(st))
    print("winnable-in-1: train=%d/24  test(held-out)=%d/24" % (train_ok, len(test)), flush=True)

    enc_t = torch.stack([encode_board(st).squeeze(0) for st in train]).to(dev)
    enc_v = torch.stack([encode_board(st).squeeze(0) for st in test]).to(dev)
    target = torch.full((len(train), 1), 1.0, device=dev)  # +1 for attacker to move

    m = HalmaNet()
    ck = torch.load(src, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def val_mean(enc):
        m.eval()
        with torch.no_grad():
            _, v = m(enc)
        return float(v.cpu().numpy().ravel().mean())

    b_tr, b_te = val_mean(enc_t), val_mean(enc_v)

    m.train()
    for ep in range(150):
        opt.zero_grad()
        _, vals = m(enc_t)
        loss = value_loss(vals, target)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 149:
            m.eval()
            with torch.no_grad():
                _, tv = m(enc_t)
                _, hv = m(enc_v)
            print("  epoch %3d  train_val=%+.4f  held-out_val=%+.4f  loss=%.4f"
                  % (ep, tv.cpu().numpy().ravel().mean(),
                     hv.cpu().numpy().ravel().mean(), float(loss)), flush=True)
            m.train()

    a_tr, a_te = val_mean(enc_t), val_mean(enc_v)
    print("", flush=True)
    print("== before -> after (value head, +1 = win for side to move) ==", flush=True)
    print("  train      : %+.4f -> %+.4f" % (b_tr, a_tr), flush=True)
    print("  held-out   : %+.4f -> %+.4f   (delta=%+.4f)" % (b_te, a_te, a_te - b_te), flush=True)
    print("", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
