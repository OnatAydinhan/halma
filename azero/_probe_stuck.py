"""Cheap sanity probe: is the value head dead, and does the net ever push
toward the win objective (fill the enemy base)?

Loads a checkpoint, plays a few games, and reports:
  - per game: length, outcome, max # of attacker pieces in the enemy base
  - value-target stats (mean |v|, fraction exactly 0)
  - the live model's value-output stats on those positions
  - policy-target entropy + mean legal-action count
  - a one-step loss split (policy_loss vs value_loss)

Run:  py -u azero/_probe_stuck.py [checkpoint.pt]
"""
import os
import sys
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np

from halma8 import Halma8, P1_BASE, P2_BASE
from azero.board import (
    ACTION_LIST, action_index, decode_action, encode_board, legal_action_indices,
)
from azero.model import HalmaNet, policy_loss, value_loss
from azero.mcts import mcts_search

N_ACTIONS = len(ACTION_LIST)
P1, P2 = 1, 2


def count_in(base, attacker):
    return sum(1 for (r, c) in base if grid_val(r, c) == attacker)


def grid_val(r, c):
    return _GRID[r][c]


_GRID = None  # set per game below


def play(model, device, n_sims=60, max_moves=200, seed=0):
    global _GRID
    state = Halma8()
    encs, pols, vals = [], [], []
    max_p1_in_p2base = 0
    max_p2_in_p1base = 0
    for _ in range(max_moves):
        _GRID = state.grid
        max_p1_in_p2base = max(max_p1_in_p2base, count_in(P2_BASE, P1))
        max_p2_in_p1base = max(max_p2_in_p1base, count_in(P1_BASE, P2))

        enc = encode_board(state).squeeze(0)
        best, policy, root = mcts_search(
            state, model, n_sims=n_sims, device=device,
            add_noise=True, seed=seed, return_root=True,
        )
        if best is None:
            break
        dense = root.value_sum / max(root.visits, 1)
        encs.append(enc)
        pols.append(policy)
        vals.append(dense)
        if best[0] == "stop":
            state.end_turn()
        else:
            state.apply_move(decode_action(action_index(best)))
        if state.winner is not None:
            break
        seed += 1
    _GRID = state.grid
    final_p1 = count_in(P2_BASE, P1)
    final_p2 = count_in(P1_BASE, P2)
    if state.winner is not None:
        # positions alternate turn starting from P1
        mover_at = [P1 if i % 2 == 0 else P2 for i in range(len(encs))]
        val_targets = [1.0 if m == state.winner else -1.0 for m in mover_at]
    else:
        val_targets = vals
    return {
        "n": len(encs), "winner": state.winner,
        "max_p1_in_p2base": max_p1_in_p2base, "max_p2_in_p1base": max_p2_in_p1base,
        "final_p1": final_p1, "final_p2": final_p2,
        "encs": encs, "pols": pols, "vals": val_targets,
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "azero/checkpoints/round_3.pt"
    model = HalmaNet()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    model.to(dev)
    print(f"probe: {ckpt_path}  device={dev}", flush=True)

    all_encs, all_pols, all_vals = [], [], []
    for g in range(3):
        r = play(model, dev, n_sims=60, max_moves=200, seed=1000 + g)
        all_encs += r["encs"]; all_pols += r["pols"]; all_vals += r["vals"]
        print(
            f"  game{g}: moves={r['n']} winner={'P1' if r['winner']==1 else 'P2' if r['winner']==2 else 'DRAW'}"
            f"  max_red_in_bluebase={r['max_p1_in_p2base']}  max_blue_in_redbase={r['max_p2_in_p1base']}"
            f"  final(red,blue)=({r['final_p1']},{r['final_p2']})",
            flush=True,
        )

    v = np.array(all_vals, dtype=np.float64)
    print(f"\nVALUE TARGETS  n={len(v)}  mean|v|={np.abs(v).mean():.4f}  frac_zero={np.mean(v==0):.3f}", flush=True)

    # live model value outputs
    with torch.no_grad():
        states = torch.stack(all_encs).to(dev)
        logits, vout = model(states)
        vout = vout.cpu().numpy().ravel()
    print(f"MODEL VALUE  mean|v|={np.abs(vout).mean():.4f}  std={vout.std():.4f}  min={vout.min():.3f} max={vout.max():.3f}", flush=True)

    # policy target entropy + legal count
    Hs, Ls = [], []
    for p in all_pols:
        p = np.asarray(p, dtype=np.float64)
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        nz = p[p > 0]
        Hs.append(-(nz * np.log(nz)).sum())
        Ls.append(len(nz))
    print(f"POLICY TARGET  mean_entropy={np.mean(Hs):.3f}  mean_legal={np.mean(Ls):.1f}  (ln(uniform 417)={math.log(N_ACTIONS):.3f})", flush=True)

    # one-step loss split
    bs = min(64, len(all_encs))
    S = torch.stack(all_encs[:bs]).to(dev)
    P = torch.tensor(np.stack(all_pols[:bs]), dtype=torch.float32).to(dev)
    V = torch.tensor(all_vals[:bs], dtype=torch.float32).view(-1, 1).to(dev)
    with torch.no_grad():
        lg, vp = model(S)
        pl = policy_loss(lg, P)
        vl = value_loss(vp, V)
    print(f"LOSS SPLIT     policy_loss={pl.item():.4f}  value_loss={vl.item():.4f}  total={ (pl+vl).item():.4f}", flush=True)


if __name__ == "__main__":
    main()
