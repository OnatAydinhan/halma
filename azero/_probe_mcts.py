"""Confirm the MCTS select-sign hypothesis on a guaranteed 1-move win.

From a near-win position (attacker to move), dump the root's children ranked by
visits with their stored value_sum, and identify where the actual 1-move
winning action sits. Hypothesis: the winning action has the most-negative
value_sum (best for the mover) yet the FEWEST visits, because select() picks
max value_sum (which is in the child-mover perspective -> it picks the worst
move for the side to move).

Seconds. ASCII-only. Run:  py -u azero/_probe_mcts.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from azero.board import ACTION_LIST
from azero.endgame import make_endgame_positions
from azero.model import HalmaNet
from azero.mcts import mcts_search


def find_win_action(st):
    from azero.board import decode_action, halma_clone, legal_action_indices
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
    m = HalmaNet()
    ck = torch.load(os.path.join(os.path.dirname(__file__), "checkpoints", "round_5.pt"),
                    map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    m.to(dev)

    st = make_endgame_positions(1, full=8, seed=7)[0]
    wi = find_win_action(st)
    print("device=%s" % dev, flush=True)
    print("1-move winning action: idx=%d  %s" % (wi, ACTION_LIST[wi]), flush=True)

    best, pol, root = mcts_search(
        st, m, n_sims=60, device=dev, add_noise=False, seed=0, return_root=True,
    )
    best_i = None
    if best is not None:
        best_i = ACTION_LIST.index(best)
    print("MCTS best_action: %s  (idx=%s)" % (best, best_i), flush=True)

    ch = sorted(root.children.items(), key=lambda kv: -kv[1].visits)
    print("\nroot children by visits (value_sum in child-mover perspective):", flush=True)
    for idx, node in ch[:10]:
        mark = "  <-- 1-MOVE WIN" if idx == wi else ""
        print("  idx=%3d  visits=%3d  value_sum=%+.3f  %s%s"
              % (idx, node.visits, node.value_sum, ACTION_LIST[idx], mark), flush=True)

    win_node = root.children.get(wi)
    if win_node is not None:
        print("\nwinning action: visits=%d  value_sum=%+.3f" % (win_node.visits, win_node.value_sum), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
