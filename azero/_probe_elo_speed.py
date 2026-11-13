"""A/B probe: why the Elo eval is slow.

Measures the *only* thing that differs between the fast self-play path and the
slow elo.py path: MCTS `batch_size`. Plays the first N moves of a real game at
n_sims=100 with batch_size=1 (what elo.py currently does) and batch_size=32
(what run.py self-play does), and reports per-move time + a 400x240 extrapolation.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from halma8 import Halma8
from azero.model import HalmaNet
from azero.board import action_index, decode_action
from azero.mcts import mcts_search

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path, dev):
    m = HalmaNet()
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    m.to(dev)
    return m


def play_n_moves(model, n_moves, n_sims, batch_size, dev):
    """Play n_moves from the start; return (per_move_s, total_s)."""
    state = Halma8()
    t0 = time.time()
    for _ in range(n_moves):
        action, _ = mcts_search(
            state, model, n_sims=n_sims, device=dev, add_noise=False,
            batch_size=batch_size,
        )
        if action is None or action[0] == "stop":
            state.end_turn()
        else:
            state.apply_move(decode_action(action_index(action)))
        if state.winner is not None:
            break
    return (time.time() - t0) / n_moves, (time.time() - t0)


def main():
    base = os.path.join(_HERE, "checkpoints")
    model = load(os.path.join(base, "round_2.pt"), DEV)

    N_MOVES = 8
    N_SIMS = 100
    games, moves = 400, 240
    workers = 8

    print(f"device={DEV}  n_moves={N_MOVES}  n_sims={N_SIMS}")
    results = {}
    for bs in (1, 32):
        pm, total = play_n_moves(model, N_MOVES, N_SIMS, bs, DEV)
        results[bs] = pm
        # full eval upper bound: games*moves moves, /workers
        full_s = games * moves * pm / workers
        print(f"  batch={bs:<3}  per_move={pm*1000:8.1f} ms   "
              f"400x240/{workers}w ~ {full_s/60:8.1f} min (all-draw worst case)")

    speedup = results[1] / results[32]
    print(f"  => batch 1 -> 32 speedup: {speedup:.1f}x per move")


if __name__ == "__main__":
    main()
