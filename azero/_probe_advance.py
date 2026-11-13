"""Diagnostic: does the distance teacher produce measurable ADVANCEMENT?

The win/loss Elo is blind to non-winning progress (every game draws ->
1000.0). This measures the two things directly, using the exact real
move-selection path (mcts_search) and the T-13 distance helpers:

  1. General advancement: how far the pieces' centroids move toward the
     enemy base (adv_total; 14.14 = no advance at all, lower = more advanced).
  2. Win proximity: how many of a player's pieces ever enter the OPPOSING
     base, and how full that base is at the moment it is invaded (a win needs
     the enemy base FULL + an attacker piece inside).

Both nets play the SAME per-game seeds (controlled comparison). ASCII-only
stdout (Windows cp1252). Run:
    py azero/_probe_advance.py --games 12 --sims 100 --max-moves 100 --batch 32
"""
import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from halma8 import Halma8, P1_BASE, P2_BASE
from azero.board import action_index, decode_action
from azero.model import HalmaNet
from azero.mcts import mcts_search
from azero.value_target import (
    P1_BASE_CENTER, P2_BASE_CENTER, centroid_distance, distance_advantage,
)

P1_BASE_SET = set(P1_BASE)
P2_BASE_SET = set(P2_BASE)


def load_net(path, device):
    m = HalmaNet()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    m.to(device)
    return m


def base_fullness(grid, base):
    return sum(1 for (r, c) in base if grid[r][c] != 0) / len(base)


def invasion_count(grid, attacker, enemy_base):
    return sum(1 for (r, c) in enemy_base if grid[r][c] == attacker)


def measure(model, device, seeds, n_sims, max_moves, batch, tag):
    n_pos = 0
    wins = 0
    adv_total_sum = 0.0
    tail_sum = 0.0
    tail_n = 0
    abs_adv_sum = 0.0
    inv_sum = 0.0
    inv_max = 0.0
    occ_when_inv_max = 0.0
    games = 0
    t0 = time.time()
    for gidx, seed in enumerate(seeds):
        state = Halma8()
        tail = []
        for _ in range(max_moves):
            ba, _, _ = mcts_search(
                state, model, n_sims=n_sims, device=device, add_noise=True,
                seed=seed, return_root=True, batch_size=batch,
            )
            if ba is None:
                break
            grid = state.grid
            d1 = centroid_distance(grid, 1, P2_BASE_CENTER)
            d2 = centroid_distance(grid, 2, P1_BASE_CENTER)
            adv_total_sum += d1 + d2
            abs_adv_sum += abs(distance_advantage(grid, state.turn, 4.0))
            # win proximity: attacker pieces inside the enemy base
            i1 = invasion_count(grid, 1, P2_BASE_SET)  # P1 in P2 base
            i2 = invasion_count(grid, 2, P1_BASE_SET)  # P2 in P1 base
            inv = max(i1, i2)
            inv_sum += inv
            if inv > inv_max:
                inv_max = inv
            if i1 >= 1 and base_fullness(grid, P2_BASE_SET) > occ_when_inv_max:
                occ_when_inv_max = base_fullness(grid, P2_BASE_SET)
            if i2 >= 1 and base_fullness(grid, P1_BASE_SET) > occ_when_inv_max:
                occ_when_inv_max = base_fullness(grid, P1_BASE_SET)
            n_pos += 1
            tail.append(d1 + d2)
            if len(tail) > 20:
                tail.pop(0)
            if ba[0] == "stop":
                state.end_turn()
            else:
                state.apply_move(decode_action(action_index(ba)))
            if state.winner is not None:
                wins += 1
                break
        tail_sum += sum(tail)
        tail_n += max(1, len(tail))
        games += 1
        print("[%s] game %d/%d done  (pos=%d)  elapsed=%.0fs"
              % (tag, gidx + 1, len(seeds), n_pos, time.time() - t0), flush=True)

    return {
        "tag": tag,
        "games": games,
        "n_pos": n_pos,
        "wins": wins,
        "adv_total_mean": adv_total_sum / max(1, n_pos),
        "adv_tail_mean": tail_sum / max(1, tail_n),
        "abs_adv_mean": abs_adv_sum / max(1, n_pos),
        "inv_mean": inv_sum / max(1, n_pos),
        "inv_max": inv_max,
        "occ_when_inv_max": occ_when_inv_max,
    }


def report(r):
    print()
    print("== %s ==" % r["tag"])
    print("  games=%d  positions=%d  wins=%d" % (r["games"], r["n_pos"], r["wins"]))
    print("  advancement: mean adv_total  = %.3f  (14.14 = no advance; lower = more advanced)" % r["adv_total_mean"])
    print("  advancement: late-game adv   = %.3f" % r["adv_tail_mean"])
    print("  teacher signal: mean |advantage| = %.4f" % r["abs_adv_mean"])
    print("  win proximity: mean enemy-base invaders/pos = %.4f" % r["inv_mean"])
    print("  win proximity: max invaders in a base       = %.0f" % r["inv_max"])
    print("  win proximity: max base-fullness when invaded = %.2f  (9/9 = win)" % r["occ_when_inv_max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--max-moves", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU")
        device = "cpu"

    ckdir = os.path.join(os.path.dirname(__file__), "checkpoints")
    paths = {
        "round_1": os.path.join(ckdir, "round_1.pt"),
        "round_5": os.path.join(ckdir, "round_5.pt"),
    }
    seeds = list(range(args.games))
    print("device=%s  games/net=%d  sims=%d  max_moves=%d  batch=%d"
          % (device, args.games, args.sims, args.max_moves, args.batch), flush=True)

    results = []
    for tag, path in paths.items():
        if not os.path.exists(path):
            print("MISSING checkpoint: %s" % path)
            continue
        model = load_net(path, device)
        r = measure(model, device, seeds, args.sims, args.max_moves, args.batch, tag)
        results.append(r)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    if len(results) == 2:
        a, b = results  # round_1, round_5
        print()
        print("== round_5 vs round_1 (positive delta = round_5 advances MORE) ==")
        print("  adv_total_mean   : %+.4f   (%.3f -> %.3f)" % (
            b["adv_total_mean"] - a["adv_total_mean"], a["adv_total_mean"], b["adv_total_mean"]))
        print("  inv_mean         : %+.4f   (%.4f -> %.4f)" % (
            b["inv_mean"] - a["inv_mean"], a["inv_mean"], b["inv_mean"]))
        print("  inv_max          : %+.0f   (%.0f -> %.0f)" % (
            b["inv_max"] - a["inv_max"], a["inv_max"], b["inv_max"]))
        print("  occ_when_inv_max : %+.2f   (%.2f -> %.2f)" % (
            b["occ_when_inv_max"] - a["occ_when_inv_max"], a["occ_when_inv_max"], b["occ_when_inv_max"]))
    print()
    print("DONE")


if __name__ == "__main__":
    main()
