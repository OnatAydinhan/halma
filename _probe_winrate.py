"""Fast probe: how often does RANDOM play (both sides) reach a winner,
and at what move count? No NN involved -- pure halma8 + random legal moves.

Purpose: estimate the upper bound on self-play sample yield. If even pure
random play rarely produces a winner within max_moves, then a random-init
NN (about as coherent) will yield ~0 training samples, making a launch pointless.
"""

import random

from halma8 import Halma8
from azero.board import ACTION_LIST, action_index, decode_action, legal_action_indices


def one_game(rng, max_moves):
    """Play with random legal moves (via azero.board's rule-aware API).

    Returns (winner, moves_made).
    """
    s = Halma8()
    moves = 0
    for _ in range(max_moves):
        idxs = legal_action_indices(s)
        if not idxs:
            break
        mv = rng.choice([ACTION_LIST[i] for i in idxs])
        if mv[0] == "stop":
            s.end_turn()
        else:
            s.apply_move(decode_action(action_index(mv)))
        moves += 1
        if s.winner is not None:
            break
    return s.winner, moves


def main():
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        n = 400
        winners = 0
        length_at_win = []
        for _ in range(n):
            w, m = one_game(rng, max_moves=100)
            if w is not None:
                winners += 1
                length_at_win.append(m)
        print("max_moves=100: winners=%d/%d (%.3f)  win-lengths(min/med/max)=%s"
              % (winners, n, winners / n,
                 (min(length_at_win) if length_at_win else "-",
                  sorted(length_at_win)[len(length_at_win)//2] if length_at_win else "-",
                  max(length_at_win) if length_at_win else "-")))
        # Bigger budget: how many finish within 300?
        rng2 = random.Random(seed)
        winners300 = 0
        for _ in range(n):
            w, m = one_game(rng2, max_moves=300)
            if w is not None:
                winners300 += 1
        print("max_moves=300: winners=%d/%d (%.3f)" % (winners300, n, winners300 / n))


if __name__ == "__main__":
    main()
