"""Diagnostic: (1) verify the win condition actually fires, (2) measure how
close random play gets to it.

Win rule (halma8._check_win): attacker wins when the ENEMY base (9 squares)
is completely occupied AND at least one attacker piece is among the occupants.
"""

import random

from halma8 import Halma8, P1, P2, P1_BASE, P2_BASE
from azero.board import ACTION_LIST, action_index, decode_action, legal_action_indices


def base_occupied(s, base):
    return sum(1 for (r, c) in base if s.grid[r][c] != 0)


def base_has_player(s, base, player):
    return any(s.grid[r][c] == player for (r, c) in base)


def test_win_condition():
    # P1 should win: P2's base (top-right 3x3) full + at least one P1 piece.
    s = Halma8()
    # Clear the board, then fill P2 base with 8 P2 pieces + 1 P1 piece.
    s.grid = [[0] * 8 for _ in range(8)]
    base = list(P2_BASE)
    for i, (r, c) in enumerate(base):
        s.grid[r][c] = P1 if i == 0 else P2   # first square = P1 piece, rest P2
    s.turn = P1
    # Manually run the win check.
    s._check_win()
    assert s.winner == P1, "expected P1 win, got %r" % (s.winner,)
    print("win-condition check: P1 correctly wins when P2 base full + 1 P1 piece  [OK]")

    # Near miss: P2 base full but NO P1 piece -> no win.
    s2 = Halma8()
    s2.grid = [[0] * 8 for _ in range(8)]
    for (r, c) in P2_BASE:
        s2.grid[r][c] = P2
    s2._check_win()
    assert s2.winner is None, "near-miss should not be a win, got %r" % (s2.winner,)
    print("win-condition check: full P2 base with no P1 piece is NOT a win  [OK]")

    # Near miss: 8/9 filled + 1 P1 piece (one empty) -> no win.
    s3 = Halma8()
    s3.grid = [[0] * 8 for _ in range(8)]
    base = list(P2_BASE)
    for i, (r, c) in enumerate(base):
        if i == 8:
            continue  # leave one square empty
        s3.grid[r][c] = P1 if i == 0 else P2
    s3._check_win()
    assert s3.winner is None, "8/9 base should not be a win, got %r" % (s3.winner,)
    print("win-condition check: 8/9 filled P2 base is NOT a win  [OK]")


def random_game_stats(rng, n, max_moves):
    stuck = 0
    hit_max = 0
    winners = 0
    max_enemy_base_fill = 0        # max (over the game) fraction of enemy base occupied
    ever_filled_enemy = 0          # did enemy base ever reach 9/9 occupancy?
    for _ in range(n):
        s = Halma8()
        peak = 0
        ended = None
        for _ in range(max_moves):
            idxs = legal_action_indices(s)
            if not idxs:
                ended = "stuck"
                break
            mv = rng.choice([ACTION_LIST[i] for i in idxs])
            if mv[0] == "stop":
                s.end_turn()
            else:
                s.apply_move(decode_action(action_index(mv)))
            # Track enemy-base fill for BOTH sides.
            p2fill = base_occupied(s, P2_BASE)  # P1's target
            p1fill = base_occupied(s, P1_BASE)  # P2's target
            peak = max(peak, p2fill, p1fill)
            if s.winner is not None:
                ended = "winner"
                break
        if ended is None:
            ended = "max_moves"
        max_enemy_base_fill = max(max_enemy_base_fill, peak)
        if peak >= 9:
            ever_filled_enemy += 1
        if ended == "stuck":
            stuck += 1
        elif ended == "winner":
            winners += 1
        else:
            hit_max += 1
    print("N=%d max_moves=%d:  winners=%d  hit_max=%d  stuck=%d" %
          (n, max_moves, winners, hit_max, stuck))
    print("   max enemy-base fill ever seen: %d/9 squares;  games that ever filled an enemy base to 9/9: %d" %
          (max_enemy_base_fill, ever_filled_enemy))


def main():
    test_win_condition()
    print()
    rng = random.Random(123)
    random_game_stats(rng, 200, 100)
    rng = random.Random(124)
    random_game_stats(rng, 200, 400)


if __name__ == "__main__":
    main()
