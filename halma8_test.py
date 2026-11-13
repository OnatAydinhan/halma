import unittest
import halma8 as halmag
from halma8 import P1, P2, base_cells


class T1Test(unittest.TestCase):
    def test_base_cells(self):
        self.assertEqual(base_cells(P1), {(r, c) for r in range(5, 8) for c in range(3)})
        self.assertEqual(base_cells(P2), {(r, c) for r in range(3) for c in range(5, 8)})

    def test_new_board_placement(self):
        g = halmag.Halma8()
        self.assertEqual(len(g.grid), 8)
        self.assertTrue(all(len(row) == 8 for row in g.grid))
        for (r, c), want in [((5, 0), P1), ((7, 2), P1), ((0, 5), P2), ((2, 7), P2), ((4, 4), 0)]:
            self.assertEqual(g.grid[r][c], want, f"cell {(r, c)}")
        for b, owner in [(halmag.base_cells(P1), P1), (halmag.base_cells(P2), P2)]:
            for (r, c) in b:
                self.assertEqual(g.grid[r][c], owner)
        empty = sum(v == 0 for row in g.grid for v in row)
        self.assertEqual(empty, 64 - 18)

    def test_counts(self):
        g = halmag.Halma8()
        self.assertEqual(sum(v == P1 for row in g.grid for v in row), 9)
        self.assertEqual(sum(v == P2 for row in g.grid for v in row), 9)

    def test_initial_state(self):
        g = halmag.Halma8()
        self.assertEqual(g.turn, P1)
        self.assertIsNone(g.winner)
        self.assertEqual(g.turn_no, 1)

    def test_illegal_move_is_valueerror(self):
        self.assertTrue(issubclass(halmag.IllegalMove, ValueError))

    def test_names(self):
        self.assertEqual(halmag.NAME[P1], "P1 (Red)")
        self.assertEqual(halmag.NAME[P2], "P2 (Blue)")


def make_board(pieces):
    """Empty 8x8 board with only the given (r,c)->player pieces placed."""
    g = halmag.Halma8()
    g.grid = [[0] * 8 for _ in range(8)]
    for (r, c), p in pieces.items():
        g.grid[r][c] = p
    return g


class T2Tests(unittest.TestCase):
    def test_initial_board_step_counts(self):
        g = halmag.Halma8()
        # (5,2) corner of P1 base: open to (4,2) and (5,3) per spec rule
        self.assertEqual(sorted(g.legal_steps(5, 2)), sorted([(4, 2), (5, 3)]))
        # (6,1) fully boxed by own pieces: no steps
        self.assertEqual(g.legal_steps(6, 1), [])

    def test_isolated_jump_right(self):
        g = make_board({(4, 4): P1, (4, 5): P2})
        self.assertEqual(g.legal_jumps(4, 4), [((4, 5), (4, 6))])

    def test_jump_over_own_piece_allowed(self):
        g = make_board({(4, 4): P1, (4, 5): P1})
        self.assertEqual(g.legal_jumps(4, 4), [((4, 5), (4, 6))])

    def test_jump_over_own_piece_from_other_side(self):
        g = make_board({(4, 6): P1, (4, 5): P1})
        self.assertEqual(g.legal_jumps(4, 6), [((4, 5), (4, 4))])

    def test_no_jump_when_middle_empty(self):
        g = make_board({(4, 4): P1, (4, 6): P2})
        self.assertEqual(g.legal_jumps(4, 4), [])

    def test_no_jump_when_destination_occupied(self):
        g = make_board({(4, 4): P1, (4, 5): P2, (4, 6): P2})
        self.assertEqual(g.legal_jumps(4, 4), [])

    def test_jump_bounded_by_board_edges(self):
        g = make_board({(0, 4): P1, (0, 5): P2})  # up over (0,5) -> (0,6) ok; down would leave board
        self.assertEqual(g.legal_jumps(0, 4), [((0, 5), (0, 6))])
        g2 = make_board({(0, 4): P1, (1, 4): P2})  # down over (1,4) -> (2,4) ok; up off-board
        self.assertEqual(g2.legal_jumps(0, 4), [((1, 4), (2, 4))])
        g3 = make_board({(7, 4): P1, (7, 5): P2})  # right over (7,5) -> (7,6) ok; ...
        self.assertEqual(g3.legal_jumps(7, 5), [((7, 4), (7, 3))])
        g4 = make_board({(7, 6): P1, (7, 7): P2})  # right off board
        self.assertEqual(g4.legal_jumps(7, 6), [])

    def test_empty_square_returns_empty_lists(self):
        g = make_board({(4, 4): P1})
        self.assertEqual(g.legal_steps(3, 3), [])
        self.assertEqual(g.legal_jumps(3, 3), [])

    def test_out_of_range_returns_empty_lists(self):
        g = halmag.Halma8()
        for rc in [(-1, 0), (0, -1), (8, 8), (5, 9)]:
            self.assertEqual(g.legal_steps(*rc), [])
            self.assertEqual(g.legal_jumps(*rc), [])

    def test_legal_moves_are_pure_board_functions(self):
        g = make_board({(4, 4): P1, (4, 5): P2})
        before = (g.legal_steps(4, 4), g.legal_jumps(4, 4))
        g.turn = P2  # pure of board state — turn must not matter
        self.assertEqual((g.legal_steps(4, 4), g.legal_jumps(4, 4)), before)
        # spot-check the ground truth on this small position
        self.assertEqual(sorted(g.legal_steps(4, 4)), [(3, 4), (4, 3), (5, 4)])
        self.assertEqual(g.legal_jumps(4, 5), [((4, 4), (4, 3))])  # over red (4,4), landing (4,3)


class T3Tests(unittest.TestCase):
    def test_step_applies_and_flips_turn(self):
        g = halmag.Halma8()
        self.assertFalse(g.apply_move(("step", (5, 2), (4, 2))))
        self.assertEqual(g.grid[4][2], P1)
        self.assertEqual(g.grid[5][2], 0)
        self.assertEqual(g.turn, P2)
        self.assertEqual(g.turn_no, 2)

    def test_jump_chain_continues_until_end_turn(self):
        # Chain-revisit ban (T3.5): after each jump, the landing square and the
        # origin are banned as continuation targets. A 2-jump chain: red (4,1),
        # blues (4,2) and (4,4).
        g = make_board({(4, 1): P1, (4, 2): P2, (4, 4): P2})
        self.assertTrue(g.apply_move(("jump", (4, 1), (4, 2), (4, 3))))
        self.assertEqual(g.grid[4][3], P1)
        self.assertEqual(g.grid[4][1], 0)
        self.assertEqual(g.turn, P1)    # chain in progress: turn untouched
        self.assertEqual(g.turn_no, 1)
        self.assertFalse(g.apply_move(("jump", (4, 3), (4, 4), (4, 5))))
        self.assertEqual(g.grid[4][5], P1)
        self.assertEqual(g.grid[4][3], 0)
        self.assertEqual(g.turn, P2)    # final jump (only continuation was the banned origin)
        self.assertEqual(g.turn_no, 2)

    def test_banned_destination_raises_when_chain_still_open(self):
        # Red (4,1), blues (4,2),(4,4),(4,6). After the 1st jump to (4,3), the
        # chain is still open (jump over (4,4) to (4,5) exists). Attempting the
        # geometrically-valid back-jump over (4,2) to the banned origin (4,1)
        # must raise IllegalMove, not be accepted.
        g = make_board({(4, 1): P1, (4, 2): P2, (4, 4): P2, (4, 6): P2})
        g.turn = P1
        self.assertTrue(g.apply_move(("jump", (4, 1), (4, 2), (4, 3))))
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("jump", (4, 3), (4, 2), (4, 1)))
        # position unchanged by the rejected move
        self.assertEqual(g.grid[4][3], P1)
        self.assertEqual(g.grid[4][1], 0)
        self.assertTrue(g.chain_active)

    def test_three_jump_chain(self):
        # Red (4,1); blues (4,2),(4,4),(4,6); (4,7) empty. Guards that each
        # new landing square is added to the ban set, not just the origin.
        g = make_board({(4, 1): P1, (4, 2): P2, (4, 4): P2, (4, 6): P2})
        g.turn = P1
        self.assertTrue(g.apply_move(("jump", (4, 1), (4, 2), (4, 3))))
        self.assertEqual(g.turn, P1)
        self.assertEqual(g.turn_no, 1)
        self.assertTrue(g.apply_move(("jump", (4, 3), (4, 4), (4, 5))))
        self.assertEqual(g.turn, P1)
        self.assertEqual(g.turn_no, 1)
        self.assertFalse(g.apply_move(("jump", (4, 5), (4, 6), (4, 7))))
        self.assertEqual(g.grid[4][7], P1)
        self.assertEqual(g.turn, P2)
        self.assertEqual(g.turn_no, 2)
        self.assertFalse(g.chain_active)

    def test_single_jump_ending_turn(self):
        # Isolated red (4,4), blue (4,5). After the jump to (4,6), the only
        # candidate continuation (back over (4,5) to the origin (4,4)) is banned,
        # so this is a final jump: returns False, turn flips to P2.
        g = make_board({(4, 4): P1, (4, 5): P2})
        g.turn = P1
        self.assertFalse(g.apply_move(("jump", (4, 4), (4, 5), (4, 6))))
        self.assertEqual(g.turn, P2)
        self.assertEqual(g.turn_no, 2)
        self.assertEqual(g.grid[4][6], P1)
        self.assertEqual(g.grid[4][4], 0)

    def test_end_turn_only_flips(self):
        g = halmag.Halma8()
        before = [row[:] for row in g.grid]
        g.end_turn()
        self.assertEqual(g.grid, before)
        self.assertEqual(g.turn, P2)
        self.assertEqual(g.turn_no, 2)

    def test_step_into_occupied_raises(self):
        g = halmag.Halma8()
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("step", (6, 0), (6, 1)))
        # board unchanged, turn unchanged
        self.assertEqual(g.grid[6][0], P1)
        self.assertEqual(g.turn, P1)
        self.assertEqual(g.turn_no, 1)

    def test_jump_with_empty_middle_raises(self):
        g = make_board({(4, 4): P1})
        g.turn = P1
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("jump", (4, 4), (4, 5), (4, 6)))
        self.assertEqual(g.grid[4][4], P1)
        self.assertEqual(g.turn, P1)

    def test_jump_with_occupied_destination_raises(self):
        g = make_board({(4, 4): P1, (4, 5): P2, (4, 6): P2})
        g.turn = P1
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("jump", (4, 4), (4, 5), (4, 6)))
        self.assertEqual(g.grid[4][4], P1)

    def test_move_off_board_raises(self):
        g = make_board({(7, 0): P1})
        g.turn = P1
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("step", (7, 0), (8, 0)))
        g2 = halmag.Halma8()
        with self.assertRaises(halmag.IllegalMove):
            g2.apply_move(("step", (5, 2), (4, 3)))
        self.assertEqual(g2.grid[5][2], P1)
        self.assertEqual(g2.turn, P1)

    def test_move_after_winner_is_set_raises(self):
        g = halmag.Halma8()
        g.winner = P2
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("step", (5, 2), (4, 2)))
        g2 = make_board({(4, 4): P1})
        g2.turn = P1
        g2.winner = P1
        with self.assertRaises(halmag.IllegalMove):
            g2.apply_move(("jump", (4, 4), (4, 5), (4, 6)))

    def test_end_turn_no_op_when_winner_set(self):
        g = halmag.Halma8()
        g.winner = P1
        g.end_turn()
        self.assertEqual(g.turn, P1)
        self.assertEqual(g.turn_no, 1)

    def test_illegal_move_is_valueerror(self):
        g = halmag.Halma8()
        with self.assertRaises(ValueError):
            g.apply_move(("step", (5, 2), (4, 3)))


class T4Tests(unittest.TestCase):
    """Win detection — "occupied space" rule (T4)."""

    def test_p1_wins_pure(self):
        """8 red pieces in P2's base, (2,5) empty; red at (2,4) steps in -> P1 wins."""
        pieces = {
            (0, 5): P1, (0, 6): P1, (0, 7): P1,
            (1, 5): P1, (1, 6): P1, (1, 7): P1,
            (2, 6): P1, (2, 7): P1,
            # (2,5) intentionally empty
            (2, 4): P1,
        }
        g = make_board(pieces)
        self.assertFalse(g.apply_move(("step", (2, 4), (2, 5))))
        self.assertEqual(g.winner, P1)
        # turn did NOT flip — game is over, end_turn is a no-op
        self.assertEqual(g.turn, P1)

    def test_p1_wins_mixed_with_parked_blue(self):
        """7 red + 1 blue parked at (2,5) in P2's base; empty (2,7).
        Red at (3,7) steps in -> P1 wins despite the blue piece in the base."""
        pieces = {
            (0, 5): P1, (0, 6): P1, (0, 7): P1,
            (1, 5): P1, (1, 6): P1, (1, 7): P1,
            (2, 5): P2,   # blue parked in P2's own base
            (2, 6): P1,
            # (2, 7) empty
            (3, 7): P1,
        }
        g = make_board(pieces)
        self.assertFalse(g.apply_move(("step", (3, 7), (2, 7))))
        self.assertEqual(g.winner, P1)
        self.assertEqual(g.turn, P1)

    def test_no_false_positive_at_8_of_9(self):
        """8 of 9 cells in P2's base occupied (mixed), 9th empty, no move made."""
        pieces = {
            (0, 5): P1, (0, 6): P1, (0, 7): P1,
            (1, 5): P2, (1, 6): P1, (1, 7): P1,
            (2, 5): P1, (2, 6): P1,
            # (2, 7) empty
        }
        g = make_board(pieces)
        g._check_win()
        self.assertIsNone(g.winner)

    def test_p2_wins_pure(self):
        """8 blue in P1's base, (7,2) empty; blue at (7,3) steps in -> P2 wins."""
        pieces = {
            (5, 0): P2, (5, 1): P2, (5, 2): P2,
            (6, 0): P2, (6, 1): P2, (6, 2): P2,
            (7, 0): P2, (7, 1): P2,
            # (7, 2) empty
            (7, 3): P2,
        }
        g = make_board(pieces)
        self.assertFalse(g.apply_move(("step", (7, 3), (7, 2))))
        self.assertEqual(g.winner, P2)

    def test_p2_wins_mixed(self):
        """7 blue + 1 red at (7,0) in P1's base; empty (7,2).
        Blue at (7,3) steps in -> P2 wins despite the red piece in the base."""
        pieces = {
            (5, 0): P2, (5, 1): P2, (5, 2): P2,
            (6, 0): P2, (6, 1): P2, (6, 2): P2,
            (7, 0): P1,   # red parked in P1's own base
            (7, 1): P2,
            # (7, 2) empty
            (7, 3): P2,
        }
        g = make_board(pieces)
        self.assertFalse(g.apply_move(("step", (7, 3), (7, 2))))
        self.assertEqual(g.winner, P2)
        # turn did not flip — game already over, end_turn is a no-op
        self.assertEqual(g.turn, P1)

    def test_move_after_winner_raises(self):
        """Once the win fires, no further moves are accepted; end_turn is a no-op."""
        # 8 cells of P2's base filled with red; (2,5) empty; red at (2,4) steps in
        pieces = {
            (0, 5): P1, (0, 6): P1, (0, 7): P1,
            (1, 5): P1, (1, 6): P1, (1, 7): P1,
            (2, 6): P1, (2, 7): P1,
            # (2,5) intentionally empty — will be filled by the step
            (2, 4): P1,
        }
        g = make_board(pieces)
        g.apply_move(("step", (2, 4), (2, 5)))
        self.assertEqual(g.winner, P1)
        # any subsequent move must raise
        g.grid[3][3] = P1
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("step", (3, 3), (3, 4)))
        # end_turn remains a no-op
        g.end_turn()
        self.assertEqual(g.turn_no, 1)

    def test_winning_jump_with_open_continuation_ends_chain(self):
        # P2's base (rows 0-2, cols 5-7) has 8 red pieces, (2,6) empty. A red
        # jump from (4,6) over (3,6) fills (2,6) -> P1 wins. The winning piece
        # still has a geometrically open jump (to (2,4)), but the win must CLOSE
        # the chain: apply_move returns False, the chain is deactivated, the turn
        # does not flip, and no continuation is accepted afterwards.
        pieces = {
            (0, 5): P1, (0, 6): P1, (0, 7): P1,
            (1, 5): P1, (1, 6): P1, (1, 7): P1,
            (2, 5): P1, (2, 7): P1,          # (2,6) empty
            (3, 6): P1,                       # jump middle
            (4, 6): P1,                       # jumping piece (origin)
        }
        g = make_board(pieces)
        g.turn = P1
        self.assertFalse(g.apply_move(("jump", (4, 6), (3, 6), (2, 6))))
        self.assertEqual(g.winner, P1)
        self.assertFalse(g.chain_active)      # chain closed by the win
        self.assertEqual(g.grid[2][6], P1)    # winning piece in place
        self.assertEqual(g.grid[4][6], 0)     # origin now empty
        self.assertEqual(g.turn, P1)          # turn did not flip (game over)
        self.assertEqual(g.turn_no, 1)        # turn_no unchanged
        with self.assertRaises(halmag.IllegalMove):
            g.apply_move(("jump", (2, 6), (2, 5), (2, 4)))   # continuation rejected


if __name__ == "__main__":
    unittest.main()
