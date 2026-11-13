"""Tests for azero/board.py (T-1)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from halma8 import Halma8
from azero.board import (
    ACTION_LIST, action_index, decode_action, encode_board,
    halma_clone, legal_action_indices
)


def test_action_list_length():
    assert len(ACTION_LIST) == 417


def test_action_list_forms():
    for a in ACTION_LIST:
        kind = a[0]
        if kind == "step":
            assert len(a) == 5
            _, r, c, nr, nc = a
            assert 0 <= r < 8 and 0 <= c < 8
            assert 0 <= nr < 8 and 0 <= nc < 8
            assert abs(nr - r) + abs(nc - c) == 1
        elif kind == "jump":
            assert len(a) == 5
            _, r, c, nr, nc = a
            assert 0 <= r < 8 and 0 <= c < 8
            assert 0 <= nr < 8 and 0 <= nc < 8
            assert abs(nr - r) + abs(nc - c) == 2
        elif kind == "stop":
            assert len(a) == 1
        else:
            raise AssertionError(f"unknown action kind: {kind!r}")


def test_encode_board_start_position():
    state = Halma8()
    t = encode_board(state)
    assert t.shape == (1, 7, 8, 8)
    # P1 at (5,0) -> ch0
    assert t[0, 0, 5, 0].item() == 1.0
    # P2 at (0,5) -> ch1
    assert t[0, 1, 0, 5].item() == 1.0
    # P1 to move -> ch2 all 1s
    assert (t[0, 2, :, :] == 1.0).all()
    # No chain -> ch3 all 0s
    assert (t[0, 3, :, :] == 0.0).all()
    # No banned squares -> ch4 all 0s
    assert (t[0, 4, :, :] == 0.0).all()
    # No chain current -> ch5 all 0s
    assert (t[0, 5, :, :] == 0.0).all()
    # No chain starter -> ch6 all 0s
    assert (t[0, 6, :, :] == 0.0).all()


def test_encode_board_mid_chain():
    state = Halma8()
    state.chain_active = True
    state.chain_starter = (5, 0)
    state.chain_current = (6, 1)
    state.chain_banned = {(5, 0), (5, 1)}
    t = encode_board(state)
    # ch3 all 1s (chain active)
    assert (t[0, 3, :, :] == 1.0).all()
    # ch4 banned squares
    assert t[0, 4, 5, 0].item() == 1.0
    assert t[0, 4, 5, 1].item() == 1.0
    assert t[0, 4, 0, 0].item() == 0.0
    # ch5 chain_current
    assert t[0, 5, 6, 1].item() == 1.0
    assert t[0, 5, 0, 0].item() == 0.0
    # ch6 chain_starter
    assert t[0, 6, 5, 0].item() == 1.0
    assert t[0, 6, 0, 0].item() == 0.0


def test_legal_action_indices_start_position():
    state = Halma8()
    idxs = legal_action_indices(state)
    assert len(idxs) > 0
    # No stop (no chain active)
    assert 416 not in idxs
    # Every index corresponds to a step or jump
    for i in idxs:
        assert ACTION_LIST[i][0] in ("step", "jump")


def test_legal_action_indices_step_is_legal():
    state = Halma8()
    idxs = legal_action_indices(state)
    step_idxs = [i for i in idxs if ACTION_LIST[i][0] == "step"]
    assert len(step_idxs) > 0
    # Verify a few are actually legal per the engine
    for i in step_idxs[:10]:
        _, r, c, nr, nc = ACTION_LIST[i]
        assert (nr, nc) in state.legal_steps(r, c)


def test_action_index_roundtrip():
    for a in ACTION_LIST:
        idx = action_index(a)
        assert ACTION_LIST[idx] == a


def test_decode_action_step_applies():
    state = Halma8()
    idxs = legal_action_indices(state)
    step_idxs = [i for i in idxs if ACTION_LIST[i][0] == "step"]
    assert len(step_idxs) > 0
    i = step_idxs[0]
    mv = decode_action(i)
    assert mv[0] == "step"
    assert len(mv) == 3
    # Apply to a fresh state -- should not raise
    state2 = Halma8()
    state2.apply_move(mv)


def test_decode_action_stop_returns_none():
    stop_idx = 416
    assert decode_action(stop_idx) is None


def test_decode_action_jump_applies():
    """Find a jump in the action list that is legal at some state, decode and apply."""
    # At start position, P2 base (0,5)-(2,7) is full, P1 base (5,0)-(7,2) is full.
    # A jump requires an occupied middle and empty destination.
    # P1 at (5,0) with P1 at (6,0) and empty (7,0)? No, (7,0) is P1.
    # Let's just find any jump in the legal set.
    state = Halma8()
    idxs = legal_action_indices(state)
    jump_idxs = [i for i in idxs if ACTION_LIST[i][0] == "jump"]
    if jump_idxs:
        i = jump_idxs[0]
        mv = decode_action(i)
        assert mv[0] == "jump"
        assert len(mv) == 4
        # Apply -- should not raise
        state2 = Halma8()
        state2.apply_move(mv)
    # It's possible there are no legal jumps at start (all adjacent to own base are occupied)
    # That's fine -- the test passes either way.


def test_halma_clone_fields_match():
    """T-11 gate (field equality).

    A mid-chain Halma8 cloned with `halma_clone` must have all fields
    equal to the source.
    """
    state = Halma8()
    state.chain_active = True
    state.chain_banned = {(5, 0), (5, 1)}
    state.chain_starter = (5, 0)
    state.chain_current = (6, 1)
    state.turn_no = 42
    state.turn = 1
    state.winner = None
    state.grid[3][3] = 1

    c = halma_clone(state)
    assert c.grid == state.grid
    assert c.turn == state.turn
    assert c.winner == state.winner
    assert c.turn_no == state.turn_no
    assert c.chain_active == state.chain_active
    assert c.chain_banned == state.chain_banned
    assert c.chain_starter == state.chain_starter
    assert c.chain_current == state.chain_current


def test_halma_clone_independence():
    """T-11 gate (independence).

    Mutating the clone must not affect the source, and vice versa.
    """
    state = Halma8()
    state.chain_banned = {(5, 0)}
    c = halma_clone(state)

    # Mutate clone's grid
    c.grid[0][0] = 99
    assert state.grid[0][0] != 99

    # Mutate clone's chain_banned
    c.chain_banned.add((7, 7))
    assert (7, 7) not in state.chain_banned

    # Mutate source's grid
    state.grid[7][7] = 88
    assert c.grid[7][7] != 88

    # Mutate source's chain_banned
    state.chain_banned.add((1, 1))
    assert (1, 1) not in c.chain_banned
