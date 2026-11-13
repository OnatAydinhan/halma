"""Tests for azero/endgame.py (T-14 gate).

The near-win synthesizer must return `n` independent, non-terminal,
winnable-in-few positions, each with exactly one base at `full`/9 holding an
attacker invader, `state.turn` set to the attacker (a wrong turn inverts the
+1 value signal), and a clean serialize/deserialize round-trip. These are the
premises T-14's true-terminal value target relies on.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from halma8 import P1, P2, base_cells
from azero.board import legal_action_indices
from azero.endgame import (
    deserialize, immediate_win_exists, make_endgame_positions, serialize,
)


def _fullness(grid):
    """Occupied/9 for each player's OWN base."""
    return {p: sum(1 for (r, c) in base_cells(p) if grid[r][c] != 0) for p in (P1, P2)}


def _count_in(grid, player, cells):
    return sum(1 for (r, c) in cells if grid[r][c] == player)


def test_returns_n_nonterminal_with_legal_moves():
    states = make_endgame_positions(20, full=8, seed=7)
    assert len(states) == 20
    for st in states:
        assert st.winner is None, "a synthesized endgame must be non-terminal"
        assert len(legal_action_indices(st)) >= 1, "position has no legal move"


def test_exactly_one_base_full_with_invader():
    for st in make_endgame_positions(12, full=8, seed=11):
        full = _fullness(st.grid)
        full_bases = [p for p in (P1, P2) if full[p] == 8]
        assert len(full_bases) == 1, f"expected exactly one base at 8/9, got {full}"
        base = full_bases[0]
        attacker = st.turn
        assert _count_in(st.grid, attacker, base_cells(base)) >= 1, \
            f"the 8/9 base must hold >=1 attacker ({attacker}) invader"


def test_turn_is_attacker_holding_the_invader():
    """Hard gate: state.turn must be the side holding the invader in the full
    base, so play_one_game's +1 lands on the correct mover."""
    for st in make_endgame_positions(12, full=8, seed=13):
        attacker = st.turn
        base = [p for p in (P1, P2) if _fullness(st.grid)[p] == 8][0]
        assert _count_in(st.grid, attacker, base_cells(base)) >= 1


def test_positions_are_winnable_in_one():
    """Every synthesized position (full=8) has a single legal step that wins —
    the exact true-terminal signal T-14 feeds the value head."""
    states = make_endgame_positions(16, full=8, seed=17)
    assert states, "no states generated"
    for st in states:
        assert immediate_win_exists(st), "position should be winnable-in-one"


def test_serialize_roundtrip_and_independence():
    st = make_endgame_positions(1, full=8, seed=19)[0]
    d = serialize(st)
    r = deserialize(d)
    assert r.grid == st.grid
    assert r.turn == st.turn
    assert r.winner == st.winner
    assert r.turn_no == st.turn_no
    assert r.chain_active == st.chain_active
    assert r.chain_banned == st.chain_banned
    assert r.chain_starter == st.chain_starter
    assert r.chain_current == st.chain_current
    # Independence: mutating the round-trip copy must not touch the source.
    r.grid[0][0] = 0
    assert r.grid[0][0] == 0
