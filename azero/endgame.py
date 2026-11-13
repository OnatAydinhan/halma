"""Endgame position synthesizer for the AlphaZero NN (Part 3, T-14).

The all-draw start-from-fresh run never feeds the value head a true terminal
+/-1 (800/800 random games -> 0 winners even at 400 moves). This module
synthesizes positions that are *already 1-3 moves from a win* -- the defender's
base is `full`/9 (default 8) with >=`attacker_min` attacker pieces, and the
attacker has a piece ready to fill the last empty cell -- so that
`play_one_game`'s value target for those games is the real outcome (the
attacker wins in one move), not the dense-MCTS ~0.

`state.turn` is always set to the **attacker**, so the +1 lands on the correct
mover (a wrong turn would invert the value signal). Rules/legality stay with
`halma8.py` (the oracle); this module only *places pieces*, then re-checks
invariants through the oracle before accepting a position.

`serialize`/`deserialize` give a plain-Python round-trip of the 8 `Halma8`
fields (the same ones `halma_clone` copies) so a position crosses the `spawn`
worker boundary (T-9) without torch.
"""

import random

from halma8 import Halma8, P1, P2, base_cells
from azero.board import ACTION_LIST, action_index, decode_action, legal_action_indices
from azero.board import halma_clone

_SIZE = 8
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _outside_neighbor(cell, base):
    """An orthogonally-adjacent square to `cell` that is OUTSIDE `base`, or None.

    A base cell that has such a neighbor is one the attacker can fill with a
    single step from just outside the base (the clean one-move finish)."""
    r, c = cell
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < _SIZE and 0 <= nc < _SIZE and (nr, nc) not in base:
            return (nr, nc)
    return None


def _template(full, attacker_min, attacker, defender, rng):
    """Build one near-win: defender's base at `full`/9 with >=`attacker_min`
    attacker pieces and an attacker piece ready to step into the last empty
    cell (a one-move win when full==8). Returns a `Halma8`, or None if the
    invariants cannot be met for this draw."""
    base = set(base_cells(defender))
    cells = sorted(base)
    # The empty cell must have an outside orthogonal neighbor (filler's spot).
    with_outside = [cell for cell in cells if _outside_neighbor(cell, base)]
    if not with_outside:
        return None
    E = rng.choice(with_outside)                    # the last empty base cell
    others = [cell for cell in cells if cell != E]
    if len(others) < full:
        return None
    occupied = rng.sample(others, full)

    grid = [[0] * _SIZE for _ in range(_SIZE)]
    roles = [attacker] * attacker_min + [defender] * (full - attacker_min)
    rng.shuffle(roles)
    for cell, who in zip(occupied, roles):
        grid[cell[0]][cell[1]] = who

    # Filler attacker piece on the outside neighbor of E (steps in -> win).
    F = _outside_neighbor(E, base)
    if F is None or grid[F[0]][F[1]] != 0:
        return None
    grid[F[0]][F[1]] = attacker

    # Scatter each side's remaining pieces in non-base cells (off BOTH bases)
    # for realism without creating an accidental double-win or unreachable
    # clutter. Each side keeps its full 9 pieces.
    both_bases = set(base_cells(P1)) | set(base_cells(P2))
    free = [(r, c) for r in range(_SIZE) for c in range(_SIZE)
            if (r, c) not in both_bases and grid[r][c] == 0]
    rng.shuffle(free)
    atk_extra = max(0, 9 - (attacker_min + 1))      # +1 = the filler
    def_extra = max(0, 9 - (full - attacker_min))
    pos = 0
    for who, cnt in ((attacker, atk_extra), (defender, def_extra)):
        for _ in range(cnt):
            if pos >= len(free):
                return None
            cell = free[pos]
            pos += 1
            grid[cell[0]][cell[1]] = who

    st = Halma8.__new__(Halma8)
    st.grid = grid
    st.turn = attacker
    st.winner = None
    st.turn_no = 1
    st.chain_active = False
    st.chain_banned = set()
    st.chain_starter = None
    st.chain_current = None

    # Re-check invariants through the oracle (defensive against bad draws).
    st._check_win()
    if st.winner is not None:
        return None
    if not legal_action_indices(st):
        return None
    return st


def make_endgame_positions(n, full=8, attacker_min=1, seed=None):
    """Return `n` independent near-win `Halma8` states.

    Each state is non-terminal, to-move = the attacker, and winnable-in-few
    (a base at `full`/9 with >=`attacker_min` attacker pieces and a ready
    fill-in piece). `full` is capped at 8 (9/9 + invader is already terminal).
    `attacker` alternates P1/P2 so both value signs are covered.
    """
    full = max(1, min(8, int(full)))
    attacker_min = max(1, int(attacker_min))
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n:
        # Alternate the attacker so the +1 signal is exercised for both sides.
        attacker = P1 if (i % 2 == 0) else P2
        defender = P2 if attacker == P1 else P1
        st = _template(full, attacker_min, attacker, defender, rng)
        if st is not None:
            out.append(st)
            i += 1
    return out


def immediate_win_exists(state):
    """True iff the player to move has a single legal action that wins.

    Used to prove a synthesized position is actually winnable-in-one (and is a
    sound oracle for a greedy finish). Delegates legality + application to the
    `Halma8` methods; never mutates `state`."""
    for idx in legal_action_indices(state):
        a = ACTION_LIST[idx]
        trial = halma_clone(state)
        if a[0] == "stop":
            trial.end_turn()
        else:
            trial.apply_move(decode_action(action_index(a)))
        if trial.winner is not None and trial.winner == state.turn:
            return True
    return False


def serialize(state):
    """Plain-Python dict of the 8 `Halma8` fields (picklable, no torch)."""
    return {
        "grid": [row[:] for row in state.grid],
        "turn": state.turn,
        "winner": state.winner,
        "turn_no": state.turn_no,
        "chain_active": state.chain_active,
        "chain_banned": [tuple(c) for c in state.chain_banned],
        "chain_starter": tuple(state.chain_starter) if state.chain_starter else None,
        "chain_current": tuple(state.chain_current) if state.chain_current else None,
    }


def deserialize(d):
    """Rebuild a `Halma8` from a `serialize` dict (independent of the source)."""
    st = Halma8.__new__(Halma8)
    st.grid = [row[:] for row in d["grid"]]
    st.turn = d["turn"]
    st.winner = d["winner"]
    st.turn_no = d["turn_no"]
    st.chain_active = d["chain_active"]
    st.chain_banned = set(tuple(c) for c in d["chain_banned"])
    st.chain_starter = tuple(d["chain_starter"]) if d["chain_starter"] else None
    st.chain_current = tuple(d["chain_current"]) if d["chain_current"] else None
    return st
