"""Board encoding and fixed action list for the AlphaZero NN (Part 3)."""

import torch

from halma8 import Halma8

P1 = 1
P2 = 2
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIZE = 8


def _build_action_list():
    actions = []
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                    actions.append(("step", r, c, nr, nc))
                nr, nc = r + 2 * dr, c + 2 * dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                    actions.append(("jump", r, c, nr, nc))
    actions.append(("stop",))
    return actions


ACTION_LIST = _build_action_list()
_INDEX_MAP = {a: i for i, a in enumerate(ACTION_LIST)}


def action_index(action):
    """O(1) lookup: action tuple -> index into ACTION_LIST."""
    return _INDEX_MAP[action]


def decode_action(index):
    """Convert an ACTION_LIST entry to a Halma8.apply_move tuple, or None for stop."""
    a = ACTION_LIST[index]
    kind = a[0]
    if kind == "stop":
        return None
    if kind == "step":
        _, r, c, nr, nc = a
        return ("step", (r, c), (nr, nc))
    # jump
    _, r, c, nr, nc = a
    mid = ((r + nr) // 2, (c + nc) // 2)
    return ("jump", (r, c), mid, (nr, nc))


def encode_board(state):
    """Convert a Halma8 state to a (1, 7, 8, 8) float32 tensor."""
    t = torch.zeros(1, 7, 8, 8, dtype=torch.float32)
    grid = state.grid
    for r in range(SIZE):
        for c in range(SIZE):
            v = grid[r][c]
            if v == P1:
                t[0, 0, r, c] = 1.0
            elif v == P2:
                t[0, 1, r, c] = 1.0
    if state.turn == P1:
        t[0, 2, :, :] = 1.0
    if state.chain_active:
        t[0, 3, :, :] = 1.0
    for (r, c) in state.chain_banned:
        t[0, 4, r, c] = 1.0
    if state.chain_current is not None:
        r, c = state.chain_current
        t[0, 5, r, c] = 1.0
    if state.chain_starter is not None:
        r, c = state.chain_starter
        t[0, 6, r, c] = 1.0
    return t


def legal_action_indices(state):
    """Return a sorted list of indices into ACTION_LIST that are legal at `state`."""
    indices = []
    turn = state.turn
    grid = state.grid
    for i, a in enumerate(ACTION_LIST):
        kind = a[0]
        if kind == "stop":
            if state.chain_active:
                indices.append(i)
        elif kind == "step":
            _, r, c, nr, nc = a
            if grid[r][c] != turn:
                continue
            if state.chain_active:
                continue
            if (nr, nc) in state.legal_steps(r, c):
                indices.append(i)
        elif kind == "jump":
            _, r, c, nr, nc = a
            if grid[r][c] != turn:
                continue
            mid = ((r + nr) // 2, (c + nc) // 2)
            if ((mid[0], mid[1]), (nr, nc)) not in state.legal_jumps(r, c):
                continue
            if state.chain_active:
                if (r, c) != state.chain_current:
                    continue
                if (nr, nc) in state.chain_banned:
                    continue
            indices.append(i)
    return indices


def halma_clone(state):
    """Fast field-by-field clone of a Halma8 state.

    Skips the generic `copy.deepcopy` protocol (memo dict, per-type dispatch)
    for what is a small fixed-field object. The clone is fully independent:
    mutating its `grid` or `chain_banned` does not affect the source.
    """
    c = Halma8.__new__(Halma8)
    c.grid = [row[:] for row in state.grid]       # fresh row lists; ints shared (immutable)
    c.turn = state.turn
    c.winner = state.winner
    c.turn_no = state.turn_no
    c.chain_active = state.chain_active
    c.chain_banned = set(state.chain_banned)      # new set; tuples shared (immutable)
    c.chain_starter = state.chain_starter          # None or tuple — share is safe
    c.chain_current = state.chain_current          # None or tuple — share is safe
    return c
