"""check_halma engine — 8x8 halma variant (Part 1).

No tkinter imports here; this module must stay UI-independent.
"""

P1, P2 = 1, 2
NAME = {P1: "P1 (Red)", P2: "P2 (Blue)"}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

P1_BASE = tuple((r, c) for r in range(5, 8) for c in range(3))
P2_BASE = tuple((r, c) for r in range(3) for c in range(5, 8))


def base_cells(player):
    """The 9 base cells of a player. P1: rows 5–7, cols 0–2; P2: rows 0–2, cols 5–7."""
    if player == P1:
        return set(P1_BASE)
    if player == P2:
        return set(P2_BASE)
    raise ValueError(f"unknown player: {player}")


class IllegalMove(ValueError):
    pass


class Halma8:
    SIZE = 8

    def __init__(self):
        self.grid = [[0] * self.SIZE for _ in range(self.SIZE)]
        for r, c in P1_BASE:
            self.grid[r][c] = P1
        for r, c in P2_BASE:
            self.grid[r][c] = P2
        self.turn = P1
        self.winner = None
        self.turn_no = 1
        self.chain_active = False
        self.chain_banned = set()
        self.chain_starter = None
        self.chain_current = None  # position of the chain piece right now

    def legal_steps(self, r, c):
        """Adjacent orthogonal empty squares. [] for empty square or out-of-range."""
        grid = self.grid
        if not (0 <= r < self.SIZE and 0 <= c < self.SIZE) or grid[r][c] == 0:
            return []
        out = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE and grid[nr][nc] == 0:
                out.append((nr, nc))
        return out

    def legal_jumps(self, r, c):
        """(middle, destination) pairs. Middle must be occupied, destination empty. [] for empty square or out-of-range."""
        grid = self.grid
        if not (0 <= r < self.SIZE and 0 <= c < self.SIZE) or grid[r][c] == 0:
            return []
        out = []
        for dr, dc in DIRS:
            mr, mc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc
            if (0 <= nr < self.SIZE and 0 <= nc < self.SIZE
                    and grid[mr][mc] != 0 and grid[nr][nc] == 0):
                out.append(((mr, mc), (nr, nc)))
        return out

    def apply_move(self, move):
        """Apply ("step", pos, dst) or ("jump", pos, mid, dst).

        Returns True iff the move was a jump and that piece has further
        legal jumps (chain continues; turn unchanged). Returns False when
        the turn ends (turn already flipped, turn_no incremented).
        Raises IllegalMove on any rule violation or if a winner is set.
        """
        if self.winner is not None:
            raise IllegalMove(f"game already over, winner {self.winner}")
        kind = move[0]
        if not (isinstance(move, tuple) and 0 < len(move) <= 4):
            raise IllegalMove(f"bad move: {move!r}")
        for sq in move[1:]:
            if not (isinstance(sq, tuple) and len(sq) == 2
                    and all(isinstance(v, int) for v in sq)
                    and 0 <= sq[0] < self.SIZE and 0 <= sq[1] < self.SIZE):
                raise IllegalMove(f"bad square in move: {move!r}")
        if kind not in ("step", "jump"):
            raise IllegalMove(f"unknown move kind: {kind!r}")
        grid = self.grid
        if kind == "step":
            _, (r, c), (nr, nc) = move
            if grid[r][c] == 0:
                raise IllegalMove(f"no piece at {(r, c)}")
            if grid[nr][nc] != 0:
                raise IllegalMove(f"destination {(nr, nc)} occupied")
            if (nr, nc) not in self.legal_steps(r, c):
                raise IllegalMove(f"illegal step {(r, c)} -> {(nr, nc)}")
            grid[nr][nc] = grid[r][c]
            grid[r][c] = 0
            self.chain_active = False  # defensive: a step is always a new turn
            self.chain_banned = set()
            self.chain_current = None
            self._check_win()
            self.end_turn()
            return False
        if kind == "jump":
            _, (r, c), (mr, mc), (nr, nc) = move
            if grid[r][c] == 0:
                raise IllegalMove(f"no piece at {(r, c)}")
            if self.chain_active:
                if (r, c) != self.chain_current:
                    raise IllegalMove(
                        f"chain in progress: only the chain piece at "
                        f"{self.chain_current} may jump, got {(r, c)}")
                if (nr, nc) in self.chain_banned:
                    raise IllegalMove(f"destination {(nr, nc)} is banned within the active chain")
            if ((mr, mc), (nr, nc)) not in self.legal_jumps(r, c):
                raise IllegalMove(f"illegal jump {(r, c)} -> {(nr, nc)}")
            grid[nr][nc] = grid[r][c]
            grid[r][c] = 0
            if not self.chain_active:
                self.chain_active = True
                self.chain_starter = (r, c)
                self.chain_banned = {(r, c)}  # origin: revisit ban
            self.chain_banned.add((nr, nc))
            self.chain_current = (nr, nc)
            self._check_win()
            if self.winner is None and self._has_open_jump(nr, nc):
                return True  # chain in progress: turn unchanged
            self.chain_active = False
            self.chain_banned = set()
            self.chain_current = None
            self.end_turn()
            return False

    def _has_open_jump(self, r, c):
        """True iff any legal jump from (r,c) lands on a non-banned square."""
        for _mid, dst in self.legal_jumps(r, c):
            if dst not in self.chain_banned:
                return True
        return False

    def end_turn(self):
        """Flip the turn and increment turn_no. No-op if a winner is set.
        Resets any active chain state."""
        if self.winner is not None:
            return
        self.turn = P2 if self.turn == P1 else P1
        self.turn_no += 1
        self.chain_active = False
        self.chain_banned = set()
        self.chain_current = None

    def _check_win(self):
        """Occupied-space rule: the attacker wins when the defender's base is
        completely filled AND at least one attacker piece is among the occupants
        (the defender's own starting pieces do not count toward the opponent's win)."""
        def _full_and_invaded(base, attacker):
            if not all(self.grid[r][c] != 0 for r, c in base):
                return False
            return any(self.grid[r][c] == attacker for r, c in base)

        if _full_and_invaded(P2_BASE, P1):   # P1 wins: P2's base full + has a red piece
            self.winner = P1
        elif _full_and_invaded(P1_BASE, P2): # P2 wins: P1's base full + has a blue piece
            self.winner = P2
