"""check_halma — Part 2, T-8: Python subprocess driver for the Rust AI.

Spawns the release binary ``ai/target/release/ai`` and speaks the line
protocol defined in ``plan_ai.md`` (the kego ``tools_smoke.py::UciEngine``
pattern, ported). The GUI (``halma8_gui.py``) uses :class:`HalmaAI` to ask
for a single ``bestmove`` on the AI's turn and feeds that move straight
through the engine's existing ``apply_move`` / ``end_turn`` — the Python
``halma8.py`` engine stays the sole authority on legality.

Wire protocol (one source of truth — see plan_ai.md):
  uci                       -> id ... / uciok
  position <pos>            -> (sets the board; <pos> from board_to_position)
  go depth <N> | go movetime <ms>
  bestmove <step|jump|stop ...>
  quit                      -> exit

The Python action format that :meth:`HalmaAI.move` returns matches the
human move tuples the GUI already consumes:
    ("step", (r,c), (nr,nc))
    ("jump", (r,c), (mr,mc), (nr,nc))
  or the string "stop" (the AI chose to end a live chain -> end_turn()).
"""

import copy
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AI_BIN = os.path.join(HERE, "ai", "target", "release", "ai.exe")

_DEFAULT_TIMEOUT = 10.0  # seconds; hard ceiling on a single move request


class HalmaAIError(RuntimeError):
    """Raised when the AI child dies, closes stdout, or exceeds the timeout."""


def _pair(tok):
    r, c = tok.split(",")
    return (int(r), int(c))


def parse_bestmove(s):
    """`step r,c nr,nc` / `jump r,c mr,mc nr,nc` / `stop` -> a Python action.

    Returns a ``"step"``/``"jump"`` tuple ready for ``Halma8.apply_move``, or
    the string ``"stop"`` meaning the AI chose to end its live chain
    (the caller maps that to ``Halma8.end_turn()``).
    """
    t = s.split()
    if not t:
        raise HalmaAIError("empty bestmove")
    kind = t[0]
    if kind == "step":
        if len(t) != 3:
            raise HalmaAIError(f"bad step token: {s!r}")
        return ("step", _pair(t[1]), _pair(t[2]))
    if kind == "jump":
        if len(t) != 4:
            raise HalmaAIError(f"bad jump token: {s!r}")
        return ("jump", _pair(t[1]), _pair(t[2]), _pair(t[3]))
    if kind == "stop":
        if len(t) != 1:
            raise HalmaAIError(f"bad stop token: {s!r}")
        return "stop"
    raise HalmaAIError(f"unknown move kind in bestmove: {s!r}")


def board_to_position(g):
    """Canonical ``halma8.Halma8`` -> wire ``position`` string (plan_ai.md)."""
    board = "".join(
        "." if v == 0 else "1" if v == 1 else "2" for row in g.grid for v in row
    )
    turn = str(g.turn)
    if (not g.chain_active) or (g.chain_starter is None):
        chain = "-"
    else:
        orr, occ = g.chain_starter
        body = ",".join(f"{r}{c}" for r, c in sorted(g.chain_banned))
        curr = f"c={g.chain_current[0]}{g.chain_current[1]}" if g.chain_current else "c=00"
        chain = f"C:o={orr}{occ};b={body};{curr}"
    return f"{board} {turn} {chain}"


def position_to_board(pos):
    """Reconstruct a ``halma8.Halma8`` from a wire ``position`` string.

    Inverse of :func:`board_to_position`; exists so the Kibitzer's :meth:`line`
    can replay a line on a copy when handed a position string rather than a
    board object. Raises :class:`HalmaAIError` on a malformed string.
    """
    import halma8

    parts = pos.split()
    if len(parts) != 3:
        raise HalmaAIError(f"bad position string: {pos!r}")
    board_s, turn_s, chain_s = parts

    size = halma8.Halma8.SIZE
    g = halma8.Halma8()
    grid = [[0] * size for _ in range(size)]
    if len(board_s) != size * size:
        raise HalmaAIError(f"board field has {len(board_s)} cells, expected {size*size}")
    val = {"1": halma8.P1, "2": halma8.P2, ".": 0}
    for idx in range(size * size):
        ch = board_s[idx]
        if ch not in val:
            raise HalmaAIError(f"bad cell {ch!r} in board field")
        grid[idx // size][idx % size] = val[ch]
    g.grid = grid
    g.turn = int(turn_s)

    if chain_s != "-":
        # `C:o=rrcc;b=r1c1,r2c2,...;c=rrcc` — origin, banned squares, chain piece.
        seg = chain_s.split(";", 1)[1]
        o = seg.split("o=", 1)[1].split(";", 1)[0]
        g.chain_active = True
        g.chain_starter = (int(o[0]), int(o[1]))
        banned = set()
        b = seg.split("b=", 1)[1].split(";", 1)[0] if "b=" in seg else ""
        for tok in b.split(","):
            if tok:
                banned.add((int(tok[0]), int(tok[1])))
        g.chain_banned = banned
        g.chain_current = None
        if "c=" in seg:
            c_tok = seg.split("c=", 1)[1].split(";", 1)[0]
            if c_tok:
                g.chain_current = (int(c_tok[0]), int(c_tok[1]))
    return g


class HalmaAI:
    """Child-process driver for ``ai/target/release/ai`` (kego UciEngine port)."""

    def __init__(self, binary=AI_BIN, depth=4, movetime_ms=500, timeout=_DEFAULT_TIMEOUT):
        self.binary = binary
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.timeout = timeout
        self.proc = None
        # Last requested per-side weight paths ((p1_path, p2_path) or None).
        # Kept so a timeout-kill respawn can re-apply them transparently.
        self._weights = None

    # -- process management ------------------------------------------------
    def _ensure(self):
        """Spawn and handshake if the child is not already alive (auto-resume
        after a timeout-kill so the next turn recovers transparently)."""
        if self.proc is not None and self.proc.poll() is None:
            return
        if not os.path.exists(self.binary):
            raise HalmaAIError(
                f"AI binary not found at {self.binary!r}; "
                "run `cargo build --release` in ai/ first")
        self.proc = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._expect_token("uciok", deadline=time.monotonic() + 10.0)
        # A respawned child starts with the default weights (the server state
        # was lost with the process); re-apply any previously requested paths so
        # a timeout-kill mid-game never silently drops the loaded model.
        self._send_weights()

    def set_weights(self, p1_path=None, p2_path=None):
        """Point the engine at a trained model per side (T-G0 wire protocol).

        ``p1_path`` is the Red/P1 seat model, ``p2_path`` the Blue/P2 seat model;
        pass either or both. Sent as ``setoption name WeightsP1|WeightsP2
        path=…``. A path with spaces is not supported (model paths here have
        none). Call before the first ``go``; if the child is already up the
        option is applied immediately, otherwise it is applied right after the
        handshake — and re-applied automatically after any later respawn.
        """
        self._weights = (p1_path, p2_path)
        was_up = self.proc is not None and self.proc.poll() is None
        self._ensure()
        if was_up:
            self._send_weights()

    def _send_weights(self):
        if not self._weights:
            return
        p1, p2 = self._weights
        if p1 is not None:
            self._send(f"setoption name WeightsP1 path={p1}")
        if p2 is not None:
            self._send(f"setoption name WeightsP2 path={p2}")

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _kill(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    # -- low-level protocol ------------------------------------------------
    def _send(self, line):
        if self.proc is None or self.proc.poll() is not None:
            raise HalmaAIError("AI process is not running")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_line(self):
        line = self.proc.stdout.readline()
        if line == "":
            raise HalmaAIError("AI process closed stdout unexpectedly")
        return line.rstrip("\r\n")

    def _expect_token(self, token, deadline):
        """Read lines until the first word is ``token``; return the full line.
        Enforces a wall-clock ``deadline`` so a wedged engine can't hang the GUI."""
        while True:
            if time.monotonic() >= deadline:
                self._kill()
                raise HalmaAIError(f"AI timed out waiting for {token!r}")
            line = self._read_line()
            if line.split()[:1] == [token]:
                return line

    # -- public API --------------------------------------------------------
    def _go(self, position, depth=None, movetime_ms=None, use_depth=False):
        """Send ``position`` + a bounded ``go``; return ``(raw_action, score)``.

        ``raw_action`` is the token string after ``bestmove`` (``"step …"``,
        ``"jump …"``, ``"stop"``, or the terminal sentinel ``"(none)"``);
        ``score`` is the ``info … score cp N`` value (or ``None`` if the engine
        emitted no such line). Shared by the move/line APIs so the score is
        captured exactly once.
        """
        if use_depth:
            depth = self.depth if depth is None else depth
            go_line = f"go depth {depth}"
        else:
            movetime_ms = self.movetime_ms if movetime_ms is None else movetime_ms
            go_line = f"go movetime {movetime_ms}"
        self._ensure()
        deadline = time.monotonic() + self.timeout
        self._send(f"position {position}")
        self._send(go_line)
        return self._expect_bestmove_score(deadline)

    def _expect_bestmove_score(self, deadline):
        """Read lines until ``bestmove``; return ``(raw_action, score)``.

        Captures the most recent ``info … score cp N`` seen before ``bestmove``.
        Enforces the wall-clock ``deadline`` (kills a wedged engine on expiry).
        """
        score = None
        while True:
            if time.monotonic() >= deadline:
                self._kill()
                raise HalmaAIError("AI timed out waiting for bestmove")
            line = self._read_line()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "info":
                if "cp" in parts:
                    i = parts.index("cp")
                    if i + 1 < len(parts):
                        try:
                            score = int(parts[i + 1])
                        except ValueError:
                            pass
                continue
            if parts[0] == "bestmove":
                if len(parts) < 2:
                    raise HalmaAIError(f"bad bestmove line: {line!r}")
                return " ".join(parts[1:]), score
            # any other pre-bestmove line (e.g. extra info) is ignored

    def move(self, position, depth=None, movetime_ms=None):
        """Ask the AI for one action from ``position`` (action only).

        Returns a Python action tuple (``("step",...)`` / ``("jump",...)``) or
        the string ``"stop"``. Use :meth:`move_with_score` to also get the eval.
        Raises :class:`HalmaAIError` if the child dies, closes stdout, or
        exceeds ``self.timeout`` (child is killed on timeout).
        """
        raw, _score = self._go(position, depth, movetime_ms)
        return parse_bestmove(raw)

    def move_with_score(self, position, depth=None, movetime_ms=None):
        """As :meth:`move`, but return ``(action, score)``.

        ``action`` has the :meth:`move` contract; ``score`` is the root
        ``info … score cp N`` (positive = good for Red/P1, negative = Blue/P2).
        """
        raw, score = self._go(position, depth, movetime_ms)
        return (parse_bestmove(raw), score)

    def move_depth(self, position, depth=None):
        """Depth-bounded move (same contract as :meth:`move`, but ``go depth``)."""
        raw, _score = self._go(position, depth, use_depth=True)
        return parse_bestmove(raw)

    def line(self, board, plies=8):
        """Build the Kibitzer's best continuation line from the current position.

        ``board`` is a ``halma8.Halma8`` instance (preferred) or a wire position
        string. The board is deep-copied, so the caller's live game is never
        mutated. Returns ``(actions, root_score)``:

        - ``actions`` — a list of Python action tuples (``("step",...)`` /
          ``("jump",...)``) and/or ``"stop"`` strings, up to ``plies`` long, or
          shorter when the game ends or the engine reports a terminal ``(none)``.
          A halma chain shows up naturally as consecutive actions by the same side.
        - ``root_score`` — the first (root) ``info … score cp`` value (or ``None``).

        Re-queries the engine once per ply (so it is ``plies``× a single move);
        keep ``plies`` modest and run this on a non-blocking path (T-G4).
        """
        if isinstance(board, str):
            board = position_to_board(board)
        g = copy.deepcopy(board)
        actions = []
        root_score = None
        for _ in range(max(1, int(plies))):
            if g.winner is not None:
                break
            raw, score = self._go(board_to_position(g))
            if root_score is None:
                root_score = score
            if raw == "(none)":
                break
            action = parse_bestmove(raw)
            actions.append(action)
            try:
                if action == "stop":
                    g.end_turn()
                else:
                    g.apply_move(action)
            except Exception:
                break
        return actions, root_score


if __name__ == "__main__":
    import halma8

    if not os.path.exists(AI_BIN):
        print(f"ERROR: {AI_BIN} not found; run `cargo build --release` in ai/ first")
        raise SystemExit(1)

    # Point the engine at a known trained model if one is present — proves the
    # T-G0 setoption wiring reaches the driver (a bad/absent path is not fatal).
    weight_file = None
    for cand in ("c1_incumbent.json", "best_weights.json"):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            weight_file = p
            break

    ai = HalmaAI()
    try:
        if weight_file:
            ai.set_weights(weight_file, weight_file)
            print(f"WEIGHTS: {weight_file}")

        g = halma8.Halma8()
        pos = board_to_position(g)
        print(f"POSITION: {pos}")

        # 1) move_with_score: action must be legal, score an int.
        action, score = ai.move_with_score(pos, movetime_ms=500)
        print(f"ACTION: {action!r}")
        print(f"SCORE: {score}")
        assert isinstance(score, int), f"score must be an int, got {score!r}"
        scratch = halma8.Halma8()
        if action == "stop":
            scratch.end_turn()
        else:
            scratch.apply_move(action)
        print(f"applied OK (legal) — turn now {scratch.turn}, chain_active={scratch.chain_active}")

        # 2) line(): non-empty action list + an int root score.
        actions, root = ai.line(pos, plies=4)
        print(f"LINE ({len(actions)}): {actions!r}  root={root}")
        assert isinstance(actions, list) and len(actions) >= 1, f"line too short: {actions!r}"
        assert isinstance(root, int), f"root score must be an int, got {root!r}"
        print("SMOKE OK")
    finally:
        ai.close()
