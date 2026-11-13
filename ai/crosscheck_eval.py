#!/usr/bin/env python
"""T-10 round-trip: the Python mirror of `evaluate.rs::evaluate_with` must
reproduce Rust's world score on the same position + weights.

This is the load-bearing Python<->Rust contract for the weight->score mapping.
It builds a set of hand-picked positions directly from the canonical `halma8`
engine, and for each one:

  * asks the Rust `evaluate_dump` example for `evaluate_with(&board, &weights)`
    using the SAME `weights.json`, and
  * computes the identical six-term linear score here in Python,

then asserts they agree within +/-1 cp. The six terms are integer-valued and
both engines use integer weights, so an exact match is expected; the +/-1 bound
is generous head-room.

Exit 0 on full agreement, 1 otherwise.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import halma8  # noqa: E402  (canonical oracle for board construction)

WEIGHTS_PATH = os.path.join(ROOT, "weights.json")

CARGO = "cargo"
RUN_ARGS = [CARGO, "run", "--quiet", "--example", "evaluate_dump",
            "--manifest-path", os.path.join(HERE, "Cargo.toml")]

# --- constants mirroring types.rs ------------------------------------------
SIZE = 8
P1, P2 = 1, 2
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def base_origin(side):
    return (5, 0) if side == P1 else (0, 5)


def base_cells(side):
    r0, c0 = base_origin(side)
    return [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]


def in_b(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def dist_to_base(r, c, r0, c0):
    dr = r - r0
    dbot = r - (r0 + 2)
    dc = c - c0
    dright = c - (c0 + 2)
    row_part = -dr if dr < 0 else (dbot if dbot > 0 else 0)
    col_part = -dc if dc < 0 else (dright if dright > 0 else 0)
    return row_part + col_part


# --- six terms, mirroring evaluate.rs --------------------------------------

def progress(g, side):
    r0, c0 = base_origin(P2 if side == P1 else P1)
    total = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if g[r][c] == side:
                total += dist_to_base(r, c, r0, c0)
    return -total


def base_presence(g, side):
    tgt = P2 if side == P1 else P1
    return sum(1 for (r, c) in base_cells(tgt) if g[r][c] == side)


def defense(g, side):
    opp = P2 if side == P1 else P1
    return -sum(1 for (r, c) in base_cells(side) if g[r][c] == opp)


def spread(g, side):
    pairs = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if g[r][c] != side:
                continue
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if in_b(nr, nc) and g[nr][nc] == side:
                    pairs += 1
    return -pairs


def mobility(g, side):
    total = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if g[r][c] != side:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                jr, jc = r + 2 * dr, c + 2 * dc
                if in_b(nr, nc) and g[nr][nc] == 0:
                    total += 1
                if in_b(nr, nc) and in_b(jr, jc) and g[nr][nc] != 0 and g[jr][jc] == 0:
                    total += 2
    return total


def threat(g, side):
    tgt = P2 if side == P1 else P1
    base = set(base_cells(tgt))
    marked = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if g[r][c] != side:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                jr, jc = r + 2 * dr, c + 2 * dc
                if in_b(nr, nc) and g[nr][nc] == 0 and (nr, nc) in base:
                    marked.add((nr, nc))
                if (in_b(nr, nc) and in_b(jr, jc) and g[nr][nc] != 0
                        and g[jr][jc] == 0 and (jr, jc) in base):
                    marked.add((jr, jc))
    return len(marked)


TERMS = {
    "progress": progress,
    "base_presence": base_presence,
    "defense": defense,
    "spread": spread,
    "mobility": mobility,
    "threat": threat,
}


def py_evaluate(grid, weights):
    def side_score(side):
        s = 0
        for t, fn in TERMS.items():
            s += weights[t] * fn(grid, side)
        return s
    return side_score(P1) - side_score(P2)


# --- board building + position string (mirror of crosscheck_board.py) ------

def blank(position):
    b = halma8.Halma8()
    for r in range(SIZE):
        for c in range(SIZE):
            b.grid[r][c] = 0
    for (r, c, p) in position:
        b.grid[r][c] = p
    return b


def board_str(b8):
    return "".join("." if v == 0 else "1" if v == 1 else "2"
                   for row in b8.grid for v in row)


def position_str(b8):
    if not b8.chain_active:
        chain = "-"
    else:
        o = b8.chain_starter
        banned = sorted(b8.chain_banned)
        cc = b8.chain_current or (0, 0)
        chain = "C:o={}{},b={};c={}".format(o[0], o[1], ",".join(f"{r}{c}" for (r, c) in banned), cc[0], cc[1])
    return f"{board_str(b8)} {b8.turn} {chain}"


def test_position_sets():
    return {
        "single_p1": [(4, 0, 1)],
        "crossed_bases": [(2, 7, 1), (6, 0, 2)],
        "adjacent_pair": [(4, 4, 1), (4, 5, 1), (0, 6, 2)],
        "antisym": [(1, 4, 1), (3, 3, 1), (0, 6, 2), (5, 5, 2)],
        "heavier": [(2, 4, 1), (4, 2, 1), (5, 4, 1), (6, 6, 1),
                    (3, 5, 2), (4, 7, 2), (6, 5, 2), (7, 3, 2)],
        "near_win": [(0, 5, 1), (0, 6, 1), (0, 7, 1), (1, 5, 1), (1, 6, 1),
                     (1, 7, 1), (2, 5, 1), (2, 6, 1), (3, 7, 1)],
        "start": None,  # the engine's default start placement
    }


def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"FAIL: {WEIGHTS_PATH} not found (run `py train_weights.py selfplay.csv -o weights.json` first)")
        return 1
    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)
    for t in TERMS:
        if t not in weights:
            print(f"FAIL: weights.json missing key '{t}'")
            return 1
    print("round-trip weights:", json.dumps(weights))

    sets = test_position_sets()
    boards = {"start": halma8.Halma8()}
    boards.update({name: blank(pieces) for name, pieces in sets.items() if pieces})

    cmds = [f"E {WEIGHTS_PATH} {position_str(boards[name])}" for name in boards]
    inp = "\n".join(cmds) + "\n"

    proc = subprocess.run(RUN_ARGS, input=inp, capture_output=True, text=True,
                          cwd=HERE, timeout=120)
    if proc.returncode != 0:
        print("FAIL: evaluate_dump exited nonzero")
        print("STDERR:", proc.stderr[-2000:])
        return 1

    got = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(got) != len(boards):
        print(f"FAIL: expected {len(boards)} output lines, got {len(got)}")
        print("---- stdout ----")
        print(proc.stdout)
        return 1

    ok = True
    for i, name in enumerate(boards):
        py_score = py_evaluate(boards[name].grid, weights)
        want = got[i].split(" ", 1)[1]
        try:
            rust_score = int(want)
        except ValueError:
            ok = False
            print(f"MISMATCH[{i}] {name}: unparseable Rust response: {got[i]!r} (py={py_score})")
            continue
        if got[i].startswith("OK") and abs(rust_score - py_score) > 1:
            ok = False
            print(f"MISMATCH[{i}] {name}: rust={rust_score} py={py_score} (delta {rust_score - py_score})")
        else:
            print(f"[{name}] rust={rust_score} py={py_score}  OK")

    if not ok:
        print("---- full stdout ----")
        print(proc.stdout)
        print("---- stderr ----")
        print(proc.stderr[-2000:])
        return 1

    print(f"PASS: {len(boards)} positions x weight set; Python mirror matches Rust evaluate_with within +/-1 cp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
