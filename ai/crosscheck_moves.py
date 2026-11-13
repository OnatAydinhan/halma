#!/usr/bin/env python
"""T-2 cross-check: Rust `legal_steps` / `legal_jumps` must match `halma8.Halma8`.

For each fixture position (the same board layouts as `T2Tests` in
`halma8_test.py`), this drives the Rust `moves_dump` example and asserts that,
for EVERY square (0,0)..(7,7), the Rust and Python engines produce identical
`legal_steps` and `legal_jumps` sets — compared byte-for-byte after the shared
canonical (ascending) serialization used by `examples/moves_dump.rs`.

Exit 0 on full agreement, 1 on any mismatch or a cargo failure.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import halma8  # noqa: E402  (canonical oracle)

RUN_ARGS = [
    "cargo", "run", "--quiet", "--example", "moves_dump",
    "--manifest-path", os.path.join(HERE, "Cargo.toml"),
]

ALL_SQUARES = [(r, c) for r in range(8) for c in range(8)]


def board_str(grid):
    """[[int]*8]*8 -> 64-char board string (row-major, '.'/1/2)."""
    return "".join(
        "." if v == 0 else "1" if v == 1 else "2"
        for row in grid for v in row
    )


def fmt_steps(steps):
    """[(r,c), ...] -> canonical '|' joined ascending string ('' if empty)."""
    return "|".join(f"{r}{c}" for r, c in sorted(steps))


def fmt_jumps(jumps):
    """[((mid),(dst)), ...] -> canonical '|' joined ascending string ('' if empty)."""
    def key(t):
        (mr, mc), (nr, nc) = t
        return (mr, mc, nr, nc)
    return "|".join(f"{mr}{mc}{nr}{nc}" for (mr, mc), (nr, nc) in sorted(jumps, key=key))


def make_board(pieces):
    """Empty 8x8 board with only the given (r,c)->player pieces placed."""
    g = halma8.Halma8()
    g.grid = [[0] * 8 for _ in range(8)]
    for (r, c), p in pieces.items():
        g.grid[r][c] = p
    return g


def fixtures():
    """(label, grid) pairs, mirroring the board layouts of halma8_test.T2Tests."""
    P1, P2 = halma8.P1, halma8.P2
    return [
        ("start", halma8.Halma8().grid),
        ("isolated_jump_right", make_board({(4, 4): P1, (4, 5): P2}).grid),
        ("jump_over_own_piece", make_board({(4, 4): P1, (4, 5): P1}).grid),
        ("jump_over_own_other_side", make_board({(4, 6): P1, (4, 5): P1}).grid),
        ("no_jump_middle_empty", make_board({(4, 4): P1, (4, 6): P2}).grid),
        ("no_jump_dst_occupied", make_board({(4, 4): P1, (4, 5): P2, (4, 6): P2}).grid),
        ("jump_edge_top", make_board({(0, 4): P1, (0, 5): P2}).grid),
        ("jump_edge_down_from_top", make_board({(0, 4): P1, (1, 4): P2}).grid),
        ("jump_edge_bottom", make_board({(7, 4): P1, (7, 5): P2}).grid),
        ("jump_offboard_right", make_board({(7, 6): P1, (7, 7): P2}).grid),
        ("single_piece", make_board({(4, 4): P1}).grid),
    ]


def main():
    fix = fixtures()

    # One batched cargo run: for each fixture, `B <board>` then all 64 squares.
    cmds = []
    for _label, grid in fix:
        cmds.append("B " + board_str(grid))
        for r, c in ALL_SQUARES:
            cmds.append(f"L {r} {c}")
    inp = "\n".join(cmds) + "\n"

    proc = subprocess.run(
        RUN_ARGS, input=inp, capture_output=True, text=True, cwd=HERE, timeout=180
    )
    if proc.returncode != 0:
        print("FAIL: cargo run exited nonzero")
        print("STDERR:", proc.stderr[-2000:])
        return 1

    got = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    want_total = len(fix) * (1 + len(ALL_SQUARES))
    if len(got) != want_total:
        print(f"FAIL: expected {want_total} output lines, got {len(got)}")
        print("---- stdout ----")
        print(proc.stdout)
        return 1

    fail = 0
    pos = 0
    for label, grid in fix:
        # Rebuild the Python oracle for this fixture.
        g = halma8.Halma8()
        g.grid = [row[:] for row in grid]

        b_line = got[pos]
        pos += 1
        if b_line != "OK":
            fail += 1
            print(f"MISMATCH {label} board-set line: {b_line!r} (want 'OK')")

        for (r, c) in ALL_SQUARES:
            want = f"OK steps={fmt_steps(g.legal_steps(r, c))} jumps={fmt_jumps(g.legal_jumps(r, c))}"
            got_line = got[pos]
            pos += 1
            if got_line != want:
                fail += 1
                print(f"MISMATCH {label} ({r},{c}):\n  want: {want}\n  got : {got_line}")

    if fail:
        print(f"FAIL: {fail} step/jump mismatches across {len(fix)} fixtures x {len(ALL_SQUARES)} squares")
        return 1

    print(f"PASS: {len(fix)} fixtures x {len(ALL_SQUARES)} squares "
          f"— Rust legal_steps/legal_jumps match halma8.Halma8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
