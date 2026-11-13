#!/usr/bin/env python
"""T-1 cross-check: Rust board state + serialization must match `halma8.Halma8`.

Runs the Rust `board_dump` example and, for a set of positions (start,
after-step, mid-chain, after-win placement) built *in Python from the
canonical engine*, asserts:
  * Rust's own `start()` board string == Python's start board string, and
  * Rust round-trips (parse + re-serialize) each Python position string
    byte-for-byte.

Exit 0 on full agreement, 1 otherwise.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import halma8  # noqa: E402  (canonical oracle)

CARGO = "cargo"
RUN_ARGS = [CARGO, "run", "--quiet", "--example", "board_dump", "--manifest-path",
            os.path.join(HERE, "Cargo.toml")]


def board_str(b8):
    return "".join(
        "." if v == 0 else "1" if v == 1 else "2"
        for row in b8.grid for v in row
    )


def position_str(b8):
    if not b8.chain_active:
        chain = "-"
    else:
        o = b8.chain_starter
        banned = sorted(b8.chain_banned)
        b = ",".join(f"{r}{c}" for (r, c) in banned)
        cc = b8.chain_current or (0, 0)
        chain = f"C:o={o[0]}{o[1]};b={b};c={cc[0]}{cc[1]}"
    return f"{board_str(b8)} {b8.turn} {chain}"


def contract_position():
    """A few contrived states straight from the Python engine."""
    states = {}

    # (a) start
    s = halma8.Halma8()
    states["start"] = s

    # (b) after one legal step (P1 (6,2) -> (6,3), an empty neighbor); turn
    #     becomes P2.
    b = halma8.Halma8()
    b.apply_move(("step", (6, 2), (6, 3)))
    states["after_step"] = b

    # (c) mid-chain: set directly to exercise the chain wire format.
    mc = halma8.Halma8()
    mc.chain_active = True
    mc.chain_starter = (6, 1)
    mc.chain_banned = {(6, 1), (6, 3), (7, 2)}
    states["mid_chain"] = mc

    # (d) after-win placement: P2's base (rows 0-2, cols 5-7) full, with one P1.
    w = halma8.Halma8()
    for (r, c) in halma8.P2_BASE:
        w.grid[r][c] = halma8.P2
    w.grid[0][5] = halma8.P1  # one red piece among the blue in P2's base
    states["win_placement"] = w

    return states


def main():
    states = contract_position()

    # Build the exact command sequence we send (order matters: one output line
    # per command, returned in the same order).
    cmds = []           # (label, command, expected_output_line)
    cmds.append(("start", "S", "OK " + board_str(states["start"])))
    for name, b8 in states.items():
        cmds.append((f"pos:{name}", f"P {position_str(b8)}", "OK " + position_str(b8)))
    for name, b8 in states.items():
        cmds.append((f"board:{name}", f"B {board_str(b8)}", "OK " + board_str(b8)))

    inp = "\n".join(c for _, c, _ in cmds) + "\n"

    proc = subprocess.run(
        RUN_ARGS,
        input=inp,
        capture_output=True,
        text=True,
        cwd=HERE,
        timeout=120,
    )
    if proc.returncode != 0:
        print("FAIL: board_dump exited nonzero")
        print("STDERR:", proc.stderr[-2000:])
        return 1

    got = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(got) != len(cmds):
        print(f"FAIL: expected {len(cmds)} output lines, got {len(got)}")
        print("---- stdout ----")
        print(proc.stdout)
        return 1

    # Positional compare: command N returns output line N, in order.
    ok = True
    for i, (label, _cmd, want) in enumerate(cmds):
        if got[i] != want:
            ok = False
            print(f"MISMATCH[{i}] {label}\n  want: {want}\n  got : {got[i]}")

    if not ok:
        print("---- full stdout ----")
        print(proc.stdout)
        print("---- stderr ----")
        print(proc.stderr[-2000:])
        return 1

    print(f"PASS: {len(states)} positions + start; Rust board/position serialization matches halma8.Halma8")
    print("[start]\n  " + board_str(states["start"]))
    for name, b8 in states.items():
        print(f"[{name}] pos = " + position_str(b8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
