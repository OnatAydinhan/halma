#!/usr/bin/env python
"""T-3 cross-check: the *stateful* rule layer must agree between Rust and Python.

The Rust engine's rule layer (`apply_move` / `end_turn` / `check_win`, the
chain-revisit ban, and the "full + at-least-one-of-mine" win check) is a port of
`halma8.Halma8`. We eliminate divergence by replaying the exact scenario list the
T-3/T-4 unit tests in `halma8_test.py` exercise, on BOTH engines, and asserting
byte-for-byte agreement of:
  * the result of each move (`T`/`F`/`E` = Ok(true)/Ok(false)/Err, i.e. the two
    subtle rules and every raise case),
  * the full resulting state (`grid`, `turn`, `turn_no`, `winner`, and the
    active-chain origin/banned set) after every step.

`halma8.py` is the canonical oracle and is never modified.

Exit 0 iff every scenario is identical on both engines (or 1 on any mismatch or
a cargo failure).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import halma8  # noqa: E402  (canonical oracle)

RUN_ARGS = [
    "cargo", "run", "--quiet", "--example", "rules_dump",
    "--manifest-path", os.path.join(HERE, "Cargo.toml"),
]


def bs(pieces):
    """{(r,c)->p} on an empty board -> 64-char board string, row-major '.'/1/2."""
    g = [[0] * 8 for _ in range(8)]
    for (r, c), p in pieces.items():
        g[r][c] = p
    return "".join(
        "1" if v == 1 else "2" if v == 2 else "."
        for row in g for v in row
    )


def parse_board(s):
    """64-char board string -> [[int]*8]*8."""
    g = [[0] * 8 for _ in range(8)]
    i = 0
    for r in range(8):
        for c in range(8):
            ch = s[i]
            i += 1
            g[r][c] = 0 if ch == "." else (1 if ch == "1" else 2)
    return g


def pos_str(g):
    """Replicate the Rust `Board::to_position_str` wire format exactly."""
    board = "".join(
        "." if v == 0 else "1" if v == 1 else "2"
        for row in g.grid for v in row
    )
    turn = str(g.turn)
    if (not g.chain_active) or (g.chain_starter is None):
        chain = "-"
    else:
        orr, occ = g.chain_starter
        body = ",".join(f"{r}{c}" for r, c in sorted(g.chain_banned))
        cc = g.chain_current or (0, 0)
        chain = f"C:o={orr}{occ};b={body};c={cc[0]}{cc[1]}"
    return f"{board} {turn} {chain}"


def state_line(g):
    """Replicate the Rust `Board::state_line` exactly."""
    w = 0 if g.winner is None else g.winner
    return f"{pos_str(g)} turn_no={g.turn_no} winner={w}"


def apply(g, mv):
    try:
        return "T" if g.apply_move(mv) else "F"
    except halma8.IllegalMove:
        return "E"


def run_py(ops):
    """Replay an op list on the canonical Python engine; return expected lines."""
    g = halma8.Halma8()
    out = []
    for op in ops:
        t = op.split()
        cmd = t[0]
        if cmd == "NEW":
            g = halma8.Halma8()
        elif cmd == "MAKES":
            g = halma8.Halma8()
            g.grid = parse_board(t[1])
        elif cmd == "TURN":
            g.turn = int(t[1])
        elif cmd == "WINNER":
            g.winner = None if int(t[1]) == 0 else int(t[1])
        elif cmd == "PLACE":
            g.grid[int(t[1])][int(t[2])] = int(t[3])
        elif cmd == "CHECKWIN":
            g._check_win()
        elif cmd == "ENDTURN":
            g.end_turn()
        elif cmd == "STATE":
            pass
        elif cmd == "STEP":
            r = apply(g, ("step", (int(t[1]), int(t[2])), (int(t[3]), int(t[4]))))
            out.append(f"{r} {state_line(g)}")
            continue
        elif cmd == "JUMP":
            r = apply(
                g, ("jump", (int(t[1]), int(t[2])),
                    (int(t[3]), int(t[4])), (int(t[5]), int(t[6]))))
            out.append(f"{r} {state_line(g)}")
            continue
        else:
            raise ValueError(f"unknown op {cmd!r}")
        out.append(state_line(g))
    return out


def scenarios():
    """Each scenario mirrors a `halma8_test.py` T-3/T-4 test (see that file)."""
    return [
        ("step_applies_and_flips_turn",
         ["NEW", "STEP 5 2 4 2"]),

        ("two_jump_chain",
         ["MAKES " + bs({(4, 1): 1, (4, 2): 2, (4, 4): 2}),
          "JUMP 4 1 4 2 4 3", "JUMP 4 3 4 4 4 5"]),

        ("banned_destination_raises_while_open",
         ["MAKES " + bs({(4, 1): 1, (4, 2): 2, (4, 4): 2, (4, 6): 2}),
          "TURN 1", "JUMP 4 1 4 2 4 3", "JUMP 4 3 4 2 4 1"]),

        ("three_jump_chain",
         ["MAKES " + bs({(4, 1): 1, (4, 2): 2, (4, 4): 2, (4, 6): 2}),
          "TURN 1", "JUMP 4 1 4 2 4 3", "JUMP 4 3 4 4 4 5", "JUMP 4 5 4 6 4 7"]),

        ("single_jump_ending_turn",
         ["MAKES " + bs({(4, 4): 1, (4, 5): 2}),
          "TURN 1", "JUMP 4 4 4 5 4 6"]),

        ("end_turn_only_flips",
         ["NEW", "ENDTURN"]),

        ("step_into_occupied_raises",
         ["NEW", "STEP 6 0 6 1"]),

        ("jump_empty_middle_raises",
         ["MAKES " + bs({(4, 4): 1}), "TURN 1", "JUMP 4 4 4 5 4 6"]),

        ("jump_occupied_dst_raises",
         ["MAKES " + bs({(4, 4): 1, (4, 5): 2, (4, 6): 2}),
          "TURN 1", "JUMP 4 4 4 5 4 6"]),

        ("move_off_board_a",
         ["MAKES " + bs({(7, 0): 1}), "TURN 1", "STEP 7 0 8 0"]),

        ("move_off_board_b",
         ["NEW", "STEP 5 2 4 3"]),

        ("move_after_winner_raises_a",
         ["NEW", "WINNER 2", "STEP 5 2 4 2"]),

        ("move_after_winner_raises_b",
         ["MAKES " + bs({(4, 4): 1}), "TURN 1", "WINNER 1", "JUMP 4 4 4 5 4 6"]),

        ("end_turn_noop_winner",
         ["NEW", "WINNER 1", "ENDTURN"]),

        ("illegal_move_step",
         ["NEW", "STEP 5 2 4 3"]),

        ("p1_wins_pure",
         ["MAKES " + bs({
             (0, 5): 1, (0, 6): 1, (0, 7): 1,
             (1, 5): 1, (1, 6): 1, (1, 7): 1,
             (2, 6): 1, (2, 7): 1,
             (2, 4): 1,
         }), "STEP 2 4 2 5"]),

        ("p1_wins_mixed_parked_blue",
         ["MAKES " + bs({
             (0, 5): 1, (0, 6): 1, (0, 7): 1,
             (1, 5): 1, (1, 6): 1, (1, 7): 1,
             (2, 5): 2, (2, 6): 1,
             (3, 7): 1,
         }), "STEP 3 7 2 7"]),

        ("no_false_positive_at_8_of_9",
         ["MAKES " + bs({
             (0, 5): 1, (0, 6): 1, (0, 7): 1,
             (1, 5): 2, (1, 6): 1, (1, 7): 1,
             (2, 5): 1, (2, 6): 1,
         }), "CHECKWIN"]),

        ("p2_wins_pure",
         ["MAKES " + bs({
             (5, 0): 2, (5, 1): 2, (5, 2): 2,
             (6, 0): 2, (6, 1): 2, (6, 2): 2,
             (7, 0): 2, (7, 1): 2,
             (7, 3): 2,
         }), "STEP 7 3 7 2"]),

        ("p2_wins_mixed",
         ["MAKES " + bs({
             (5, 0): 2, (5, 1): 2, (5, 2): 2,
             (6, 0): 2, (6, 1): 2, (6, 2): 2,
             (7, 0): 1, (7, 1): 2,
             (7, 3): 2,
         }), "STEP 7 3 7 2"]),

        ("move_after_winner_then_rejected_turn_noop",
         ["MAKES " + bs({
             (0, 5): 1, (0, 6): 1, (0, 7): 1,
             (1, 5): 1, (1, 6): 1, (1, 7): 1,
             (2, 6): 1, (2, 7): 1,
             (2, 4): 1,
         }), "STEP 2 4 2 5", "PLACE 3 3 1", "STEP 3 3 3 4", "ENDTURN"]),

        ("winning_jump_open_ends_chain",
         ["MAKES " + bs({
             (0, 5): 1, (0, 6): 1, (0, 7): 1,
             (1, 5): 1, (1, 6): 1, (1, 7): 1,
             (2, 5): 1, (2, 7): 1,
             (3, 6): 1,
             (4, 6): 1,
         }), "TURN 1", "JUMP 4 6 3 6 2 6", "JUMP 2 6 2 5 2 4"]),
    ]


def main():
    scns = scenarios()
    triples = []
    all_ops = []
    for label, ops in scns:
        py = run_py(ops)
        for op, exp in zip(ops, py):
            triples.append((label, op, exp))
        all_ops.extend(ops)

    inp = "\n".join(all_ops) + ("\n" if all_ops else "")
    proc = subprocess.run(
        RUN_ARGS, input=inp, capture_output=True, text=True, cwd=HERE, timeout=240
    )
    if proc.returncode != 0:
        print("FAIL: cargo run exited nonzero")
        print("STDERR:", proc.stderr[-2500:])
        return 1

    got = [ln for ln in proc.stdout.splitlines() if ln != ""]
    if len(got) != len(triples):
        print(f"FAIL: expected {len(triples)} reply lines, got {len(got)}")
        print("---- rust stdout ----")
        print(proc.stdout[-3000:])
        return 1

    fail = 0
    for i, (label, op, exp) in enumerate(triples):
        if got[i] != exp:
            fail += 1
            print(f"MISMATCH [{label}] op: {op}")
            print(f"  want: {exp}")
            print(f"  got : {got[i]}")

    if fail:
        print(f"FAIL: {fail} divergent step(s) across {len(scns)} scenarios ({len(all_ops)} ops)")
        return 1

    print(
        f"PASS: {len(scns)} T-3/T-4 scenarios x {len(all_ops)} ops "
        f"— Rust rule layer matches halma8.Halma8 (result class + full state, zero divergence)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
