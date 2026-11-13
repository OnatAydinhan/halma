#!/usr/bin/env python
"""One-shot depth-vs-time bench: for 3 random mid-game positions, run a
50-second time-budgeted search and record the per-depth wall-clock time
from each `info` line as it arrives. Prints a clean table.

Run:  py bench_depths.py
Output: bench_depths_out.txt (also printed to stdout)
"""
import sys as _sys
# Windows console defaults to cp1252 and chokes on non-ASCII (e.g. the U+2248
# in the summary note); force UTF-8 stdout/stderr so the script can't crash on
# a locale mismatch.
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import subprocess
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halma8 import Halma8, P1, P2  # noqa: E402

AI_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ai", "target", "release", "ai.exe")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bench_depths_out.txt")


def pos_str(g):
    board = "".join(
        "." if v == 0 else "1" if v == 1 else "2"
        for row in g.grid for v in row
    )
    turn = str(g.turn)
    if not g.chain_active or g.chain_starter is None:
        chain = "-"
    else:
        orr, occ = g.chain_starter
        body = ",".join(f"{r}{c}" for r, c in sorted(g.chain_banned))
        cc = g.chain_current or (0, 0)
        chain = f"C:o={orr}{occ};b={body};c={cc[0]}{cc[1]}"
    return f"{board} {turn} {chain}"


def random_position(seed, num_moves=50):
    """Play `num_moves` random legal moves from the starting position."""
    b = Halma8()
    rng = random.Random(seed)
    for _ in range(num_moves):
        moves = []
        for r in range(8):
            for c in range(8):
                if b.grid[r][c] != b.turn:
                    continue
                for nr, nc in b.legal_steps(r, c):
                    moves.append(("step", (r, c), (nr, nc)))
                for mid, dst in b.legal_jumps(r, c):
                    if b.chain_active and dst in b.chain_banned:
                        continue
                    moves.append(("jump", (r, c), mid, dst))
        if not moves:
            if b.chain_active:
                b.end_turn()
            continue
        b.apply_move(rng.choice(moves))
        # Stop the chain sometimes so positions vary
        if b.chain_active and rng.random() < 0.4:
            b.end_turn()
        if b.winner:
            break
    return b


def time_budgeted_search(position_str, movetime_ms=50000):
    """
    Run one time-budgeted search. Returns
    (list_of_(depth, score_cp, elapsed_ms), bestmove_line, total_elapsed_ms)
    """
    proc = subprocess.Popen(
        [AI_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def recv_until(token):
        while True:
            line = proc.stdout.readline().rstrip("\n")
            if line.startswith(token):
                return line

    # Handshake
    send("uci")
    recv_until("uciok")
    send("isready")
    recv_until("readyok")

    # Set position
    send(f"position {position_str}")

    # Start the timed search
    t_start = time.time()
    send(f"go movetime {movetime_ms}")

    results = []   # (depth, score_cp, elapsed_ms_since_go_sent)
    bestmove = None

    while True:
        line = proc.stdout.readline().rstrip("\n")
        elapsed_ms = int((time.time() - t_start) * 1000)
        if line.startswith("bestmove"):
            bestmove = line
            break
        if line.startswith("info"):
            parts = line.split()
            depth = score = None
            for i, p in enumerate(parts):
                if p == "depth" and i + 1 < len(parts):
                    try:
                        depth = int(parts[i + 1])
                    except ValueError:
                        pass
                if p == "cp":
                    try:
                        score = int(parts[i + 1])
                    except (ValueError, IndexError):
                        pass
            if depth is not None:
                results.append((depth, score, elapsed_ms))

    total_ms = int((time.time() - t_start) * 1000)

    # Clean up
    try:
        send("quit")
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
        proc.wait()

    return results, bestmove, total_ms


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 68)
    emit("  check_halma AI — depth vs wall-clock time table")
    emit(f"  Machine: this box  |  Build: release  |  No pruning")
    emit(f"  Budget per position: 50 s (movetime 50000)")
    emit("=" * 68)

    positions = []
    for seed in (42, 137, 256):
        b = random_position(seed, num_moves=50)
        if b.winner:
            emit(f"\n[skip] seed={seed}: game already finished (P{b.winner} won at turn {b.turn_no})")
            continue
        label = f"P{b.turn}-to-move, turn {b.turn_no}"
        positions.append((seed, pos_str(b), label))
        emit(f"\nPosition seed={seed}: {label}")

    emit()

    # Store (seed, max_depth, total_ms) for the summary
    summary = []

    for seed, pos, label in positions:
        emit(f"\n--- seed={seed}  ({label}) ---")
        try:
            results, bestmove, total_ms = time_budgeted_search(pos, movetime_ms=50000)
        except Exception as e:
            emit(f"  ERROR: {e}")
            continue

        if not results:
            emit("  (no info lines received — check Rust build)")
            continue

        emit(f"  {'Depth':<8} {'ms (cumulative)':<18} {'score (cp)':<12}")
        emit(f"  {'-'*4} {'-'*16} {'-'*10}")
        for depth, score, elapsed in results:
            score_str = str(score) if score is not None else "(mate)"
            emit(f"  {depth:<8} {elapsed:<18} {score_str:<12}")

        emit(f"  Total: {total_ms} ms  ->  {total_ms/1000:.1f} s")
        if bestmove:
            emit(f"  Best move: {bestmove}")

        max_depth = max(d for d, _, _ in results)
        summary.append((seed, max_depth, total_ms))

    emit("\n" + "=" * 68)
    emit(f"  Summary: deepest completed depth within 50-s budget")
    emit("=" * 68)
    for seed, max_depth, total_ms in summary:
        emit(f"  seed={seed}:  depth {max_depth}  in  {total_ms/1000:.1f} s")

    emit()
    emit("Note: cumulative ms counts from the moment `go movetime 50000` was sent.")
    emit("      Delta between consecutive rows ≈ wall-clock time spent at that depth.")
    emit("      Tree size grows roughly 4-10x per depth iteration (branch factor x depth).")
    emit()

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
