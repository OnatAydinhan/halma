"""Replay games/human_wins_2026-11-05.txt through the engine; report board + stats."""
import re
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from halma8 import Halma8, P1, P2, NAME

LOG = "games/human_wins_2026-11-05.txt"
JUMP = re.compile(r"^(\d+)\s+P(\d)\s+\(\w+\):\s+\((\d+),\s*(\d+)\)\s+jumps over\s+\((\d+),\s*(\d+)\)\s*->\s*\((\d+),\s*(\d+)\)\s+jump")
STEP = re.compile(r"^(\d+)\s+P(\d)\s+\(\w+\):\s+\((\d+),\s*(\d+)\)\s*->\s*\((\d+),\s*(\d+)\)\s*step")

g = Halma8()
moves = []
counts = {P1: 0, P2: 0}
jumps = {P1: 0, P2: 0}
chain_lens = {P1: 0, P2: 0}
maxchain = {P1: 0, P2: 0}

def parse(line):
    m = JUMP.match(line)
    if m:
        g = [int(m.group(i)) for i in range(1, 9)]
        # g = [turn, player, r, c, mr, mc, nr, nc]
        return g[1], ("jump", (g[2], g[3]), (g[4], g[5]), (g[6], g[7]))
    m = STEP.match(line)
    g = [int(m.group(i)) for i in range(1, 7)]
    # g = [turn, player, r, c, nr, nc]
    return g[1], ("step", (g[2], g[3]), (g[4], g[5]))

cur = 0
for line in open(LOG, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip() or line[0].isdigit() is False:
        continue  # header / format lines
    player, move = parse(line)
    if g.chain_active and g.turn != player:
        g.end_turn()  # human pressed Done to stop an open jump chain
    assert g.turn == player, f"turn mismatch at {line!r}: engine {g.turn}, log {player}"
    cont = g.apply_move(move)
    counts[player] += 1
    if move[0] == "jump":
        jumps[player] += 1
    moves.append((line, cont))

# longest single jump chain (consecutive 'jumps over' lines by any player)
mxc = 0
for idx, (line, cont) in enumerate(moves):
    is_jump = "jumps over" in line
    if is_jump:
        cur += 1
        mxc = max(mxc, cur)
    else:
        cur = 0

def board_str(gr):
    out = []
    for r in range(8):
        row = []
        for c in range(8):
            v = gr[r][c]
            row.append("  . " if v == 0 else ("  R " if v == P1 else "  B "))
        out.append(" | ".join(row) + f"  row {r}")
    return "\n".join(out)

print("FINAL WINNER:", NAME.get(g.winner, "none/ongoing"), "turn_no", g.turn_no)
print()
print(board_str(g.grid))
print()
print("pliy counts  R=%d B=%d" % (counts[P1], counts[P2]))
print("jump plies   R=%d B=%d" % (jumps[P1], jumps[P2]))
print("single longest chain (any player) = %d jumps" % mxc)

# base occupancy
def base_info(base, label):
    occ = sum(1 for r, c in base if g.grid[r][c] != 0)
    rp = sum(1 for r, c in base if g.grid[r][c] == P1)
    bp = sum(1 for r, c in base if g.grid[r][c] == P2)
    print(f"{label} base: {occ}/9 occupied, red={rp}, blue={bp}")

base_info(tuple((r, c) for r in range(5, 8) for c in range(3)), "P1 (Red)  ")
base_info(tuple((r, c) for r in range(3) for c in range(5, 8)), "P2 (Blue) ")
