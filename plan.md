# check_halma — Part 1 Implementation Plan (COMPLETE)

An 8×8 halma variant ("checkers board, no diagonals"). Part 1 = two-local-player GUI
(Python + Tkinter). **Part 1 is feature-complete and shipped.** Part 2 (AI) lives in
`plan_ai.md` — do not develop it from this file.

## HANDOFF (read this first)

- **Status: Part 1 (T1–T6 + clocks) COMPLETE.** Engine + GUI + tests + exe all working.
- **Files:** `halma8.py` (engine, tkinter-free — the canonical rule authority),
  `halma8_gui.py` (Tkinter UI), `halma8_test.py` (36 tests), `dist/check_halma.exe`.
- **First command in any new session:** `py halma8_test.py` → 36 tests, OK, exit 0.
  Run before **and** after any change.
- **Do NOT start Part 2 (AI) from this file** — that is `plan_ai.md`.
- **Hard rules (every session):**
  1. One task at a time; never "improve" outside the current task.
  2. Engine and UI stay **separate**: no engine code imports tkinter; the GUI may import
     the engine, never the reverse.
  3. Each task ends with its test gate green (run it, paste-verify, then stop).
  4. Use **`py`**, not `python` (broken MS Store alias). Shell timeouts ≤ 60 000 ms.
  5. No external dependencies (stdlib + tkinter only); single project dir.
- **Rollback:** plain files, no git assumptions. If a task corrupts things, restore the
  passing state from the end of the previous task (each task is small and self-contained).

## Game spec (single source of truth — do not deviate)

- **Board:** 8×8, alternating squares. Coordinates `(row,col)`, 0-based, `(0,0)` =
  top-left. P1 = **red**, 9 pieces in bottom-left 3×3 (rows 5–7, cols 0–2). P2 = **blue**,
  9 pieces in top-right 3×3 (rows 0–2, cols 5–7). **P1 moves first.**
- **Movement:** orthogonal only (no diagonals).
  - **Step:** to an adjacent empty square.
  - **Jump:** over ONE adjacent occupied square onto the empty square beyond. **No
    capture** — the jumped piece stays.
  - **Jump chains:** after a jump, if that same piece has another jump, it may continue.
    The turn ends when the player stops (explicitly, or no jump remains); a plain step
    always ends the turn.
  - **Chain-revisit ban (2026-10-31):** within one piece's chain, it may not land on the
    turn's origin or any already-landed square. Without this, every jump can always be
    played back over the same piece to the vacated origin, so a turn could never end
    naturally. Consequence: an isolated single jump is a **final** jump — `apply_move`
    returns `False` and flips the turn.
- **Win — "full + invaded" rule (2026-10-31):**
  - P1 wins when **all 9 squares of the top-right base (rows 0–2, cols 5–7) are occupied
    AND at least one is a P1 piece**. P2 wins symmetrically on the bottom-left base.
  - Occupancy is checked **automatically** after a move. A player who parks a piece in
    their own base can no longer block the opponent's win (removes the classic
    parked-piece stalemate). The opponent's own starting pieces filling their base do
    *not* trigger a win for anyone.

## Completed tasks

- **T1** engine skeleton + test harness. **T2** `legal_steps`/`legal_jumps` (pure move
  generation). **T3** `apply_move`/`end_turn` + turn/chain handling. **T3.5** chain-revisit
  ban (`chain_active`/`chain_banned`/`chain_starter` state; banned destination raises
  `IllegalMove`; final jump ends the chain and returns `False`). **T4** `_check_win()`
  ("full + invaded", both bases, P1 checked first). **T5** Tkinter GUI: board rendering +
  click-to-move + highlights + status line. **T6** full playable loop: jump-chain UX
  (Done / re-click / click-elsewhere to stop), win banner (board fully frozen, no hint
  dots), New game, move log, undo.
- **Per-player chess clocks (added 2026-11-01, GUI only):** Settings dialog (base minutes
  + increment seconds per player, default **10 min / 0 sec**). No clock ticks until the
  first move; then normal alternation with the mover's increment banked on turn end.
  `_tick()` via 100 ms Tk `after`; hitting 0 sets the opponent as winner and freezes the
  board. Undo restores the clock. Exe rebuilt after the change.
- **Gate (all):** `py halma8_test.py` → **36 tests, OK, exit 0** (engine untouched).
  Verified headlessly across starting position, select/step, jump chain,
  full+invaded win banner, New game, undo, clock handoff, increment banking, and
  timeout→opponent-wins.

## Part 2 (AI) — see `plan_ai.md`

Not in this file's scope. The engine is already UI-independent (`Halma8`,
`legal_steps`, `legal_jumps`, `apply_move`, `end_turn`). The AI is a Rust subprocess
speaking the line protocol, and Python `halma8.py` remains the sole legality authority.
