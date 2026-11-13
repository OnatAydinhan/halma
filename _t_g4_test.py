"""T-G4 automated gate — drives the real HalmaGUI + real ai.exe (Kibitzer).

Sub-tests (one GUI instance, sequential):
  1. Kibitzer on at the start position (H/H): panel shows a valid eval
     (expect 0/balanced on the symmetric board) and a non-empty line.
  2. Human move with Kibitzer on: the panel refreshes (new line lands) and
     the live board was NOT mutated by the analysis.
  3. Undo back to the start: the panel returns to the start eval.
  4. A/A auto-run with Kibitzer on: the game keeps moving (no freeze) and a
     landed analysis is visible in the panel.
  5. Toggle off: panel resets to "Kibitzer off".
  6. Error path: a wedged analysis child surfaces as a one-line warning,
     not a crash.

Prints "T-G4 GATE OK" on success. ASCII-only output (cp1252 console).
"""
import copy
import os
import re
import time
import traceback

import halma8
import halma8_gui

HERE = halma8_gui.HERE
MODEL_A = os.path.join(HERE, "c1_incumbent.json")
MODEL_KIB = os.path.join(HERE, "best_weights.json")
EVAL_RE = re.compile(r"^(0  \(balanced\)|[+-]\d+  \((Red|Blue) favours\)|#\d+  \(.*)$")


def pump(gui, until, timeout_s=90.0):
    """Pump Tk (fires after-callbacks: AI turns + Kibitzer polls) until until()."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        gui.root.update()
        if until():
            return True
        time.sleep(0.05)
    return False


def eval_text(gui):
    return gui.kib_eval.cget("text")


def line_text(gui):
    return gui.kib_line.cget("text")


def landed(gui):
    t = eval_text(gui)
    return t not in ("", "analysing…", "Kibitzer off") and t[0] != "⚠"


def first_legal_step(b, side):
    for r in range(b.SIZE):
        for c in range(b.SIZE):
            if b.grid[r][c] == side:
                steps = b.legal_steps(r, c)
                if steps:
                    return ("step", (r, c), steps[0])
    return None


def set_seat(gui, side, kind, model_basename=None, depth=3, movetime_ms=100):
    gui.seat_kind_var[side].set(kind)
    if kind == "ai":
        gui.seat_model_var[side].set(model_basename)
    gui.seat_depth_var[side].set(depth)
    gui.seat_mt_var[side].set(movetime_ms)
    gui._on_seat_kind(side)


def main():
    gui = halma8_gui.HalmaGUI(halma8.Halma8())
    try:
        # sanity: Kibitzer widgets exist and start off
        assert gui.kibitz_on is False
        assert gui.kib_btn.cget("text") == "Kibitzer: off"
        assert eval_text(gui) == "Kibitzer off"
        gui.kib_cfg["model"] = MODEL_KIB
        print("0) widgets present, Kibitzer starts off: OK")

        # --- 1) on at start position: valid eval + non-empty line ----------
        # NOTE: the engine has a documented first-mover (P1/Red) bias, so the
        # start-position root score is P1-favoured (positive), not 0 — see the
        # _t4_probe.py results recorded in plan_gui.md.
        gui._toggle_kibitz()
        assert gui.kibitz_on is True
        assert gui.kib_btn.cget("text") == "Kibitzer: on"
        grid_before = [row[:] for row in gui.board.grid]
        assert pump(gui, lambda: landed(gui), timeout_s=90.0), \
            f"start-position analysis never landed: {eval_text(gui)!r}"
        m = EVAL_RE.match(eval_text(gui))
        assert m, f"start eval not in a valid format: {eval_text(gui)!r}"
        assert "Red favours" in eval_text(gui), \
            f"start eval should be P1-favoured (first-mover bias): {eval_text(gui)!r}"
        assert "->" in line_text(gui), f"start line empty: {line_text(gui)!r}"
        assert [row[:] for row in gui.board.grid] == grid_before, \
            "analysis mutated the live board"
        start_eval = eval_text(gui)
        print(f"1) start position: {start_eval!r} + non-empty line: OK")

        # --- 2) human move: panel refreshes; analysis must not add a change --
        b = gui.board
        move = first_legal_step(b, halma8.P1)
        assert move is not None
        # Expected grid after EXACTLY this one human move (apply it to a
        # scratch board). The Kibitzer runs on its own deepcopy, so the live
        # grid after the move + a landed analysis must match this precisely —
        # any divergence means the analysis leaked a change into the live board.
        scratch = copy.deepcopy(b)
        scratch.apply_move(move)
        expected_grid = [row[:] for row in scratch.grid]
        gui._do_move(move)
        assert pump(gui, lambda: landed(gui) and line_text(gui) != "", timeout_s=90.0), \
            f"post-move analysis never landed: {eval_text(gui)!r}"
        assert EVAL_RE.match(eval_text(gui)), f"bad eval text: {eval_text(gui)!r}"
        assert "->" in line_text(gui) or "stop" in line_text(gui)
        assert [row[:] for row in gui.board.grid] == expected_grid, \
            "live board grid changed outside the single human move"
        assert b.turn == halma8.P2, "move did not hand the turn to Blue"
        print("2) panel refreshed after a human move, board not mutated: OK")

        # --- 3) undo back to start: eval returns to the start eval ---------
        gui.undo()
        assert b.turn == halma8.P1
        assert pump(gui, lambda: eval_text(gui) == start_eval, timeout_s=90.0), \
            f"undo did not restore start eval: {eval_text(gui)!r} (want {start_eval!r})"
        print(f"3) undo: panel back to start eval {start_eval!r}: OK")

        # --- 4) A/A auto-run with Kibitzer on: no freeze, panel lands ------
        set_seat(gui, halma8.P1, "ai", "c1_incumbent.json")
        set_seat(gui, halma8.P2, "ai", "best_weights.json")
        gui.kib_cfg["on"] = True
        gui.kib_on_var.set(True)
        gui._apply_players_config()
        b = gui.board
        t0 = time.monotonic()
        assert pump(gui, lambda: b.turn != halma8.P1 or b.winner is not None,
                    timeout_s=90.0), "A/A: Red (AI) did not move first"
        assert pump(gui, lambda: landed(gui), timeout_s=90.0), \
            f"A/A: no analysis landed: {eval_text(gui)!r}"
        moves = gui.log.get("1.0", "end").count("\n")
        assert moves >= 1, "A/A: no move logged"
        print(f"4) A/A with Kibitzer on: {moves} move(s), analysis landed "
              f"({eval_text(gui)!r}) in {time.monotonic() - t0:.1f}s: OK")

        # --- 5) toggle off: panel resets ------------------------------------
        gui._toggle_kibitz()
        assert gui.kibitz_on is False
        assert eval_text(gui) == "Kibitzer off"
        assert line_text(gui) == ""
        print("5) toggle off: panel reset: OK")

        # --- 6) error path: wedged child -> one-line warning, no crash ------
        # The A/A game above ended the position; the Kibitzer (by design) will
        # not analyse a finished board, so reset to a fresh live H/H board first
        # to guarantee an analysis actually runs, THEN wedge the analysis child.
        set_seat(gui, halma8.P1, "human")
        set_seat(gui, halma8.P2, "human")
        gui.new_game()
        assert gui.board.winner is None, "expected a live board to exercise the error path"
        if not gui.kibitz_on:
            gui._toggle_kibitz()  # on + live board -> a (good) analysis lands
        assert gui.kibitz_on is True
        assert pump(gui, lambda: landed(gui), timeout_s=90.0), \
            f"pre-wedge analysis never landed: {eval_text(gui)!r}"
        # Now wedge the child: kill it and point it at a missing binary so the
        # worker's respawn raises HalmaAIError -> a one-line warning, not a crash.
        assert gui._kib is not None, "expected a live analysis child"
        gui._kib.proc.kill()
        gui._kib.binary = os.path.join(HERE, "no_such_ai.exe")
        gui.kib_eval.config(text="analysing…")
        gui._kibitz_busy = False
        gui._kibitz_pending = False
        gui._kib_result = None
        gui._kibitz_epoch += 1
        gui._maybe_kibitz()
        assert pump(gui, lambda: eval_text(gui).startswith("⚠"), timeout_s=90.0), \
            f"error path did not surface: {eval_text(gui)!r}"
        # ASCII-only print: the warning text carries a U+26A0 glyph that the
        # cp1252 console can't encode (see project_env_windows_cp1252 memory).
        print("6) wedged child: one-line warning shown, no crash: OK")

        print("T-G4 GATE OK")
    except Exception:
        traceback.print_exc()
        print("T-G4 GATE FAILED")
        raise SystemExit(1)
    finally:
        gui._on_close()


if __name__ == "__main__":
    main()
