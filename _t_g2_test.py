"""T-G2/T-G3 automated gate — drives the real HalmaGUI + real ai.exe.

Sub-tests (one GUI instance, sequential):
  1. H/H  — both seats human: seat model widgets disabled (renders correctly).
  2. H/A  — human Red plays a legal step; AI Blue must answer; undo back to
     Blue's turn must let AI Blue play on (undo + AI continuation).
  3. A/H  — AI Red must move first on New game.
  4. A/A  — different models per seat (weights re-apply on New game): the game
     auto-runs, Red (P1) moves first and Blue lands a logged move.

Prints "T-G2 GATE OK" on success. ASCII-only output (cp1252 console).
"""
import os
import time
import traceback

import halma8
import halma8_gui
import halma_ai

HERE = halma8_gui.HERE
MODEL_A = os.path.join(HERE, "c1_incumbent.json")
MODEL_B = os.path.join(HERE, "best_weights.json")
RED, BLUE = halma8.NAME[halma8.P1], halma8.NAME[halma8.P2]


def pump(gui, until, timeout_s=60.0):
    """Pump Tk (fires after-callbacks, i.e. AI turns) until until() is true."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        gui.root.update()
        if until():
            return True
        time.sleep(0.05)
    return False


def log_text(gui):
    return gui.log.get("1.0", "end")


def set_seat(gui, side, kind, model_basename=None, depth=3, movetime_ms=100):
    gui.seat_kind_var[side].set(kind)
    if kind == "ai":
        gui.seat_model_var[side].set(model_basename)
    gui.seat_depth_var[side].set(depth)
    gui.seat_mt_var[side].set(movetime_ms)
    gui._on_seat_kind(side)


def first_legal_step(b, side):
    for r in range(b.SIZE):
        for c in range(b.SIZE):
            if b.grid[r][c] == side:
                steps = b.legal_steps(r, c)
                if steps:
                    return ("step", (r, c), steps[0])
    return None


def main():
    gui = halma8_gui.HalmaGUI(halma8.Halma8())
    try:
        # --- 1) H/H: widgets render disabled for human seats ----------------
        set_seat(gui, halma8.P1, "human")
        set_seat(gui, halma8.P2, "human")
        for side in (halma8.P1, halma8.P2):
            assert str(gui.seat_widgets[side]["cb"]["state"]) == "disabled"
            assert str(gui.seat_widgets[side]["browse"]["state"]) == "disabled"
        gui._apply_players_config()  # plain human game, must not error
        assert gui.board.winner is None and gui.board.turn == halma8.P1
        print("1) H/H render + new game: OK")

        # --- 2) H/A: human Red step, AI Blue answers; undo; AI continues ----
        set_seat(gui, halma8.P1, "human")
        set_seat(gui, halma8.P2, "ai", "c1_incumbent.json")
        gui._apply_players_config()
        b = gui.board
        assert b.turn == halma8.P1 and b.winner is None
        move = first_legal_step(b, halma8.P1)
        assert move is not None, "no legal step for Red at start"
        gui._do_move(move)
        assert b.turn == halma8.P2
        assert pump(gui, lambda: b.turn == halma8.P1 or b.winner is not None), \
            "AI Blue did not answer"
        assert BLUE in log_text(gui), f"no Blue move logged: {log_text(gui)!r}"
        print("2a) H/A: AI Blue answered human Red: OK")

        # undo the AI's answer -> back to Blue's turn (top snapshot is the
        # one taken before the AI move); the AI seat must pick up and play on
        gui.undo()
        assert b.turn == halma8.P2, "undo did not restore Blue's turn"
        assert pump(gui, lambda: b.turn == halma8.P1 or b.winner is not None), \
            "AI Blue did not continue after undo"
        print("2b) undo + AI continues: OK")

        # --- 3) A/H: AI Red moves first on New game ---------------------------
        set_seat(gui, halma8.P1, "ai", "c1_incumbent.json")
        set_seat(gui, halma8.P2, "human")
        gui._apply_players_config()
        b = gui.board
        assert pump(gui, lambda: b.turn == halma8.P2 or b.winner is not None), \
            "AI Red did not move first"
        assert RED in log_text(gui), f"no Red move logged: {log_text(gui)!r}"
        print("3) A/H: AI Red moved first: OK")

        # --- 4) A/A: distinct models per seat, auto-run both sides ----------
        set_seat(gui, halma8.P1, "ai", "c1_incumbent.json")
        set_seat(gui, halma8.P2, "ai", "best_weights.json")
        gui._apply_players_config()
        b = gui.board
        assert b.turn == halma8.P1
        assert pump(gui, lambda: BLUE in log_text(gui) or b.winner is not None,
                    timeout_s=120.0), "A/A auto-run: Blue never moved"
        txt = log_text(gui)
        first_line = txt.splitlines()[0]
        assert RED in first_line, f"first move was not Red: {first_line!r}"
        assert RED in txt and BLUE in txt, f"missing a side in log: {txt!r}"
        print("4) A/A auto-run (Red first, both sides logged): OK")

        print("T-G2 GATE OK")
    except Exception:
        traceback.print_exc()
        print("T-G2 GATE FAILED")
        raise SystemExit(1)
    finally:
        gui._on_close()


if __name__ == "__main__":
    main()
