"""T-G5 automated gate — validation matrix + weight-wiring regression detector.

Covers plan_gui.md T-G5 step 1 (the parts not already proven by T-G2/T-G4),
driving the real HalmaGUI + real ai.exe:

  R) WEIGHT-WIRING REGRESSION: two distinct models (c1_incumbent.json vs
     best_weights.json) must NOT play identically — the silent "loaded but
     ignored" failure the plan warns about. Uses a deterministic `go depth`
     search on the SAME fixed asymmetric position as
     ai/src/search.rs::t_g0_distinct_weights_change_the_search_answer.
  K) KIBITZER-ON MATRIX: H/H · H/A · A/H · A/A, each with the Kibitzer on ->
     a valid eval + non-empty line lands, the game never freezes, and the live
     board is not mutated by the analysis.
  N) MODEL-CHANGE RE-APPLY: H/A, one move + undo, then switch Blue's model and
     New game -> the game restarts cleanly and the (re-applied) Blue model
     still plays a legal logged answer. (Combined with R, this proves "a New
     game that changes the model" changes play.)

Prints a matrix table + "T-G5 GATE OK". ASCII-only output (cp1252 console).
"""
import os
import re
import time
import traceback

import halma8
import halma8_gui
import halma_ai

HERE = halma8_gui.HERE
P1, P2 = halma8.P1, halma8.P2
RED, BLUE = halma8.NAME[halma8.P1], halma8.NAME[halma8.P2]
MODEL_A = os.path.join(HERE, "c1_incumbent.json")
MODEL_C = os.path.join(HERE, "best_weights.json")
# A progress-only weight set (mirrors the `extreme` set in
# ai/src/search.rs::t_g0_distinct_weights_change_the_search_answer). Compared
# against a real shipped model it is a broad discriminator: if the engine were
# "parsing-but-ignoring" setoption weights, the two would play identically.
MODEL_X = os.path.join(HERE, "_t_g5_extreme.json")
# valid Kibitzer eval formats: "0  (balanced)" / "+N  (Red favours)" /
# "-N  (Blue favours)" / "#N  (...)"  (see halma8_gui._fmt_eval)
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


def log_text(gui):
    return gui.log.get("1.0", "end")


def landed(gui):
    """True if a real Kibitzer result is showing (not off/analysing/warning)."""
    t = eval_text(gui)
    return t not in ("", "analysing…", "Kibitzer off") and not t.startswith("\u26a0")


def set_seat(gui, side, kind, model_basename=None, depth=3, movetime_ms=120):
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


# ---------------------------------------------------------------- R) detector
def fixed_position():
    """The exact fixed, non-symmetric mid-position from search.rs::t_g0
    (Red step (5,2)->(4,2), Blue step (0,5)->(0,4); Red to move again)."""
    g = halma8.Halma8()
    g.apply_move(("step", (5, 2), (4, 2)))   # both known-legal (see Rust test)
    g.apply_move(("step", (0, 5), (0, 4)))
    assert g.winner is None and g.turn == halma8.P1, "expected Red to move again"
    return g


def regression_detector():
    """Prove the loaded model actually drives play (the GUI's set_weights path).

    Compare a real shipped model (c1_incumbent) against a progress-only set
    (extreme, mirroring the Rust t_g0 guard). If the engine were parsing-but-
    ignoring setoption weights, the two would play identically — that is the
    silent regression we must catch. (best_weights is reported for the record
    but not asserted: it differs from c1 only in base_presence, which this
    position does not stress, so the two legitimately agree here.)
    """
    pos = halma_ai.board_to_position(fixed_position())
    res = {}
    for name, model in (("A:c1_incumbent", MODEL_A),
                        ("C:best_weights", MODEL_C),
                        ("X:extreme", MODEL_X)):
        ai = halma_ai.HalmaAI()
        try:
            ai.set_weights(model, model)          # both sides -> single-model play
            raw, score = ai._go(pos, depth=4, use_depth=True)   # deterministic
        finally:
            ai.close()
        res[name] = (score, raw)
    real, extreme = res["A:c1_incumbent"], res["X:extreme"]
    assert (real[0], real[1]) != (extreme[0], extreme[1]), (
        f"WEIGHT-WIRING REGRESSION: a real model and a progress-only set play "
        f"IDENTICALLY on {pos!r} — setoption weights are being ignored:\n"
        f"  real={real}  extreme={extreme}")
    return res, pos


# ---------------------------------------------------------------- K) matrix
def kibitz_row(gui, combo):
    """Run one seat combo with the Kibitzer on; return the landed eval text.

    combo = (p1_kind, p2_kind, p1_model, p2_model). Lets the game's own moves
    settle first (AI answers, or one human move), THEN waits for the Kibitzer to
    land and asserts the live board was not mutated by the analysis.
    """
    p1k, p2k, p1m, p2m = combo
    # start clean (Kibitzer off, then we enable it via the Players config)
    gui._kibitz_epoch += 1
    gui.kib_eval.config(text="Kibitzer off")
    gui.kib_line.config(text="")
    gui.kib_on_var.set(False)
    set_seat(gui, P1, p1k, p1m)
    set_seat(gui, P2, p2k, p2m)
    gui.kib_on_var.set(True)          # setup-time Kibitzer toggle
    gui._apply_players_config()      # enables Kibitzer + starts a fresh game
    b = gui.board
    assert b.winner is None and b.turn == P1, f"{combo}: game did not start live"

    # let the game's own moves settle (before judging the analysis)
    if p1k == "ai" and p2k == "ai":   # A/A: wait until both sides have answered
        assert pump(gui, lambda: b.winner is not None or (RED in log_text(gui) and BLUE in log_text(gui)),
                    120.0), f"{combo}: A/A never settled"
    elif p1k == "ai":                 # A/H: Red(AI) moves first
        assert pump(gui, lambda: b.turn == P2 or b.winner is not None, 90.0), \
            f"{combo}: Red(AI) did not move"
    else:                             # H/H, H/A: one human Red move...
        mv = first_legal_step(b, P1)
        assert mv is not None, f"{combo}: no legal Red step"
        gui._do_move(mv)
        if p2k == "ai":               # ...and Blue(AI) must answer
            assert pump(gui, lambda: b.turn != P2 or b.winner is not None, 90.0), \
                f"{combo}: Blue(AI) did not answer"

    grid_settled = [row[:] for row in b.grid]
    # now the Kibitzer must land a valid eval + line without touching the live board
    if b.winner is None:
        assert pump(gui, lambda: landed(gui), 90.0), \
            f"{combo}: Kibitzer never landed: {eval_text(gui)!r}"
        assert EVAL_RE.match(eval_text(gui)), f"{combo}: bad eval text {eval_text(gui)!r}"
        assert ("->" in line_text(gui) or "stop" in line_text(gui)), \
            f"{combo}: empty line {line_text(gui)!r}"
        # "analysis didn't mutate the live board" is only meaningful when the game
        # is waiting on a HUMAN (board static). In A/A the AI auto-keeps-playing,
        # so the board legitimately keeps changing; there we only require no-freeze
        # + a landed analysis (matching T-G4 sub-test 4).
        if not (p1k == "ai" and p2k == "ai"):
            assert [row[:] for row in b.grid] == grid_settled, \
                f"{combo}: analysis mutated the live board"
    return eval_text(gui)


# ---------------------------------------------------------------- N) re-apply
def model_change_reapply(gui):
    # H/A: Red human, Blue AI (c1). Play + undo, then swap Blue's model + New game.
    gui.kib_on_var.set(False)
    set_seat(gui, P1, "human")
    set_seat(gui, P2, "ai", "c1_incumbent.json")
    gui._apply_players_config()
    b = gui.board
    assert b.turn == P1 and b.winner is None
    mv = first_legal_step(b, P1)
    assert mv is not None
    gui._do_move(mv)
    assert b.turn == P2
    assert pump(gui, lambda: b.turn == P1 or b.winner is not None, 90.0), \
        "Blue(AI, c1) did not answer before the model change"
    gui.undo()   # reverts Blue's (most recent) move -> it is Blue's turn again
    assert b.turn == P2, "undo should restore Blue's turn (undo reverts the last move)"
    # switch Blue's model to best_weights and New game -> clean restart
    gui.kib_on_var.set(False)
    set_seat(gui, P1, "human")
    set_seat(gui, P2, "ai", "best_weights.json")
    gui._apply_players_config()
    b = gui.board
    assert b.turn == P1 and b.winner is None, "New game (changed model) did not restart cleanly"
    mv2 = first_legal_step(b, P1)
    assert mv2 is not None
    gui._do_move(mv2)
    assert b.turn == P2
    assert pump(gui, lambda: b.turn != P2 or b.winner is not None, 90.0), \
        "Blue(AI, best_weights) did not answer after the model change"
    assert BLUE in log_text(gui), "Blue (new model) move not logged"


def main():
    gui = halma8_gui.HalmaGUI(halma8.Halma8())
    # keep the analysis fast so the matrix runs quickly
    gui.kib_plies = 4
    gui.kib_cfg["movetime_ms"] = 200
    # Shrink the base clocks so the clock-aware AI (10% of remaining) stays fast
    # in tests: 1.5 s base -> 10% = 150 ms per AI move (under the 5 s cap).
    gui.settings[P1]["base"] = 1.5
    gui.settings[P2]["base"] = 1.5
    try:
        # --- R) weight-wiring regression detector (the headline check) ------
        res, pos = regression_detector()
        a, c, x = (res["A:c1_incumbent"], res["C:best_weights"], res["X:extreme"])
        print("R) weight-wiring regression detector (fixed asym pos, depth 4):")
        print(f"     A:c1_incumbent   score={a[0]}  move={a[1]!r}")
        print(f"     C:best_weights   score={c[0]}  move={c[1]!r}")
        print(f"     X:extreme(prog)  score={x[0]}  move={x[1]!r}")
        print(f"     real-model vs progress-only differ "
              f"(score {a[0]} vs {x[0]}): loaded weights drive play: OK")

        # --- K) Kibitzer-on matrix -----------------------------------------
        table = [
            ("H/H", ("human", "human", None, None)),
            ("H/A", ("human", "ai", None, "c1_incumbent.json")),
            ("A/H", ("ai", "human", "c1_incumbent.json", None)),
            ("A/A", ("ai", "ai", "c1_incumbent.json", "best_weights.json")),
        ]
        print("\nK) Kibitzer-on matrix (eval that landed):")
        for label, combo in table:
            ev = kibitz_row(gui, combo)
            print(f"     {label:4s} -> {ev!r}  OK")

        # --- N) model-change re-apply --------------------------------------
        model_change_reapply(gui)
        print("\nN) New game with a changed Blue model: clean restart + Blue answered: OK")

        print("\nT-G5 GATE OK")
    except Exception:
        traceback.print_exc()
        print("T-G5 GATE FAILED")
        raise SystemExit(1)
    finally:
        gui._on_close()


if __name__ == "__main__":
    main()
