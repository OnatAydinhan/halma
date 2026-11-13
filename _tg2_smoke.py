"""T-G2 non-interactive smoke: builds the real GUI, sets all four seat
combos, plays H/A and A/A fragments, checks weights wiring + undo."""
import os
import time

import halma8
import halma8_gui

HERE = halma8_gui.HERE
C1 = os.path.join(HERE, "c1_incumbent.json")


def pump(root, pred, secs=45.0):
    end = time.time() + secs
    while time.time() < end:
        root.update()
        if pred():
            return True
        time.sleep(0.05)
    return bool(pred())


def set_seat(gui, side, kind, model=C1, depth=4, ms=500):
    gui.seat_kind_var[side].set(kind)
    if kind == "ai":
        gui.seat_model_var[side].set(os.path.basename(model))
    gui.seat_depth_var[side].set(depth)
    gui.seat_mt_var[side].set(ms)
    gui._on_seat_kind(side)


def main():
    gui = halma8_gui.HalmaGUI(halma8.Halma8())
    errors = []

    # 1) all four combos are settable and validate correctly
    gui._players_error = lambda msg: errors.append(msg) or "ERROR:" + msg
    gui.kib_on_var.set(False)

    for combo in (("human", "human"), ("ai", "human"), ("human", "ai"), ("ai", "ai")):
        set_seat(gui, halma8_gui.P1, combo[0])
        set_seat(gui, halma8_gui.P2, combo[1])
        ok = gui._read_players()
        label = f"{combo[0]}/{combo[1]}"
        if combo[0] == "ai" or combo[1] == "ai":
            assert ok, f"combo {label} should validate with a model set: {errors}"
            assert gui.seat_cfg[halma8_gui.P1]["model"] == C1 or combo[0] == "human"
            assert gui.seat_cfg[halma8_gui.P2]["model"] == C1 or combo[1] == "human"
        else:
            assert ok, f"combo {label} (H/H) should validate"
        print(f"COMBO {label}: valid={ok}  cfg=({gui.seat_cfg[halma8_gui.P1]['kind']},"
              f" {gui.seat_cfg[halma8_gui.P2]['kind']})")

    # AI seat with NO model must be rejected (validation gate)
    gui.seat_kind_var[halma8_gui.P2].set("ai")
    gui.seat_model_var[halma8_gui.P2].set("nope_missing.json")
    gui._on_seat_kind(halma8_gui.P2)
    assert not gui._read_players(), "missing model must fail validation"
    print("VALIDATION: missing model correctly rejected")

    # 2) H/A: human (Red) plays one move, Blue AI answers
    gui._players_error = halma8_gui.HalmaGUI._players_error.__get__(gui)
    set_seat(gui, halma8_gui.P1, "human")
    set_seat(gui, halma8_gui.P2, "ai")
    gui.kib_on_var.set(False)
    gui._apply_players_config()
    assert gui.board.turn == halma8_gui.P1
    # a legal Red step: any P1 piece with a step
    move = None
    for r in range(halma8.Halma8.SIZE):
        for c in range(halma8.Halma8.SIZE):
            if gui.board.grid[r][c] == halma8.P1:
                for dst in gui.board.legal_steps(r, c):
                    move = ("step", (r, c), dst)
                    break
            if move:
                break
        if move:
            break
    assert move, "no Red step found (impossible at start)"
    gui._do_move(move)
    assert gui.board.turn == halma8_gui.P2
    ok = pump(gui.root, lambda: len(gui.undo_stack) >= 2, secs=45)
    assert ok, "Blue AI did not answer the human move"
    log = gui.log.get("1.0", "end-1c").strip().splitlines()
    print(f"HALMA: log={log!r}")
    # weights wiring: shared child got P2=C1 only (Red is human -> None)
    print("WEIGHTS after H/A apply:", gui._ai._weights)
    assert gui._ai._weights == (None, C1), gui._ai._weights
    # undo works mid-game
    n_before = len(gui.undo_stack)
    gui.undo()
    assert len(gui.undo_stack) == n_before - 1
    print("UNDO OK — turn now", gui.board.turn)

    # 3) A/A auto-run: Red (P1) must move first without any input.
    # (A single halma turn can be a long jump chain — one snapshot per jump —
    # so wait for Red to hand over the turn, not for a fixed stack depth.)
    set_seat(gui, halma8_gui.P1, "ai")
    set_seat(gui, halma8_gui.P2, "ai")
    gui.kib_on_var.set(False)
    gui._apply_players_config()
    assert gui.board.turn == halma8_gui.P1 and gui.undo_stack == []
    ok = pump(gui.root, lambda: gui.board.turn == halma8_gui.P2, secs=60)
    assert ok, f"A/A Red never completed its first turn; log={gui.log.get('1.0','end-1c')!r}"
    log = gui.log.get("1.0", "end-1c").strip().splitlines()
    assert log and log[0].startswith("1 P1 (Red):"), f"first A/A move not Red: {log[:2]}"
    print(f"AAA AUTO-RUN: {len(log)} logged moves")
    for line in log:
        print("  ", line)
    assert gui._ai._weights == (C1, C1), gui._ai._weights
    print("WEIGHTS after A/A apply:", gui._ai._weights)

    gui._ai.close()
    gui.root.destroy()
    print("TG2 SMOKE OK")


if __name__ == "__main__":
    main()
