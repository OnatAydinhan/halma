"""A/A-after-H/A with live state printing to find where the GUI loop stalls."""
import os
import time
import traceback

import halma8
import halma8_gui

HERE = halma8_gui.HERE
C1 = os.path.join(HERE, "c1_incumbent.json")
P1, P2 = halma8_gui.P1, halma8_gui.P2


def set_seat(gui, side, kind, model=C1):
    gui.seat_kind_var[side].set(kind)
    if kind == "ai":
        gui.seat_model_var[side].set(os.path.basename(model))
    gui._on_seat_kind(side)


def main():
    gui = halma8_gui.HalmaGUI(halma8.Halma8())
    gui._players_error = lambda m: print("!! validation:", m)
    gui.kib_on_var.set(False)

    def dump(tag):
        print(f"--- {tag}: turn={gui.board.turn} undo={len(gui.undo_stack)} "
              f"busy={gui._ai_busy} winner={gui.board.winner}\n"
              f"    status={gui.status.cget('text')!r}", flush=True)

    # H/A
    set_seat(gui, P1, "human")
    set_seat(gui, P2, "ai")
    gui._apply_players_config()
    gui._do_move(("step", (5, 0), (4, 0)))
    dump("after human step (AI should answer)")
    end = time.time() + 20
    while len(gui.undo_stack) < 2 and time.time() < end:
        gui.root.update()
        time.sleep(0.05)
    dump("after H/A answer")

    gui.undo()
    dump("after undo")

    # A/A
    set_seat(gui, P1, "ai")
    set_seat(gui, P2, "ai")
    gui._apply_players_config()
    dump("after A/A apply")

    end = time.time() + 40
    last = ""
    while len(gui.undo_stack) < 4 and time.time() < end:
        gui.root.update()
        cur = (gui.board.turn, len(gui.undo_stack), gui._ai_busy,
               gui.status.cget("text"))
        if cur != last:
            print(f"... turn={cur[0]} undo={cur[1]} busy={cur[2]} "
                  f"status={cur[3]!r}", flush=True)
            last = cur
        time.sleep(0.2)
    dump("final")
    log = gui.log.get("1.0", "end-1c").strip().splitlines()
    print(f"LOG ({len(log)}):")
    for l in log:
        print("   ", l)
    gui._ai.close()
    gui.root.destroy()
    print("DONE")


if __name__ == "__main__":
    main()
