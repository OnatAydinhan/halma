"""Same A/A-after-H/A repro, with spies on _maybe_ai/_ai_take_turn/_ai.move."""
import os
import time

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
    gui._players_error = lambda m: print("!! validation:", m, flush=True)
    gui.kib_on_var.set(False)

    # spies
    orig_maybe = gui._maybe_ai
    def maybe_ai(delay_ms=200):
        print(f"SPY maybe_ai (busy={gui._ai_busy}, turn={gui.board.turn})", flush=True)
        return orig_maybe(delay_ms)
    gui._maybe_ai = maybe_ai

    orig_take = halma8_gui.HalmaGUI._ai_take_turn
    def take(self):
        print(f"SPY take ENTER busy={self._ai_busy} turn={self.board.turn}", flush=True)
        t0 = time.monotonic()
        try:
            return orig_take(self)
        finally:
            print(f"SPY take EXIT after {time.monotonic()-t0:.2f}s", flush=True)
    halma8_gui.HalmaGUI._ai_take_turn = take

    orig_move = gui._ai.move
    def move(position, depth=None, movetime_ms=None):
        print(f"SPY move ENTER movetime={movetime_ms}", flush=True)
        t0 = time.monotonic()
        try:
            r = orig_move(position, depth, movetime_ms)
        except BaseException as e:
            print(f"SPY move RAISED {type(e).__name__}: {e} "
                  f"after {time.monotonic()-t0:.2f}s", flush=True)
            raise
        print(f"SPY move OK {r!r} after {time.monotonic()-t0:.2f}s", flush=True)
        return r
    gui._ai.move = move

    end = time.time() + 45
    def pump_until(pred):
        while time.time() < end:
            gui.root.update()
            if pred():
                return True
            time.sleep(0.2)
        return False

    # H/A
    set_seat(gui, P1, "human")
    set_seat(gui, P2, "ai")
    gui._apply_players_config()
    gui._do_move(("step", (5, 0), (4, 0)))
    print("STEP: human move played", flush=True)
    if not pump_until(lambda: len(gui.undo_stack) >= 2):
        print("H/A answer never came", flush=True)
    gui.undo()
    print("STEP: undid", flush=True)

    # A/A
    set_seat(gui, P1, "ai")
    set_seat(gui, P2, "ai")
    gui._apply_players_config()
    print("STEP: A/A applied", flush=True)
    if not pump_until(lambda: len(gui.undo_stack) >= 4):
        print("A/A auto-run stalled", flush=True)
        print(f"state: turn={gui.board.turn} undo={len(gui.undo_stack)} "
              f"busy={gui._ai_busy} status={gui.status.cget('text')!r}", flush=True)

    gui._ai.close()
    gui.root.destroy()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
