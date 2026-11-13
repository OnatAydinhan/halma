"""Reproduce the GUI's exact driver sequence for A/A after H/A, with timing."""
import os
import time

import halma8
import halma_ai

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.join(HERE, "c1_incumbent.json")


def t(label):
    print(f"[{time.monotonic() % 3600:8.3f}] {label}", flush=True)


def main():
    ai = halma_ai.HalmaAI(timeout=8.0)
    try:
        t("handshake done")
        ai.set_weights(None, C1)  # H/A apply
        t("set_weights(None, C1)")
        # H/A: HUMAN steps first (Red), then Blue AI answers — exact GUI order
        g = halma8.Halma8()
        g.apply_move(("step", (5, 0), (4, 0)))  # the move the smoke used
        act2, score2 = ai.move_with_score(halma_ai.board_to_position(g),
                                          movetime_ms=500)
        t(f"P2 answer: {act2!r} score={score2}")
        if act2 != "stop":
            g.apply_move(act2)
            if g.chain_active:  # close the chain the same way the GUI does
                act3, _ = ai.move_with_score(halma_ai.board_to_position(g),
                                             movetime_ms=500)
                t(f"chain follow-up: {act3!r}")
                if act3 == "stop":
                    g.end_turn()
                else:
                    g.apply_move(act3)
        t("H/A phase done (turn now {})".format(g.turn))
        # A/A apply: re-point both weights mid-conversation
        ai.set_weights(C1, C1)
        t("set_weights(C1, C1) — mid conversation")
        # A/A: fresh start, P1 to move
        s = halma8.Halma8()
        act3, score3 = ai.move_with_score(halma_ai.board_to_position(s),
                                          movetime_ms=500)
        t(f"A/A P1 move: {act3!r} score={score3}")
        # one more: P2 answer
        s.apply_move(act3)
        if s.turn == halma8.P2 and s.winner is None:
            act4, score4 = ai.move_with_score(halma_ai.board_to_position(s),
                                              movetime_ms=500)
            t(f"A/A P2 answer: {act4!r} score={score4}")
        print("DBG1 OK")
    finally:
        ai.close()


if __name__ == "__main__":
    main()
