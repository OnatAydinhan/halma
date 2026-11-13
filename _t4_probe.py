"""T-G4 probe: root scores at the symmetric start position.

Compares complete depth-limited searches (go depth N, deterministic) against
time-bounded ones (go movetime, cutoff-biased), across weight sets.
ASCII-only output.
"""
import os

import halma8
import halma_ai

HERE = halma_ai.HERE
START = halma_ai.board_to_position(halma8.Halma8())
MODELS = [
    ("none (default)", None),
    ("c1_incumbent", os.path.join(HERE, "c1_incumbent.json")),
    ("best_weights", os.path.join(HERE, "best_weights.json")),
]


def probe(ai, pos, depth):
    raw, score = ai._go(pos, depth, use_depth=True)
    return score


def main():
    for label, model in MODELS:
        ai = halma_ai.HalmaAI()
        try:
            if model is not None:
                ai.set_weights(model, model)
            depths = []
            for d in (1, 2, 3, 4, 5, 6):
                depths.append(f"d{d}={probe(ai, START, d)}")
            mts = []
            for _ in range(3):
                _raw, s = ai._go(START, movetime_ms=500)
                mts.append(s)
            print(f"{label:15s} | {' '.join(depths)} | movetime500 x3: {mts}")
        finally:
            ai.close()


if __name__ == "__main__":
    main()
