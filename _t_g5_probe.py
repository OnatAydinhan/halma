"""Quick (Tk-free) probe: which model pair separates on the fixed asymmetric
position, at which depth. Used to pick a reliable weight-wiring discriminator
for _t_g5_test.py (the c1 vs best pair are too close to discriminate)."""
import os
import halma8
import halma_ai

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "c1_incumbent": os.path.join(HERE, "c1_incumbent.json"),
    "best_weights": os.path.join(HERE, "best_weights.json"),
    "extreme": os.path.join(HERE, "_t_g5_extreme.json"),
}


def fixed_position():
    g = halma8.Halma8()
    g.apply_move(("step", (5, 2), (4, 2)))
    g.apply_move(("step", (0, 5), (0, 4)))
    return g


def answer(model_path, depth):
    ai = halma_ai.HalmaAI()
    try:
        ai.set_weights(model_path, model_path)
        raw, score = ai._go(halma_ai.board_to_position(fixed_position()),
                            depth=depth, use_depth=True)
    finally:
        ai.close()
    return score, raw


def main():
    pos = halma_ai.board_to_position(fixed_position())
    print(f"position: {pos}")
    for depth in (4, 6):
        print(f"\n--- depth {depth} ---")
        res = {name: answer(path, depth) for name, path in MODELS.items()}
        for name, (score, raw) in res.items():
            print(f"  {name:14s} score={score}  move={raw!r}")
        # report pairwise separation
        names = list(res)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                same = res[names[i]] == res[names[j]]
                print(f"  {names[i]} vs {names[j]}: {'IDENTICAL' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
