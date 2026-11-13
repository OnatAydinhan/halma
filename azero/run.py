"""CLI entry point for the AlphaZero NN pipeline (Part 3, T-6).

Usage:
  py azero/run.py --rounds 5 --games 500 --sims 200 --epochs 10 --device cuda
  py azero/run.py --rounds 1 --games 5 --sims 10 --epochs 2 --device cpu
  py azero/run.py --resume azero/checkpoints/round_3.pt --rounds 5 ...
"""

import argparse
import os
import sys

# Ensure the project root is on the path so `import halma8` works.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from azero.model import HalmaNet
from azero.pipeline import run_pipeline
from azero.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Run the AlphaZero self-play + training pipeline for check_halma."
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--games", type=int, default=500, help="Games per round for self-play")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per round")
    parser.add_argument("--batch", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from (e.g. azero/checkpoints/round_3.pt)"
    )
    parser.add_argument("--max-moves", type=int, default=100, help="Max moves per game")
    parser.add_argument("--elo-games", type=int, default=40, help="Games per round for the Elo eval (vs current champion)")
    parser.add_argument("--elo-sims", type=int, default=100, help="MCTS sims per move for the Elo eval (vs current champion)")
    parser.add_argument(
        "--champion", type=str, default=None,
        help="Checkpoint that seeds the 'champion' opponent. Default (auto): "
             "champion.pt from a previous run, else the latest round_*.pt, else "
             "the frozen baseline. Each round plays against the current champion "
             "and promotes it if the round wins (score > 0.5)."
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="T-9: self-play OS processes per round (default 8). Each runs serial "
             "MCTS in Python and shares the 4090; 8-12 is good on a 24-core box."
    )
    parser.add_argument(
        "--mcts-batch", type=int, default=32,
        help="T-10: MCTS leaves evaluated per batched net forward pass (default 32). "
             "16-64 recommended on CUDA; composes with --workers."
    )
    parser.add_argument(
        "--value-teacher", type=float, default=0.0,
        help="Bootstrap weight W in [0,1] on the value target from the hand-crafted "
             "eval (best_weights.json). 0 = pure self-play (default). 1 = value head "
             "distills the eval; use for a jump-start, then fade to 0 for pure self-play."
    )
    parser.add_argument(
        "--teacher-weights", type=str, default=None,
        help="Path to eval weights JSON for --value-teacher (default: best_weights.json)."
    )
    parser.add_argument(
        "--teacher-scale", type=float, default=0.0,
        help="Divisor for tanh(eval/scale); 0 (default) auto-calibrates from sample positions."
    )
    parser.add_argument(
        "--dist-teacher", type=float, default=0.0,
        help="T-13: bootstrap weight W in [0,1] on the value target from the "
             "distance-to-enemy-base advancement advantage (a focused 'getting "
             "closer to the win' signal). 0 = off (default). Blends with "
             "--value-teacher; the residual weight stays on the MCTS/outcome "
             "target. Fade to 0 for pure self-play."
    )
    parser.add_argument(
        "--dist-scale", type=float, default=0.0,
        help="T-13: divisor for tanh(advantage/scale); 0 (default) auto-calibrates "
             "from sample positions."
    )
    parser.add_argument(
        "--endgame", type=int, default=0,
        help="T-14: endgame near-win games per round (base at --endgame-full/9, "
             "to-move=attacker) played ON TOP of --games fresh-board games. These "
             "give the value head a true terminal +/-1 the all-draw fresh games "
             "never produce. 0 = off (default)."
    )
    parser.add_argument(
        "--endgame-full", type=int, default=8,
        help="T-14: target base fullness (out of 9) for the endgame near-win "
             "positions (default 8; 9/9 + invader would already be terminal)."
    )
    parser.add_argument(
        "--endgame-sims", type=int, default=200,
        help="T-14 fix: MCTS sims per move for the endgame near-win games "
             "(default 200). Separate from --sims because the net's prior on the "
             "winning move is near-uniform: at a low sim budget the winning "
             "action gets starved and the game never converts, so the true +/-1 "
             "target is never generated. Conversion measured ~9/10 at 200 vs "
             "~3/10 at 30."
    )
    parser.add_argument(
        "--endgame-noise", action="store_true", default=False,
        help="T-14 fix: apply Dirichlet root noise to the endgame near-win games "
             "(default OFF). Keep OFF: noise is counterproductive here because "
             "the low-prior winning move gets starved to ~0 visits at low sims."
    )
    parser.add_argument(
        "--finish-rate", type=int, default=0,
        help="T-14: held-out near-win positions per round for the finish-rate "
             "metric (the one metric that can SEE the finish the full-game Elo is "
             "blind to). 0 = off (default); otherwise reported each round."
    )
    parser.add_argument(
        "--finish-rate-sims", type=int, default=100,
        help="T-14: MCTS sims per move for the finish-rate metric (default 100). "
             "Keep this >= the sim budget the net needs to convert (it can "
             "convert at ~200 but ~0 at 30), else the metric under-reports."
    )
    args = parser.parse_args()

    # Verify device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    model = HalmaNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_round = 1

    if args.resume:
        print(f"Resuming from {args.resume}")
        round_no = load_checkpoint(model, optimizer, args.resume, args.device)
        start_round = round_no + 1
        print(f"Resuming at round {start_round}")

    model.to(args.device)

    run_pipeline(
        n_rounds=args.rounds,
        games_per_round=args.games,
        n_sims=args.sims,
        n_train_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        start_round=start_round,
        model=model,
        optimizer=optimizer,
        max_moves=args.max_moves,
        elo_n_games=args.elo_games,
        elo_n_sims=args.elo_sims,
        workers=args.workers,
        mcts_batch=args.mcts_batch,
        champion=args.champion,
        value_teacher=args.value_teacher,
        teacher_weights_path=args.teacher_weights,
        teacher_scale=args.teacher_scale,
        dist_teacher=args.dist_teacher,
        dist_scale=args.dist_scale,
        endgame_n=args.endgame,
        endgame_full=args.endgame_full,
        endgame_n_sims=args.endgame_sims,
        endgame_add_noise=args.endgame_noise,
        finish_rate_n=args.finish_rate,
        finish_rate_sims=args.finish_rate_sims,
    )


if __name__ == "__main__":
    main()
