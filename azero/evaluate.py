"""NN vs Rust engine evaluation (Part 3, T-7).

Plays the trained NN against the current-best Rust engine (the 448/540-weight
engine pointed at ``best_weights.json``). By default the seats are double-blind
-- the NN plays P1/Red in half the games and P2/Blue in the other half -- so
the result is not inflated by the first-move advantage (pass ``--no-alternate``
to revert to NN-always-P1).

The Rust engine is driven through the existing ``halma_ai.HalmaAI``
subprocess driver -- the same path the GUI uses. ``halma8.Halma8`` stays
the sole authority on legality: every move (NN or Rust) is applied through
``apply_move`` / ``end_turn``, and a move the oracle rejects aborts that
game rather than corrupting the state.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from halma8 import Halma8
from halma_ai import HalmaAI, board_to_position

from azero.board import action_index, decode_action
from azero.model import HalmaNet
from azero.mcts import mcts_search

# The "current best" Rust engine weights (448/540, see best_weights.json).
DEFAULT_WEIGHTS = os.path.join(_PROJECT_ROOT, "best_weights.json")
FINAL_CKPT = os.path.join(_HERE, "checkpoints", "final.pt")


def load_model(path, device):
    """Load a trained HalmaNet checkpoint into an eval-mode model on device."""
    model = HalmaNet()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def _nn_action(state, model, device, n_sims):
    """The NN's best action from `state` (no exploration noise)."""
    action, _policy = mcts_search(
        state, model, n_sims=n_sims, device=device, add_noise=False
    )
    return action


def evaluate_nn_vs_rust(
    model,
    n_games=50,
    n_sims=200,
    device="cuda",
    max_moves=480,
    rust_weights=DEFAULT_WEIGHTS,
    movetime_ms=500,
    timeout=30.0,
    alternate=True,
):
    """Play `n_games` games: the NN vs the Rust engine.

    Returns ``(nn_wins, rust_wins, ongoing)``, counted from the NN's
    perspective (an "ongoing" game hit `max_moves` with no winner).

    Seating: with ``alternate`` (double-blind, the default) the NN plays
    P1/Red in the odd-numbered games and P2/Blue in the even ones, so the
    first-move advantage cancels out. With ``alternate=False`` the NN is
    always P1 (the original plan behavior). Either way the Rust engine is
    pointed at ``rust_weights`` for both seats, so it is the 448/540 engine
    whichever seat it occupies. Each side's move is applied through the
    halma8 oracle.
    """
    model.eval()
    ai = HalmaAI(movetime_ms=movetime_ms, timeout=timeout)
    nn_wins = rust_wins = ongoing = 0
    try:
        ai.set_weights(rust_weights, rust_weights)
        for g in range(1, n_games + 1):
            # Double-blind: NN is P1 in the odd games, P2 in the even.
            nn_is_p1 = (g % 2 == 1) if alternate else True
            state = Halma8()
            result = "ongoing"
            try:
                for _ in range(max_moves):
                    if state.winner is not None:
                        break
                    nn_moves = (state.turn == 1) == nn_is_p1
                    if nn_moves:
                        action = _nn_action(state, model, device, n_sims)
                        if action is None:
                            break
                        if action[0] == "stop":
                            state.end_turn()
                        else:
                            state.apply_move(decode_action(action_index(action)))
                    else:
                        action = ai.move(board_to_position(state), movetime_ms=movetime_ms)
                        if action == "stop":
                            state.end_turn()
                        else:
                            state.apply_move(action)
                if state.winner is None:
                    result = "ongoing"
                else:
                    nn_side = 1 if nn_is_p1 else 2
                    result = "nn" if state.winner == nn_side else "rust"
            except Exception as e:  # a bad/illegal move or engine hiccup
                print(f"  game {g}/{n_games}: aborted ({e}); counting as ongoing")
                result = "ongoing"

            if result == "nn":
                nn_wins += 1
            elif result == "rust":
                rust_wins += 1
            else:
                ongoing += 1
            seat = "P1" if nn_is_p1 else "P2"
            print(f"  game {g}/{n_games} (NN={seat}): {result}  "
                  f"(winner={state.winner}, turn_no={state.turn_no})")
    finally:
        ai.close()
    return nn_wins, rust_wins, ongoing


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained NN (P1) against the best Rust engine (P2)."
    )
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per NN move")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--max-moves", type=int, default=480, help="Max moves before a game is 'ongoing'")
    parser.add_argument("--movetime", type=int, default=500, help="Rust engine movetime (ms) per move")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Rust engine weights JSON")
    parser.add_argument("--checkpoint", type=str, default=FINAL_CKPT, help="NN checkpoint to load")
    seat = parser.add_mutually_exclusive_group()
    seat.add_argument("--alternate", dest="alternate", action="store_true", default=True,
                      help="double-blind: NN plays P1 in half the games, P2 in the other (default)")
    seat.add_argument("--no-alternate", dest="alternate", action="store_false",
                      help="NN always plays P1 (original plan behavior)")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        print("Train first: py azero/run.py --rounds 1 --games 5 --sims 10 --epochs 2 --device cpu")
        raise SystemExit(1)
    if not os.path.exists(args.weights):
        print(f"ERROR: Rust weights not found: {args.weights}")
        raise SystemExit(1)

    print(f"loading NN from {args.checkpoint}")
    print(f"Rust engine weights: {args.weights}")
    model = load_model(args.checkpoint, device)

    seat_mode = "double-blind (NN=P1 & NN=P2)" if args.alternate else "NN=P1 only"
    print(f"playing {args.games} game(s): {seat_mode}  (n_sims={args.sims}, "
          f"max_moves={args.max_moves}, movetime={args.movetime}ms)")
    nn_wins, rust_wins, ongoing = evaluate_nn_vs_rust(
        model,
        n_games=args.games,
        n_sims=args.sims,
        device=device,
        max_moves=args.max_moves,
        rust_weights=args.weights,
        movetime_ms=args.movetime,
        alternate=args.alternate,
    )

    total = nn_wins + rust_wins + ongoing
    print("=" * 52)
    print("NN vs Rust engine" + ("  [double-blind]" if args.alternate else "  [NN=P1]"))
    print(f"  games      : {total}")
    print(f"  NN wins    : {nn_wins}")
    print(f"  Rust wins  : {rust_wins}")
    print(f"  ongoing    : {ongoing}")
    decided = nn_wins + rust_wins
    if decided:
        print(f"  NN rate    : {nn_wins / decided:.3f} of decided games")
    print("=" * 52)


if __name__ == "__main__":
    main()
