"""Baseline-relative Elo evaluation for the AlphaZero NN (Part 3, T-6).

Compares the current model against a frozen baseline (random-init, pinned
at 1000 Elo) via a fixed set of self-matches. The score-to-Elo formula is
simple:  elo = 1000 + K * (score - 0.5), with K=32.
"""

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from halma8 import Halma8

from azero.board import action_index, decode_action, encode_board
from azero.mcts import mcts_search

K = 64


def score_to_elo(score):
    """Convert an observed win-rate vs baseline to a baseline-relative Elo."""
    return 1000.0 + K * (score - 0.5)


def _elog(gi, total, res, elapsed, rate, tag=""):
    """One line of per-game progress (ASCII only; flushed so it shows live)."""
    print("[elo] {tag}game {gi}/{total} = {res}   elapsed={el:.0f}s  {rt:.2f} g/s".format(
        tag=tag, gi=gi, total=total, res=res, el=elapsed, rt=rate), flush=True)


def _nn_move(state, model, device, n_sims, mcts_batch=1):
    """Get the NN's best action for the current state. Returns (action, policy)."""
    action, policy = mcts_search(
        state, model, n_sims=n_sims, device=device, add_noise=False,
        batch_size=mcts_batch,
    )
    return action


def _play_eval_game(model, baseline, model_is_p1, device, n_sims, max_moves,
                    mcts_batch=1):
    """Play one NN-vs-baseline game. Returns 'win'/'loss'/'draw' (model's view)."""
    state = Halma8()
    for _ in range(max_moves):
        if state.winner is not None:
            break
        mover = state.turn
        # P1 always moves first; determine which model plays this move
        use_model = (mover == 1) == model_is_p1
        active_model = model if use_model else baseline

        action = _nn_move(state, active_model, device, n_sims, mcts_batch)
        if action is None:
            # Stuck; game ends with no winner.
            break
        if action[0] == "stop":
            state.end_turn()
        else:
            state.apply_move(decode_action(action_index(action)))

    if state.winner is not None:
        model_side = 1 if model_is_p1 else 2
        return "win" if state.winner == model_side else "loss"
    return "draw"


def _score_from_counts(counts):
    """(score, elo) from a {'win','loss','draw'} count dict (model's view)."""
    wins = counts.get("win", 0)
    draws = counts.get("draw", 0)
    losses = counts.get("loss", 0)
    total = wins + draws + losses
    score = 0.5 if total == 0 else (wins + 0.5 * draws) / total
    return score, score_to_elo(score)


def elo_worker(args):
    """OS-process Elo worker (analog of selfplay_worker) — plays a chunk of games.

    Module-level so it pickles by reference under Windows `spawn`. Both nets
    travel as CPU state dicts (a CUDA module is not picklable). Returns a
    picklable {'win','loss','draw'} count dict for this chunk.

    `args`:
      model_state    : HalmaNet state dict (CPU) — the model under test
      baseline_state : HalmaNet state dict (CPU) — the frozen baseline
      games          : list of (game_index, model_is_p1) for this chunk
      n_sims, max_moves : ints
      device         : "cuda" or "cpu"
    """
    import torch
    from azero.model import HalmaNet

    def _load(sd, dev):
        m = HalmaNet()
        m.load_state_dict(sd)
        m.eval()
        m.to(dev)
        return m

    device = args["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = _load(args["model_state"], device)
    baseline = _load(args["baseline_state"], device)

    games = args["games"]
    total = args.get("total", len(games))
    w = args.get("worker_id", 0)
    n_batch = args.get("mcts_batch", 1)
    counts = {"win": 0, "loss": 0, "draw": 0}
    t0 = time.time()
    for done, (_game, model_is_p1) in enumerate(games, start=1):
        res = _play_eval_game(model, baseline, model_is_p1, device,
                              args["n_sims"], args["max_moves"], mcts_batch=n_batch)
        counts[res] += 1
        el = time.time() - t0
        _elog(_game, total, res, el, (done / el if el > 0 else 0.0), tag="w%d " % w)
    return counts


def evaluate_elo(
    model,
    baseline_model,
    n_games=40,
    n_sims=100,
    device="cpu",
    max_moves=100,
    workers=1,
    mcts_batch=1,
):
    """Play `n_games` NN-vs-baseline games, alternating who is P1.

    Returns (score, elo) where score is from `model`'s perspective:
      win=1, loss=0, draw (max_moves)=0.5.

    `workers` (default 1): if > 1, the games are split into `workers` spawned
    OS processes that share the GPU and their (win/loss/draw) counts are
    aggregated. Result semantics are identical to the sequential path; only the
    self-play wall-clock drops.

    `mcts_batch` (default 1): MCTS leaves per batched net forward pass, forwarded
    to every search. 1 = the original (slow) path; self-play uses 32. Raises it
    to match self-play for a several-fold speedup.
    """
    model.eval()
    baseline_model.eval()

    # Global double-blind seat: model is P1 on even game indices, P2 on odd.
    games = [(g, (g % 2 == 0)) for g in range(n_games)]

    if workers <= 1:
        counts = {"win": 0, "loss": 0, "draw": 0}
        t0 = time.time()
        for done, (_g, model_is_p1) in enumerate(games, start=1):
            res = _play_eval_game(model, baseline_model, model_is_p1,
                                  device, n_sims, max_moves, mcts_batch=mcts_batch)
            counts[res] += 1
            el = time.time() - t0
            _elog(_g, n_games, res, el, (done / el if el > 0 else 0.0))
    else:
        import multiprocessing as mp

        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        baseline_state = {k: v.detach().cpu() for k, v in baseline_model.state_dict().items()}

        n_workers = min(workers, n_games)
        base = n_games // n_workers
        rem = n_games % n_workers

        chunks = []
        idx = 0
        for _w in range(n_workers):
            n = base + (1 if _w < rem else 0)
            if n > 0:
                chunks.append(games[idx:idx + n])
                idx += n

        # Spawn re-imports azero.elo, so make sure the project root is on the
        # child's import path (PYTHONPATH is inherited by the fresh interpreter).
        ppath = os.environ.get("PYTHONPATH", "")
        if _PROJECT_ROOT not in [p for p in ppath.split(os.pathsep) if p]:
            os.environ["PYTHONPATH"] = (ppath + os.pathsep if ppath else "") + _PROJECT_ROOT

        ctx = mp.get_context("spawn")
        work = [
            {"model_state": model_state, "baseline_state": baseline_state,
             "games": chunk, "n_sims": n_sims, "max_moves": max_moves, "device": device,
             "mcts_batch": mcts_batch, "total": n_games, "worker_id": k}
            for k, chunk in enumerate(chunks)
        ]
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(elo_worker, work)

        counts = {"win": 0, "loss": 0, "draw": 0}
        for chunk_counts in results:
            for k in counts:
                counts[k] += chunk_counts.get(k, 0)

    return _score_from_counts(counts)


def main():
    """Elo-vs-baseline for a trained checkpoint (eval only, no self-play).

    Loads `--checkpoint` (a round_N.pt or final.pt) and the frozen random
    baseline, plays N self-matches, and reports score + Elo — the same metric
    the pipeline logs.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure a checkpoint's Elo vs the frozen random baseline."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint to score (e.g. azero/checkpoints/round_2.pt)")
    parser.add_argument("--baseline",
                        default=os.path.join(_HERE, "checkpoints", "baseline.pt"),
                        help="Frozen baseline (default: azero/checkpoints/baseline.pt)")
    parser.add_argument("--games", type=int, default=40, help="Self-matches to play")
    parser.add_argument("--sims", type=int, default=100, help="MCTS sims per move")
    parser.add_argument("--max-moves", type=int, default=100,
                        help="Max moves before a game counts as a draw")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--workers", type=int, default=8,
                        help="OS processes to split the eval across (default 8); the "
                             "4090 is time-sliced. 1 = sequential (original).")
    parser.add_argument("--mcts-batch", type=int, default=32,
                        help="MCTS leaves per batched net forward pass (default 32, "
                             "matching run.py self-play). 1 = the original slow path.")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    from azero.model import HalmaNet

    def _load(path, dev):
        m = HalmaNet()
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        m.to(dev)
        return m

    model = _load(args.checkpoint, device)
    baseline = _load(args.baseline, device)

    print(f"checkpoint : {args.checkpoint}", flush=True)
    print(f"games      : {args.games}  (n_sims={args.sims}, max_moves={args.max_moves}, "
          f"workers={args.workers}, mcts_batch={args.mcts_batch}, device={device})",
          flush=True)
    print(f"--- playing; one line per game (watch g/s to gauge pace) ---", flush=True)
    t0 = time.time()
    score, elo = evaluate_elo(
        model, baseline, n_games=args.games, n_sims=args.sims,
        device=device, max_moves=args.max_moves, workers=args.workers,
        mcts_batch=args.mcts_batch,
    )
    total = time.time() - t0
    print("-" * 60, flush=True)
    print(f"score      : {score:.3f}  (vs frozen random baseline)")
    print(f"elo        : {elo:.1f}   (= 1000 + 64*(score-0.5))")
    print(f"total      : {total:.0f}s  ({args.games} games, "
          f"{args.games / total if total > 0 else 0:.2f} g/s overall)", flush=True)


if __name__ == "__main__":
    main()
