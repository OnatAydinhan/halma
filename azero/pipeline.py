"""Full self-play -> train -> evaluate pipeline (Part 3, T-6).

One command runs the complete AlphaZero loop: collect self-play data,
train, save checkpoint, evaluate Elo vs the current champion (best-trained
model so far), and log results.
"""

import json
import multiprocessing as mp
import os
import time

import torch

from azero.elo import K, evaluate_elo
from azero.model import HalmaNet
from azero.selfplay import self_play_batch, selfplay_worker
from azero.train import load_checkpoint, save_checkpoint, train

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "baseline.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "training_log.txt")


def _get_baseline(device):
    """Load or create the frozen baseline model (1000 Elo reference)."""
    if os.path.exists(BASELINE_PATH):
        model = HalmaNet()
        ckpt = torch.load(BASELINE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(device)
        return model
    # Create fresh baseline from random init
    baseline = HalmaNet()
    baseline.eval()
    baseline.to(device)
    # Save it (we don't have an optimizer state to save, use a dummy)
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    torch.save({"model_state": baseline.state_dict()}, BASELINE_PATH)
    return baseline


def _load_net(path, device):
    """Load a HalmaNet from a checkpoint file onto `device` (eval mode)."""
    model = HalmaNet()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def _latest_round_ckpt(checkpoint_dir):
    """Path of the highest-numbered round_*.pt in the dir, or None."""
    best, best_n = None, -1
    if os.path.isdir(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            if name.startswith("round_") and name.endswith(".pt"):
                try:
                    n = int(name[len("round_"):-len(".pt")])
                except ValueError:
                    continue
                if n > best_n:
                    best_n, best = n, os.path.join(checkpoint_dir, name)
    return best


def _load_champion(device, checkpoint_dir, explicit=None):
    """Resolve the 'champion' opponent this run plays against.

    Preference: `explicit` path (--champion) > a champion from a previous run
    (champion.pt + champion.json) > the latest round_*.pt > the frozen random
    baseline (1000). Returns (champion_model, champ_elo, champ_label). The
    champion's absolute Elo is 1000.0 when seeded and advances only when a
    later round beats it.
    """
    if explicit:
        return _load_net(explicit, device), 1000.0, os.path.basename(explicit)

    champ_path = os.path.join(checkpoint_dir, "champion.pt")
    champ_meta = os.path.join(checkpoint_dir, "champion.json")
    if os.path.isfile(champ_path) and os.path.isfile(champ_meta):
        with open(champ_meta) as f:
            meta = json.load(f)
        return (_load_net(champ_path, device), float(meta.get("elo", 1000.0)),
                meta.get("label", "champion"))

    latest = _latest_round_ckpt(checkpoint_dir)
    if latest:
        return _load_net(latest, device), 1000.0, os.path.basename(latest)

    return _get_baseline(device), 1000.0, "baseline(1000)"


def _promote_champion(model, elo, label, checkpoint_dir, device):
    """Persist `model` as the new champion and return an independent copy.

    The copy is reloaded from disk so it owns its own tensors (it must not
    alias the live training model, which the next round overwrites in place).
    """
    champ_path = os.path.join(checkpoint_dir, "champion.pt")
    champ_meta = os.path.join(checkpoint_dir, "champion.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        champ_path,
    )
    with open(champ_meta, "w") as f:
        json.dump({"elo": elo, "label": label, "path": os.path.abspath(champ_path)}, f, indent=2)
    return _load_net(champ_path, device)


def _parse_log_line(line):
    """key=value pairs from a training-log line (tolerates extra tokens)."""
    d = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            d[k] = v
    return d


def run_pipeline(
    n_rounds=5,
    games_per_round=500,
    n_sims=200,
    n_train_epochs=10,
    batch_size=256,
    lr=0.002,
    device="cuda",
    checkpoint_dir=None,
    start_round=1,
    model=None,
    optimizer=None,
    max_moves=100,
    elo_n_games=40,
    elo_n_sims=100,
    workers=1,
    mcts_batch=1,
    champion=None,
    value_teacher=0.0,
    teacher_weights_path=None,
    teacher_scale=0.0,
    dist_teacher=0.0,
    dist_scale=0.0,
    endgame_n=0,
    endgame_full=8,
    endgame_n_sims=200,
    endgame_add_noise=False,
    finish_rate_n=0,
    finish_rate_sims=30,
    seed=None,
):
    """Run the full self-play -> train -> evaluate loop.

    `workers` (T-9): if > 1, the per-round self-play phase is split into
    `workers` OS processes that each play a contiguous chunk of
    `games_per_round` games and share the GPU; the returned tuples are
    concatenated in order. Data volume per round is unchanged; only the
    self-play wall-clock drops. `workers=1` (default) is the original
    sequential behavior. `seed`, when given, is offset per worker chunk so the
    workers do not replay identical games. The same `workers` count also
    parallelizes each round's end-of-round Elo eval (see `evaluate_elo`).

    `endgame_n` (T-14): when > 0, the round additionally plays `endgame_n`
    near-win **endgame** games (base at `endgame_full`/9 with an invader, to-
    move = the attacker) on top of the `games_per_round` fresh-board games.
    This is a *supplement* to the approach signal: the endgame games give the
    value head a true terminal +/-1 target (fresh-board games are all draws
    within `max_moves`). `endgame_n=0` (default) is the original behavior.
    `endgame_n_sims` (default 200) is the sim budget for those near-win games;
    `endgame_add_noise` (default False) keeps them noise-off. Both differ from
    the fresh-board `n_sims`/noise because the net's prior on the winning move
    is near-uniform: at a low sim budget with Dirichlet noise the winning
    action gets ~0 visits and the game never converts, so the true +/-1 target
    (the whole point of the curriculum) is never generated.

    `finish_rate_n`/`finish_rate_sims` (T-14): when `finish_rate_n > 0`, each
    round also measures the held-out finish-rate (fraction of fresh near-win
    positions converted to a true win) and logs it. This is the metric that
    can actually SEE the finish (full-game Elo is blind to it). `finish_rate_n
    = 0` (default) skips it.

    Returns the final model.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Create or load model
    if model is None:
        model = HalmaNet()
    model.to(device)

    # Resolve the champion (best-trained opponent) this run plays against.
    champ_model, champ_elo, champ_label = _load_champion(device, checkpoint_dir, champion)
    print(f"champion ready: {champ_label}  (elo {champ_elo:.1f}) - each round plays against it")

    # Teacher value target (bootstrap the value head out of the all-draw ~0
    # attractor). value_teacher=0 -> pure self-play (original behavior).
    teacher_weights = None
    teacher_scale = 1.0
    if value_teacher and value_teacher > 0:
        from azero.value_target import calibrate_scale, load_weights
        if teacher_weights_path is None:
            teacher_weights_path = os.path.join(os.path.dirname(__file__), "..", "best_weights.json")
        teacher_weights = load_weights(teacher_weights_path)
        teacher_scale = float(teacher_scale) if (teacher_scale and teacher_scale > 0) else calibrate_scale(teacher_weights)
        print(f"teacher value target ON: W={value_teacher}  scale={teacher_scale:.1f}  {teacher_weights_path}")

    # T-13 distance-to-enemy-base shaping target (bootstrap; fade to 0). The
    # scale is auto-calibrated from sample positions unless a positive value is
    # given; a fixed default (4.0) is used only if calibration is not run.
    dist_scale_resolved = 4.0
    if dist_teacher and dist_teacher > 0:
        from azero.value_target import calibrate_distance_scale
        dist_scale_resolved = float(dist_scale) if (dist_scale and dist_scale > 0) else calibrate_distance_scale()
        print(f"distance value target ON: W={dist_teacher}  scale={dist_scale_resolved:.1f}")

    # T-14 endgame curriculum: extra near-win games per round (supplement, not
    # replacement) + the held-out finish-rate reported each round.
    if endgame_n and endgame_n > 0:
        print(f"endgame curriculum ON: +{endgame_n} near-win games/round "
              f"(base {endgame_full}/9, to-move=attacker)")
    if finish_rate_n and finish_rate_n > 0:
        print(f"finish-rate metric ON: {finish_rate_n} held-out near-win positions/round")

    # Training log lines
    log_lines = []

    for r in range(start_round, start_round + n_rounds):
        t0 = time.time()

        # 1. Self-play
        w = max(1, workers)
        eg_note = f" + {endgame_n} endgame" if (endgame_n and endgame_n > 0) else ""
        print(f"round {r}: collecting self-play data ({games_per_round} games{eg_note}, "
              f"n_sims={n_sims}, workers={w})...")
        data = _collect_selfplay(
            model,
            n_games=games_per_round,
            device=device,
            n_sims=n_sims,
            max_moves=max_moves,
            workers=w,
            mcts_batch=mcts_batch,
            seed=seed,
            value_teacher=value_teacher,
            teacher_weights=teacher_weights,
            teacher_scale=teacher_scale,
            dist_teacher=dist_teacher,
            dist_scale=dist_scale_resolved,
            endgame_n=endgame_n,
            endgame_full=endgame_full,
            endgame_n_sims=endgame_n_sims,
            endgame_add_noise=endgame_add_noise,
        )
        print(f"round {r}: self-play done, {len(data)} samples collected")

        if len(data) == 0:
            print(f"round {r}: WARNING - no samples collected, skipping training")
            # Save checkpoint so resume works
            if optimizer is None:
                import torch.optim as optim
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            ckpt_path = os.path.join(checkpoint_dir, f"round_{r}.pt")
            save_checkpoint(model, optimizer, r, ckpt_path)
            # Still evaluate vs the champion and log
            model.eval()
            opponent = champ_label
            score = evaluate_elo(model, champ_model, n_games=elo_n_games, n_sims=elo_n_sims, device=device, max_moves=max_moves, workers=workers, mcts_batch=mcts_batch)[0]
            elo = champ_elo + K * (score - 0.5)
            new_champ = 1 if score > 0.5 else 0
            if new_champ:
                champ_model = _promote_champion(model, elo, f"round_{r}", checkpoint_dir, device)
                champ_elo, champ_label = elo, f"round_{r}"
            fr = _measure_finish_rate(model, device, finish_rate_n, endgame_full or 8,
                                     max_moves, finish_rate_sims, 20261110 + r, mcts_batch)
            elapsed = time.time() - t0
            line = (f"round={r}  samples=0  avg_loss=N/A  vs={opponent}  "
                    f"score={score:.3f}  elo={elo:.1f}  finish_rate={fr:.3f}  "
                    f"new_champ={new_champ}  time_s={elapsed:.1f}")
            print(line)
            log_lines.append(line)
            _append_log(line)
            continue

        # 2. Train
        model.train()
        optimizer, avg_loss = train(
            model,
            data,
            n_epochs=n_train_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            optimizer=optimizer,
        )
        print(f"round {r}: training done (avg_loss={avg_loss:.4f})")

        # 3. Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"round_{r}.pt")
        save_checkpoint(model, optimizer, r, ckpt_path)
        print(f"round {r}: saved checkpoint to {ckpt_path}")

        # 4. Elo eval vs the current champion (opponent = champion BEFORE this round)
        model.eval()
        opponent = champ_label
        score = evaluate_elo(
            model, champ_model, n_games=elo_n_games, n_sims=elo_n_sims, device=device, max_moves=max_moves, workers=workers, mcts_batch=mcts_batch
        )[0]
        elo = champ_elo + K * (score - 0.5)
        new_champ = 1 if score > 0.5 else 0
        if new_champ:
            champ_model = _promote_champion(model, elo, f"round_{r}", checkpoint_dir, device)
            champ_elo, champ_label = elo, f"round_{r}"
        print(f"round {r}: vs {opponent} -> score {score:.3f}, elo {elo:.1f}"
              + (", NEW CHAMPION" if new_champ else ""))

        # 4b. Held-out finish-rate (the metric that can SEE the finish).
        fr = _measure_finish_rate(model, device, finish_rate_n, endgame_full or 8,
                                  max_moves, finish_rate_sims, 20261110 + r, mcts_batch)
        if fr > 0.0:
            print(f"round {r}: finish_rate={fr:.3f} (held-out near-wins converted)")

        # 5. Log line
        elapsed = time.time() - t0
        line = (f"round={r}  samples={len(data)}  avg_loss={avg_loss:.4f}  vs={opponent}  "
                f"score={score:.3f}  elo={elo:.1f}  finish_rate={fr:.3f}  "
                f"new_champ={new_champ}  time_s={elapsed:.1f}")
        print(line)
        log_lines.append(line)
        _append_log(line)

    # Final checkpoint
    model.eval()
    final_path = os.path.join(checkpoint_dir, "final.pt")
    if optimizer is not None:
        save_checkpoint(model, optimizer, start_round + n_rounds - 1, final_path)
    else:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save({"model_state": model.state_dict()}, final_path)
    print(f"saved final checkpoint to {final_path}")

    # Summary
    print("\n" + "=" * 78)
    print(f"{'round':<8}{'samples':<10}{'vs':<16}{'score':<9}{'elo':<9}{'finish':<9}{'new':<6}")
    print("-" * 78)
    for line in log_lines:
        p = _parse_log_line(line)
        print(
            f"{p.get('round','?'):<8}"
            f"{p.get('samples','?'):<10}"
            f"{p.get('vs','?'):<16}"
            f"{p.get('score','?'):<9}"
            f"{p.get('elo','?'):<9}"
            f"{p.get('finish_rate','?'):<9}"
            f"{p.get('new_champ','?'):<6}"
        )
    print("=" * 78)

    return model


def _append_log(line):
    """Append a line to the training log file."""
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _measure_finish_rate(model, device, n_positions, full, max_moves, n_sims, seed, mcts_batch):
    """Held-out finish-rate for this round (0.0 when the metric is disabled).

    Uses a seed distinct from the training endgame positions so this set is
    genuinely held-out. This is the one metric that can SEE the finish the
    all-draw full-game Elo is blind to."""
    if not (n_positions and n_positions > 0):
        return 0.0
    from azero.finish_rate import finish_rate
    return finish_rate(
        model, n_positions=n_positions, full=full,
        max_moves=max(max_moves, 20), n_sims=n_sims, device=device,
        seed=seed, mcts_batch=mcts_batch,
    )


def _collect_selfplay(model, n_games, device, n_sims, max_moves, workers=1,
                      mcts_batch=1, seed=None, value_teacher=0.0,
                      teacher_weights=None, teacher_scale=1.0,
                      dist_teacher=0.0, dist_scale=4.0,
                      endgame_n=0, endgame_full=8,
                      endgame_n_sims=200, endgame_add_noise=False):
    """Collect `n_games` self-play tuples, optionally across `workers` processes (T-9).

    `workers == 1` is the original sequential path. `workers > 1` splits the
    games into `workers` contiguous chunks (a non-divisible remainder goes to
    the first chunks, so the total games played is exactly `n_games`), runs each
    chunk in a spawned OS process via `selfplay_worker`, and concatenates the
    results in chunk order. All workers are given the same weights (the parent
    model's state dict, copied to CPU so it pickles), so they see identical
    inputs and the merged data is equivalent to the sequential run modulo RNG.

    `mcts_batch` (T-10) is the per-search MCTS leaf-batch size, forwarded to
    every worker/sequential self-play; 1 = original behavior.

    `endgame_n` (T-14): when > 0, additionally play `endgame_n` near-win
    endgame games (base at `endgame_full`/9, to-move = the attacker) ON TOP of
    the `n_games` fresh-board games, in BOTH paths — the sequential
    `self_play_batch(start_states=...)` call and the per-worker `start_states`
    chunks — so the value head gets a true terminal +/-1 the all-draw fresh
    games never produce. `endgame_n=0` (default) is byte-identical to the
    pre-T-14 behavior.
    """
    from azero.endgame import deserialize, make_endgame_positions, serialize

    endgame_states = None
    if endgame_n and endgame_n > 0:
        endgame_states = [
            serialize(st)
            for st in make_endgame_positions(endgame_n, full=endgame_full, seed=seed)
        ]

    if workers <= 1:
        data = self_play_batch(
            model, n_games=n_games, device=device, n_sims=n_sims, max_moves=max_moves,
            mcts_batch=mcts_batch,
            value_teacher=value_teacher, teacher_weights=teacher_weights,
            teacher_scale=teacher_scale,
            dist_teacher=dist_teacher, dist_scale=dist_scale,
        )
        if endgame_states:
            # Near-win curriculum: noise off + dedicated (higher) sim budget so
            # MCTS reliably converts each game to a true +/-1 target (the
            # low-prior winning move is starved by Dirichlet noise at low sims).
            data += self_play_batch(
                model, n_games=0, device=device, n_sims=endgame_n_sims, max_moves=max_moves,
                add_noise=endgame_add_noise,
                start_states=[deserialize(d) for d in endgame_states],
                mcts_batch=mcts_batch,
                value_teacher=value_teacher, teacher_weights=teacher_weights,
                teacher_scale=teacher_scale,
                dist_teacher=dist_teacher, dist_scale=dist_scale,
            )
        return data

    # At least one worker when there is any work (fresh games or endgames).
    n_workers = min(workers, max(1, n_games))
    base = n_games // n_workers
    rem = n_games % n_workers

    # Weights travel as a plain CPU state dict (a live / CUDA module is not
    # picklable). Every worker loads the same dict, so they are identical nets.
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Contiguous chunks with a running start index, so a per-chunk seed offset
    # makes every game's seed globally distinct (no replayed games). The first
    # worker is always kept (even at 0 fresh games) so an endgames-only round
    # still has a worker to carry the serialized endgame positions.
    chunks = []
    idx = 0
    for w in range(n_workers):
        n = base + (1 if w < rem else 0)
        if n > 0 or (w == 0 and endgame_states):
            chunks.append((n, idx))
            idx += n

    # Windows uses `spawn`: the child re-imports `azero.selfplay` to resolve
    # `selfplay_worker`, so make sure the project root is on the child's
    # import path (PYTHONPATH is inherited by the fresh interpreter).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ppath = os.environ.get("PYTHONPATH", "")
    if project_root not in [p for p in ppath.split(os.pathsep) if p]:
        os.environ["PYTHONPATH"] = (ppath + os.pathsep if ppath else "") + project_root

    # T-14: split the endgame positions across the workers (contiguous chunks),
    # mirroring the fresh-game chunking. Each worker plays its endgame share on
    # top of its fresh-board chunk (see `selfplay_worker`'s `start_states`).
    n_chunks = max(1, len(chunks))
    eg = endgame_states or []
    eg_per = len(eg) // n_chunks
    eg_rem = len(eg) % n_chunks
    eg_chunks, ei = [], 0
    for c in range(len(chunks)):
        cnt = eg_per + (1 if c < eg_rem else 0)
        eg_chunks.append(eg[ei:ei + cnt])
        ei += cnt

    work = []
    for i, (n, start) in enumerate(chunks):
        work.append({
            "model_state_dict": state_dict,
            "n_games": n,
            "n_sims": n_sims,
            "max_moves": max_moves,
            "device": device,
            "seed": None if seed is None else seed + start,
            "add_noise": True,
            "mcts_batch": mcts_batch,
            "value_teacher": value_teacher,
            "teacher_weights": teacher_weights,
            "teacher_scale": teacher_scale,
            "dist_teacher": dist_teacher,
            "dist_scale": dist_scale,
            "start_states": eg_chunks[i] if eg_chunks[i] else None,
            "endgame_n_sims": endgame_n_sims,
            "endgame_add_noise": endgame_add_noise,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(selfplay_worker, work)

    data = []
    for chunk in results:
        data.extend(chunk)
    return data
