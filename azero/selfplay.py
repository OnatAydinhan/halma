"""Self-play for the AlphaZero NN (Part 3, T-4).

Plays full games using MCTS + the network, collecting
(position, policy, value) training tuples. Rules and legality are always
delegated to `Halma8` / `azero.board`.

Value convention is negamax: a tuple's value is +1.0 if the player to move
at that position wins, -1.0 if they lose. For games that end in a winner the
true outcome is used for every position. For games that never produce a
winner (max_moves or stuck) the dense MCTS root-value estimate is used, so
non-winning games still contribute training data (a pure outcome target would
discard ~all weak-model games, since the "attacker piece in a full enemy
base" win is rare under weak play).
"""

import torch

from halma8 import Halma8

from azero.board import action_index, decode_action, encode_board, halma_clone
from azero.model import HalmaNet
from azero.mcts import mcts_search
from azero.value_target import distance_advantage, normalized_value


def play_one_game(
    model,
    device="cpu",
    n_sims=200,
    max_moves=100,
    add_noise=True,
    seed=None,
    start_state=None,
    mcts_batch=1,
    value_teacher=0.0,
    teacher_weights=None,
    teacher_scale=1.0,
    dist_teacher=0.0,
    dist_scale=4.0,
):
    """Play one full game from the start position.

    Returns a list of training tuples:
      (encoded_state: (7,8,8) tensor,
       policy: (417,) numpy array,
       value: float in [-1.0, +1.0])

    Value target:
      - If the game ends in a winner, every position's value is the true
        outcome for the mover: +1.0 if the mover won, -1.0 if they lost.
      - If the game never produces a winner (max_moves or stuck), each
        position's value is the dense MCTS root-value estimate (negamax) for
        the mover at that position. This keeps non-winning games as training
        data so a weak model still gets a value gradient.

    `start_state` (optional) may be a `Halma8` to start from; defaults to a
    fresh start position. Used by tests to reach a forced-win position.

    Value-target bootstrapping (both default 0 -> pure self-play target):
      - `value_teacher` W in [0,1]: weight on the 6-term eval value (T-12).
      - `dist_teacher`  W in [0,1]: weight on the T-13 distance-to-enemy-base
        advancement advantage (focused "getting closer to the win" signal).
    The final target is a convex blend of the enabled shaping terms with the
    base MCTS/outcome value; the residual weight (1 - sum) stays on the base.
    """
    state = Halma8() if start_state is None else start_state
    # (enc, policy, mover, dense_value) for every position reached this game.
    records = []

    for _ in range(max_moves):
        enc = encode_board(state).squeeze(0)          # (7, 8, 8)
        mover = state.turn

        best_action, root_policy, root = mcts_search(
            state,
            model,
            n_sims=n_sims,
            device=device,
            add_noise=add_noise,
            seed=seed,
            return_root=True,
            batch_size=mcts_batch,
        )
        if best_action is None:
            # No legal move at the root (stuck): stop this game. Positions
            # already collected are kept with dense value targets below.
            break

        dense_value = root.value_sum / max(root.visits, 1)
        # Snapshot the grid (mutated in place later) for the teacher value target.
        grid_snap = [row[:] for row in state.grid]
        records.append((enc, root_policy, mover, dense_value, grid_snap))

        if best_action[0] == "stop":
            state.end_turn()
        else:
            state.apply_move(decode_action(action_index(best_action)))

        if state.winner is not None:
            break

    winner = state.winner
    out = []
    for enc, policy, mover, dense_value, grid in records:
        # Base target: true outcome if the game was decided, else the dense
        # MCTS root value (the original behavior).
        base = (1.0 if mover == winner else -1.0) if winner is not None else dense_value
        # Teacher bootstrap: convex blend of up to two shaping targets (the
        # 6-term eval and the T-13 distance-to-enemy-base term) with the base
        # MCTS/outcome value. Both weights default 0 -> `value == base` exactly
        # (byte-identical to the pre-teacher behavior). Fade both to 0 for pure
        # self-play once the net wins on its own.
        w_eval = value_teacher if (value_teacher and value_teacher > 0 and teacher_weights is not None) else 0.0
        w_dist = dist_teacher if (dist_teacher and dist_teacher > 0) else 0.0
        if w_eval == 0.0 and w_dist == 0.0:
            value = base
        else:
            eval_target = normalized_value(grid, mover, teacher_weights, teacher_scale) if w_eval > 0.0 else base
            dist_target = distance_advantage(grid, mover, dist_scale) if w_dist > 0.0 else base
            s = w_eval + w_dist
            w_base = max(0.0, 1.0 - s)
            value = (w_eval * eval_target + w_dist * dist_target + w_base * base) / (s + w_base)
        out.append((enc, policy, value))
    return out


def self_play_batch(
    model,
    n_games=500,
    device="cpu",
    n_sims=200,
    max_moves=100,
    add_noise=True,
    seed=None,
    mcts_batch=1,
    value_teacher=0.0,
    teacher_weights=None,
    teacher_scale=1.0,
    dist_teacher=0.0,
    dist_scale=4.0,
    start_states=None,
):
    """Play games sequentially, return a flat list of all tuples.

    `start_states` (T-14): when a list of `Halma8` states is given, play
    **exactly `len(start_states)` games**, game g from `start_states[g]`
    (forwarded to the existing `play_one_game(start_state=...)`). This is the
    endgame-curriculum path (near-win starts -> true +/-1 value target).
    `start_states=None` (default) is byte-identical to the original behavior:
    play `n_games` fresh-board games.

    `mcts_batch` (T-10) is the per-search MCTS leaf-batch size forwarded to
    `mcts_search`; 1 (default) is the original sequential behavior.
    `value_teacher`/`teacher_weights`/`teacher_scale` (T-12 eval) and
    `dist_teacher`/`dist_scale` (T-13 distance) forward to `play_one_game`
    (all 0.0 = original pure self-play target).
    """
    if start_states is not None:
        games = len(start_states)
        starts = [halma_clone(st) for st in start_states]
    else:
        games = n_games
        starts = [None] * games

    tuples = []
    for g in range(games):
        s = None if seed is None else seed + g
        tuples.extend(
            play_one_game(
                model,
                device=device,
                n_sims=n_sims,
                max_moves=max_moves,
                add_noise=add_noise,
                seed=s,
                mcts_batch=mcts_batch,
                value_teacher=value_teacher,
                teacher_weights=teacher_weights,
                teacher_scale=teacher_scale,
                dist_teacher=dist_teacher,
                dist_scale=dist_scale,
                start_state=starts[g],
            )
        )
    return tuples


def selfplay_worker(args):
    """OS-process self-play worker (T-9) — run by a spawned child process.

    Must be a **module-level** function so it can be pickled by reference under
    Windows' `spawn` start method (a closure/lambda cannot). The live `torch`
    module is not picklable, so the weights travel as a plain state dict in
    `args`; this worker rebuilds the network, plays its chunk of games, and
    returns the flat tuple list.

    `args` is a picklable dict:
      model_state_dict  : HalmaNet state dict (CPU tensors)
      n_games           : int, fresh-board games this worker must play
      n_sims            : int, MCTS sims per move
      max_moves         : int, max moves per game
      device            : str, "cpu" or "cuda"
      seed              : int | None, base seed (self_play_batch adds the
                          per-game offset); use a distinct base per worker so
                          chunks do not replay identical games
      add_noise         : bool, Dirichlet noise at the root
      mcts_batch        : int, MCTS leaf-batch size per search (T-10); 1 = sequential
      start_states      : list[dict] | None (T-14) — serialized near-win endgame
                          states (`azero.endgame.serialize`) to play IN ADDITION
                          to `n_games`; the worker deserializes them here.

    Returns a flat list of (enc:(7,8,8) tensor, policy:(417,) ndarray, value:
    float) tuples, all moved to CPU so they pickle cleanly back to the parent.
    """
    state_dict = args["model_state_dict"]
    model = HalmaNet()
    model.load_state_dict(state_dict)
    model.eval()

    device = args.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    common = dict(
        device=device,
        n_sims=args["n_sims"],
        max_moves=args["max_moves"],
        add_noise=args.get("add_noise", True),
        mcts_batch=args.get("mcts_batch", 1),
        value_teacher=args.get("value_teacher", 0.0),
        teacher_weights=args.get("teacher_weights"),
        teacher_scale=args.get("teacher_scale", 1.0),
        dist_teacher=args.get("dist_teacher", 0.0),
        dist_scale=args.get("dist_scale", 4.0),
    )
    seed = args.get("seed")

    tuples = self_play_batch(model, n_games=args.get("n_games", 0), seed=seed, **common)

    # T-14 endgame chunk: serialized dicts cross the spawn boundary; rebuild
    # them here and play them on top of the fresh-board games.
    e_states = args.get("start_states")
    if e_states:
        from azero.endgame import deserialize
        e_seed = None if seed is None else seed + 1000003
        # Near-win curriculum games are played noise-OFF at a dedicated (higher)
        # sim budget so MCTS reliably converts each to a true +/-1 target. The
        # net's prior on the winning move is near-uniform (it does not "know"
        # the win), so at a low sim budget Dirichlet noise starves that action
        # to ~0 visits and the game never converts -> the chicken-and-egg
        # deadlock. Fresh-board games above keep noise on (diversity).
        eg = dict(common)
        eg["n_sims"] = args.get("endgame_n_sims", common["n_sims"])
        eg["add_noise"] = args.get("endgame_add_noise", False)
        tuples += self_play_batch(
            model, n_games=0, seed=e_seed,
            start_states=[deserialize(d) for d in e_states], **eg)

    # Detach + CPU: the parent may be running a CUDA model, and a CUDA tensor
    # is not picklable, so every returned tensor/array is copied to CPU.
    return [(enc.detach().cpu(), policy, float(value)) for enc, policy, value in tuples]
