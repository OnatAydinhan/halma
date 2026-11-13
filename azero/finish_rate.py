"""Held-out finish-rate metric for the AlphaZero NN (Part 3, T-14).

The full-game Elo is *blind* to the finish: within `max_moves` a weak net's
fresh-board games are all draws, so Elo reads ~1000 whether or not the net
knows how to convert a near-win. This metric measures exactly that skill:
from a FRESH, held-out set of near-win positions (seed different from the
training endgame positions), play each with the net (add_noise=False) and
return the fraction it converts to a **true win** within `max_moves`.

A rising `finish_rate` (off 0.00) is the leading indicator the value head has
learned to finish -- it will precede any full-game Elo movement.
"""

from azero.board import action_index, decode_action
from azero.mcts import mcts_search


def _converts(model, state, device, n_sims, max_moves, seed, mcts_batch):
    """Play from `state` (no root noise); True iff the mover (state.turn, the
    attacker) ends as the winner within `max_moves`. Legality/application stay
    with the Halma8 oracle."""
    attacker = state.turn
    for _ in range(max_moves):
        best_action, _policy = mcts_search(
            state, model, n_sims=n_sims, device=device,
            add_noise=False, seed=seed, batch_size=mcts_batch,
        )
        if best_action is None:
            break
        if best_action[0] == "stop":
            state.end_turn()
        else:
            state.apply_move(decode_action(action_index(best_action)))
        if state.winner is not None:
            break
    return state.winner == attacker


def finish_rate(model, n_positions=120, full=8, max_moves=40, n_sims=50,
                device="cpu", seed=None, mcts_batch=1):
    """Fraction of fresh near-win positions the net converts to a true win.

    `seed` (default a fixed held-out base) must differ from the training
    endgame seed so this set is genuinely held-out. Returns a float in [0, 1].
    """
    from azero.endgame import make_endgame_positions
    from azero.board import halma_clone

    base_seed = seed if seed is not None else 20261110
    states = make_endgame_positions(n_positions, full=full, seed=base_seed)
    wins = 0
    for i, st in enumerate(states):
        s = None if seed is None else base_seed + 1000000 + i
        if _converts(model, halma_clone(st), device, n_sims, max_moves, s, mcts_batch):
            wins += 1
    return wins / max(1, len(states))
