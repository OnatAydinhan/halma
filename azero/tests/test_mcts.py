"""Tests for azero/mcts.py (T-3)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from halma8 import Halma8, P1, P2
from azero.board import ACTION_LIST, action_index, legal_action_indices
from azero.model import HalmaNet
from azero.mcts import mcts_search


def _fresh_model(seed=1234):
    torch.manual_seed(seed)
    m = HalmaNet()
    m.eval()
    return m


def _make_state(p1_cells, p2_cells, turn=P1):
    """Build a Halma8 with an explicit piece placement and no chain."""
    s = Halma8()
    s.grid = [[0] * 8 for _ in range(8)]
    for (r, c) in p1_cells:
        s.grid[r][c] = P1
    for (r, c) in p2_cells:
        s.grid[r][c] = P2
    s.turn = turn
    s.winner = None
    s.chain_active = False
    s.chain_banned = set()
    s.chain_starter = None
    s.chain_current = None
    return s


def test_search_returns_legal_action():
    model = _fresh_model()
    state = Halma8()
    best_action, root_policy = mcts_search(
        state, model, n_sims=20, add_noise=False, seed=7
    )
    assert best_action is not None
    legal = legal_action_indices(state)
    assert action_index(best_action) in legal


def test_root_policy_shape_and_sum():
    model = _fresh_model()
    state = Halma8()
    best_action, root_policy = mcts_search(
        state, model, n_sims=20, add_noise=False, seed=7
    )
    assert root_policy.shape == (417,)
    assert abs(root_policy.sum() - 1.0) < 1e-5
    assert (root_policy >= 0).all()


def test_forced_win_found():
    """P1 can win in one move: 8 of P2's 9 base cells are occupied, the last
    empty cell is (1,5), and P1 has a piece at (1,4) that can step into it."""
    model = _fresh_model()
    p2_cells = [(0, 5), (0, 6), (0, 7), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)]
    p1_cells = [(1, 4)]
    state = _make_state(p1_cells, p2_cells, turn=P1)
    assert state.winner is None

    best_action, _ = mcts_search(state, model, n_sims=50, add_noise=False, seed=7)
    assert best_action is not None
    # Applying the chosen move must produce a P1 win.
    import copy
    from azero.mcts import _apply_action
    s2 = copy.deepcopy(state)
    _apply_action(s2, best_action)
    assert s2.winner == P1


def _walk(node, acc):
    acc.append(node)
    for child in node.children.values():
        _walk(child, acc)
    return acc


def test_root_visits_match_sims():
    model = _fresh_model()
    state = Halma8()
    n_sims = 20
    best_action, root_policy, root = mcts_search(
        state, model, n_sims=n_sims, add_noise=False, seed=7, return_root=True
    )
    # Every simulation backpropagates exactly once to the root.
    assert root.visits == n_sims


def test_no_nan_inf_after_search():
    model = _fresh_model()
    state = Halma8()
    best_action, root_policy, root = mcts_search(
        state, model, n_sims=30, add_noise=True, seed=7, return_root=True
    )
    nodes = _walk(root, [])
    assert len(nodes) >= 2  # root plus at least one child
    for n in nodes:
        assert torch.isfinite(torch.tensor(n.value_sum, dtype=torch.float32))
        assert n.visits >= 0
    assert torch.isfinite(torch.tensor(root_policy, dtype=torch.float32)).all()


def test_mcts_batch_equiv():
    """T-10 gate.

    The contract is: `batch_size=1` reproduces the **original** one-leaf-per-
    iteration search **exactly**, and `batch_size>1` is a sound, deterministic
    batched variant that batches the network evals.

    NOTE: exact `best_action` equality between `batch_size=1` and `batch_size>1`
    is deliberately NOT asserted. With an *untrained* (near-constant) net the
    best move is decided by PUCT tie-breaking / visit order, which legitimately
    differs between the batched and sequential descents (the "order-sensitive"
    case flagged in the T-10 risk note). The meaningful, model-independent
    anchor is `batch_size=1` == the original algorithm, asserted exactly below
    against a reference copy of the pre-T-10 loop.
    """
    import copy
    import math
    import numpy as np
    import torch

    from azero.board import ACTION_LIST, encode_board
    from azero.mcts import (
        MCTSNode, _legal, _is_terminal, _terminal_value, _apply_action,
        _dirichlet_noise,
    )

    def reference_mcts(state, model, n_sims, add_noise, seed,
                       c_puct=1.5, da=0.3, de=0.25, device="cpu"):
        """Verbatim copy of the original (pre-T-10) sequential MCTS."""
        dev = torch.device(device)
        rng = np.random.default_rng(seed)
        model.eval()
        root = MCTSNode(copy.deepcopy(state))

        def net(s):
            x = encode_board(s).to(dev)
            logits, value = model(x)
            return logits[0], value[0].item()

        def compute_priors(logits, legal, noisy):
            mask = torch.full((logits.shape[-1],), float("-inf"), device=logits.device)
            for i in legal:
                mask[i] = 0.0
            probs = torch.softmax(logits + mask, dim=-1)
            priors = {i: float(probs[i].item()) for i in legal}
            if noisy:
                _dirichlet_noise(priors, da, de, rng)
            return priors

        def expand(node, noisy):
            node.expanded = True
            legal = _legal(node.state)
            if not legal:
                node.leaf_value = -1.0
                return -1.0
            logits, value = net(node.state)
            node.priors = compute_priors(logits, legal, noisy)
            node.leaf_value = value
            for i in legal:
                action = ACTION_LIST[i]
                cs = copy.deepcopy(node.state)
                _apply_action(cs, action)
                node.children[i] = MCTSNode(cs, parent=node, action=action,
                                            prior=node.priors[i])
            return value

        def select(node):
            best = None
            best_score = float("-inf")
            sqrt_n = math.sqrt(node.visits)
            for child in node.children.values():
                q = child.value_sum / max(child.visits, 1)
                u = c_puct * child.prior * sqrt_n / (1 + child.visits)
                if q + u > best_score:
                    best_score = q + u
                    best = child
            return best

        def backprop(node, v, leaf_turn):
            while node is not None:
                sign = 1.0 if node.state.turn == leaf_turn else -1.0
                node.visits += 1
                node.value_sum += sign * v
                node = node.parent

        with torch.no_grad():
            for _ in range(n_sims):
                node = root
                while node.expanded:
                    node = select(node)
                if _is_terminal(node.state):
                    v = _terminal_value(node.state)
                else:
                    v = expand(node, noisy=(node is root and add_noise))
                backprop(node, v, leaf_turn=node.state.turn)

        root_policy = np.zeros(len(ACTION_LIST), dtype=np.float32)
        total = sum(c.visits for c in root.children.values())
        if total > 0:
            for i, child in root.children.items():
                root_policy[i] = child.visits / total
            s = root_policy.sum()
            if s > 0:
                root_policy /= s
        best_action = None
        if root.children:
            best_i = max(root.children, key=lambda i: root.children[i].visits)
            best_action = ACTION_LIST[best_i]
        return best_action, root_policy, root

    model = _fresh_model()
    positions = [
        Halma8(),
        _make_state([(0, 0), (1, 1), (2, 2)], [(7, 7), (6, 6), (5, 5)], turn=P1),
    ]

    for state in positions:
        for add_noise, seed in [(False, 7), (True, 7), (False, 3)]:
            # -- batch_size=1 must EXACTLY reproduce the original algorithm --
            ra, rp, rroot = reference_mcts(state, model, 60, add_noise, seed)
            a1, p1, _ = mcts_search(state, model, n_sims=60, add_noise=add_noise,
                                    seed=seed, batch_size=1, return_root=True)
            assert a1 == ra, f"batch_size=1 best_action {a1} != original {ra}"
            assert np.array_equal(p1, rp), (
                f"batch_size=1 root_policy diverges from original; "
                f"max|diff|={float(np.max(np.abs(p1 - rp)))}"
            )
            # -- batch_size>1: sound, deterministic, legal --
            a16, p16 = mcts_search(state, model, n_sims=60, add_noise=add_noise,
                                   seed=seed, batch_size=16)
            a16b, p16b = mcts_search(state, model, n_sims=60, add_noise=add_noise,
                                    seed=seed, batch_size=16)
            assert a16 == a16b and np.array_equal(p16, p16b), "batch_size=16 not deterministic"
            assert a16 is not None
            legal = set(legal_action_indices(state))
            assert action_index(a16) in legal, f"batch_size=16 picked illegal {a16}"
            assert p16.shape == (417,) and abs(p16.sum() - 1.0) < 1e-5
            assert np.isfinite(p16).all() and (p16 >= 0).all()


def test_mcts_halma_clone_equiv():
    """T-11 gate: `mcts_search` (now using `halma_clone`) returns the same
    `best_action` as a `copy.deepcopy`-based reference for the same seed
    and positions. Proves the deepcopy → halma_clone swap is transparent.
    """
    import copy
    import math
    import numpy as np

    from azero.board import ACTION_LIST, encode_board
    from azero.mcts import (
        MCTSNode, _legal, _is_terminal, _terminal_value, _apply_action,
        _dirichlet_noise,
    )

    def ref_mcts_deepcopy(state, model, n_sims, add_noise, seed,
                          c_puct=1.5, da=0.3, de=0.25, device="cpu"):
        dev = torch.device(device)
        rng = np.random.default_rng(seed)
        model.eval()
        root = MCTSNode(copy.deepcopy(state))

        def net(s):
            x = encode_board(s).to(dev)
            logits, value = model(x)
            return logits[0], value[0].item()

        def compute_priors(logits, legal, noisy):
            mask = torch.full((logits.shape[-1],), float("-inf"), device=logits.device)
            for i in legal:
                mask[i] = 0.0
            probs = torch.softmax(logits + mask, dim=-1)
            priors = {i: float(probs[i].item()) for i in legal}
            if noisy:
                _dirichlet_noise(priors, da, de, rng)
            return priors

        def expand(node, noisy):
            node.expanded = True
            legal = _legal(node.state)
            if not legal:
                node.leaf_value = -1.0
                return -1.0
            logits, value = net(node.state)
            node.priors = compute_priors(logits, legal, noisy)
            node.leaf_value = value
            for i in legal:
                action = ACTION_LIST[i]
                cs = copy.deepcopy(node.state)
                _apply_action(cs, action)
                node.children[i] = MCTSNode(cs, parent=node, action=action,
                                            prior=node.priors[i])
            return value

        def select(node):
            best = None
            best_score = float("-inf")
            sqrt_n = math.sqrt(node.visits)
            for child in node.children.values():
                q = child.value_sum / max(child.visits, 1)
                u = c_puct * child.prior * sqrt_n / (1 + child.visits)
                if q + u > best_score:
                    best_score = q + u
                    best = child
            return best

        def backprop(node, v, leaf_turn):
            while node is not None:
                sign = 1.0 if node.state.turn == leaf_turn else -1.0
                node.visits += 1
                node.value_sum += sign * v
                node = node.parent

        with torch.no_grad():
            for _ in range(n_sims):
                node = root
                while node.expanded:
                    node = select(node)
                if _is_terminal(node.state):
                    v = _terminal_value(node.state)
                else:
                    v = expand(node, noisy=(node is root and add_noise))
                backprop(node, v, leaf_turn=node.state.turn)

        if not root.children:
            return None
        best_i = max(root.children, key=lambda i: root.children[i].visits)
        return ACTION_LIST[best_i]

    model = _fresh_model()
    positions = [
        Halma8(),
        _make_state([(0, 0), (1, 1), (2, 2)], [(7, 7), (6, 6), (5, 5)], turn=P1),
    ]
    for state in positions:
        for add_noise, seed in [(False, 7), (True, 7), (False, 3)]:
            ref = ref_mcts_deepcopy(state, model, 40, add_noise, seed)
            got, _ = mcts_search(state, model, n_sims=40, add_noise=add_noise,
                                 seed=seed, batch_size=1)
            assert got == ref, (
                f"halma_clone diverged: got={got!r} ref={ref!r} "
                f"(add_noise={add_noise}, seed={seed})"
            )
