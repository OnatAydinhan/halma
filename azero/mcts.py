"""AlphaZero-style MCTS for check_halma (Part 3, T-3).

Searches a position with the neural network's policy (priors) and value
head (leaf value), no rollouts. Value convention is negamax: +1 = the
player to move wins, -1 = the player to move loses. Chain continuations
(apply_move returns True) keep the same turn, which is handled naturally by
the sign-flip in backpropagation.

Rules and legality are always delegated to `azero.board` / `Halma8`.
"""

import math

import numpy as np
import torch

from azero.board import (
    ACTION_LIST,
    decode_action,
    encode_board,
    halma_clone,
    legal_action_indices,
)


class MCTSNode:
    """A node in the search tree."""

    __slots__ = (
        "state", "parent", "action", "children", "visits", "value_sum",
        "prior", "expanded", "leaf_value", "priors",
    )

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state          # deep-copied Halma8 state at this node
        self.parent = parent
        self.action = action        # ACTION_LIST entry that led here (None at root)
        self.children = {}          # action_index -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior          # prior set at creation from parent's policy
        self.expanded = False
        self.leaf_value = 0.0       # value (net or terminal) used when this node was a leaf
        self.priors = {}            # action_index -> prior, populated at expansion


_INDEX = {a: i for i, a in enumerate(ACTION_LIST)}


def _apply_action(state, action):
    """Apply an ACTION_LIST entry to `state`, in place."""
    if action[0] == "stop":
        state.end_turn()
        return
    state.apply_move(decode_action(_INDEX[action]))


def _legal(state):
    return legal_action_indices(state)


def _is_terminal(state):
    if state.winner is not None:
        return True
    return len(_legal(state)) == 0


def _terminal_value(state):
    # Terminal node, from the perspective of the player to move here.
    if state.winner is not None:
        return 1.0 if state.winner == state.turn else -1.0
    # Stuck: the player to move has no move and loses.
    return -1.0


def _dirichlet_noise(priors, alpha, eps, rng):
    """Blend `priors` (dict over a set of keys) with Dirichlet noise."""
    keys = list(priors.keys())
    if not keys:
        return
    noise = rng.dirichlet([alpha] * len(keys))
    for k, i in enumerate(keys):
        priors[i] = (1.0 - eps) * priors[i] + eps * float(noise[k])


def mcts_search(
    state,
    model,
    n_sims=200,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    device="cpu",
    add_noise=True,
    seed=None,
    return_root=False,
    batch_size=1,
):
    """Run MCTS from `state`.

    `batch_size` (T-10): the number of leaves evaluated per outer iteration.
    The B leaves' network evaluations are stacked into a single `model(batch)`
    forward pass, amortizing per-call launch overhead. `batch_size=1` (default)
    reproduces the original one-leaf-per-iteration search **exactly** (identical
    RNG draws, node visits, and best_action); it is the safe default and the
    gate anchor. `batch_size>1` batches evals (see T-10 in plan_nn.md); it may
    yield a marginally different root visit distribution than `batch_size=1`
    because the exploration order changes, but returns the same best_action.

    Returns:
      best_action: the ACTION_LIST entry with the most visits at the root
      root_policy: np.ndarray of shape (417,), normalized visit distribution
                   over all actions (0.0 for illegal ones).
      (If `return_root` is True, also returns the root `MCTSNode`.)
    """
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    model.eval()

    root = MCTSNode(halma_clone(state))
    batch_size = max(1, int(batch_size))

    def compute_priors(logits, legal, noisy):
        # Masked softmax over legal actions only.
        mask = torch.full(
            (logits.shape[-1],), float("-inf"), device=logits.device
        )
        for i in legal:
            mask[i] = 0.0
        probs = torch.softmax(logits + mask, dim=-1)
        priors = {i: float(probs[i].item()) for i in legal}
        if noisy:
            _dirichlet_noise(priors, dirichlet_alpha, dirichlet_eps, rng)
        return priors

    def expand_with(node, logits, value, noisy):
        # Expand `node` using precomputed network output (logits: (417,)
        # tensor, value: float). `batch_size=1` feeds this the exact same
        # (logits[0], value[0].item()) the old `net()`/`expand()` produced.
        node.expanded = True
        legal = _legal(node.state)
        if not legal:
            node.leaf_value = -1.0
            return -1.0
        node.priors = compute_priors(logits, legal, noisy)
        node.leaf_value = value
        for i in legal:
            action = ACTION_LIST[i]
            child_state = halma_clone(node.state)
            _apply_action(child_state, action)
            node.children[i] = MCTSNode(
                child_state, parent=node, action=action, prior=node.priors[i]
            )
        return value

    def select(node):
        best = None
        best_score = float("-inf")
        sqrt_n = math.sqrt(node.visits)
        for child in node.children.values():
            q = child.value_sum / max(child.visits, 1)
            u = c_puct * child.prior * sqrt_n / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        return best

    def backprop(node, v, leaf_turn):
        while node is not None:
            sign = 1.0 if node.state.turn == leaf_turn else -1.0
            node.visits += 1
            node.value_sum += sign * v
            node = node.parent

    with torch.no_grad():
        done = 0
        while done < n_sims:
            # B leaves this iteration; last one takes the remainder so the
            # total number of leaf evaluations is exactly n_sims.
            B = min(batch_size, n_sims - done)
            done += B

            # 1. B independent root-to-leaf descents. Each reached leaf gets a
            #    *virtual loss* (visits += 1, value_sum -= 1) so the later
            #    descents in the same batch see it as already expanded and are
            #    steered elsewhere — this is the canonical batched-MCTS trick
            #    (Leela Zero / KataGo) that keeps batched search close to the
            #    sequential one. With batch_size=1 there are no siblings in the
            #    batch, so the virtual loss is applied and then removed around a
            #    single backprop and cancels out: the search is bit-identical
            #    to the original one-leaf-per-iteration algorithm.
            leaves = []
            for _ in range(B):
                node = root
                while node.expanded:
                    node = select(node)
                node.visits += 1
                node.value_sum -= 1
                leaves.append(node)

            # 2. Distinct leaves to expand / evaluate (descents can still
            #    collide on the same node).
            distinct = {}
            for leaf in leaves:
                distinct[id(leaf)] = leaf

            # 3. ONE batched forward pass for the distinct non-terminal leaves.
            need_net = [leaf for leaf in distinct.values() if not _is_terminal(leaf.state)]
            net_out = {}
            if need_net:
                # encode_board returns (1,7,8,8); concat along dim 0 → (B,7,8,8).
                x = torch.cat([encode_board(leaf.state) for leaf in need_net]).to(dev)
                logits, values = model(x)
                for j, leaf in enumerate(need_net):
                    net_out[id(leaf)] = (logits[j], values[j].item())

            # 4. Expand each distinct leaf once, then backprop once per (possibly
            #    duplicated) leaf so the total number of backprops — and hence
            #    root visits — is exactly n_sims. The virtual loss is removed
            #    after each backprop so it only influenced the descents.
            expanded = set()
            for leaf in leaves:
                k = id(leaf)
                if _is_terminal(leaf.state):
                    v = _terminal_value(leaf.state)
                else:
                    if k not in expanded:
                        lg, vv = net_out[k]
                        expand_with(leaf, lg, vv, noisy=(leaf is root and add_noise))
                        expanded.add(k)
                    v = leaf.leaf_value
                backprop(leaf, v, leaf.state.turn)
                leaf.visits -= 1
                leaf.value_sum += 1

    # Root policy: normalized visit distribution over all actions.
    root_policy = np.zeros(len(ACTION_LIST), dtype=np.float32)
    total = sum(c.visits for c in root.children.values())
    if total > 0:
        for i, child in root.children.items():
            root_policy[i] = child.visits / total
        s = root_policy.sum()
        if s > 0:
            root_policy /= s

    if root.children:
        best_i = max(root.children, key=lambda i: root.children[i].visits)
        best_action = ACTION_LIST[best_i]
    else:
        best_action = None
    if return_root:
        return best_action, root_policy, root
    return best_action, root_policy
