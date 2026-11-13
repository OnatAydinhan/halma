import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch

from azero.model import HalmaNet
from azero.endgame import make_endgame_positions
from azero.board import action_index, halma_clone, legal_action_indices
from azero.mcts import mcts_search
from azero.selfplay import play_one_game


def load(path, device):
    m = HalmaNet()
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval().to(device)
    return m


def find_win_action(st):
    from azero.board import ACTION_LIST, decode_action
    for idx in legal_action_indices(st):
        a = ACTION_LIST[idx]
        tr = halma_clone(st)
        if a[0] == "stop":
            tr.end_turn()
        else:
            tr.apply_move(decode_action(idx))
        if tr.winner is not None and tr.winner == st.turn:
            return a
    return None


device = "cuda" if torch.cuda.is_available() else "cpu"
m = load(os.path.join(os.path.dirname(__file__), "azero/checkpoints/round_5.pt"), device)
states = make_endgame_positions(6, full=8, attacker_min=1, seed=7)

for batch in (1, 32):
    conv = 0
    detail = []
    for i, st in enumerate(states):
        wa = find_win_action(st)
        best, _ = mcts_search(halma_clone(st), m, n_sims=30, device=device,
                              add_noise=False, seed=1, batch_size=batch)
        ok = best == wa
        conv += 1 if ok else 0
        detail.append((i, ok))
    print("batch=%d  converts=%d/%d  %s" % (batch, conv, len(states), detail))

# Root vote anatomy for one position, batch=32
st = states[0]
wa = find_win_action(st)
root = mcts_search(halma_clone(st), m, n_sims=30, device=device,
                   add_noise=False, seed=1, batch_size=32, return_root=True)[2]
from azero.board import ACTION_LIST
kids = sorted(root.children.items(), key=lambda kv: -kv[1].visits)[:6]
widx = action_index(wa)
wk = root.children.get(widx)
print("batch=32 root vote: top kids (action, visits, Q) %s"
      % [(ACTION_LIST[i], k.visits, round(k.value_sum / max(k.visits, 1), 2)) for i, k in kids])
print("win action kid: visits=%s q=%.2f" % (wk.visits, wk.value_sum / max(wk.visits, 1)))

# play_one_game sequential (batch=1)
tup = play_one_game(m, device=device, n_sims=100, max_moves=6, add_noise=True,
                    seed=5, start_state=halma_clone(states[1]), mcts_batch=1)
print("play_one_game(batch=1): tuples=%d values=%s" % (len(tup), [round(v, 3) for _, _, v in tup]))
