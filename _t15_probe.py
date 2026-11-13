import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch

from azero.model import HalmaNet
from azero.endgame import make_endgame_positions, immediate_win_exists
from azero.board import (
    ACTION_LIST,
    action_index,
    decode_action,
    encode_board,
    halma_clone,
    legal_action_indices,
)
from azero.mcts import mcts_search
from azero.selfplay import play_one_game
from azero.finish_rate import finish_rate


def load(path, device):
    m = HalmaNet()
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval().to(device)
    return m


def find_win_action(st):
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
print("device:", device)

states = make_endgame_positions(10, full=8, attacker_min=1, seed=7)
imm = sum(1 for s in states if immediate_win_exists(s))
print("gen: immediate-win positions:", imm, "/", len(states))

st = states[0]
wa = find_win_action(st)
print("sample: turn=%s legal=%d win_action=%s" % (st.turn, len(legal_action_indices(st)), wa))

for ck in ("azero/checkpoints/round_5.pt", "azero/checkpoints/round_6.pt"):
    m = load(os.path.join(os.path.dirname(__file__), ck), device)
    logits, values = m(encode_board(st).to(device))
    mask = torch.full((logits.shape[-1],), float("-inf"), device=logits.device)
    for i in legal_action_indices(st):
        mask[i] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)[0]
    p_win = float(probs[action_index(wa)]) if wa is not None else -1.0
    best, _ = mcts_search(halma_clone(st), m, n_sims=30, device=device,
                          add_noise=False, seed=1, batch_size=32)
    print("%s  net_value=%.3f  policy_p(win)=%.4f  mcts_best=%s  converts=%s"
          % (os.path.basename(ck), values.item(), p_win, best, best == wa))

m5 = load(os.path.join(os.path.dirname(__file__), "azero/checkpoints/round_5.pt"), device)
tup = play_one_game(m5, device=device, n_sims=100, max_moves=6, add_noise=True,
                    seed=5, start_state=halma_clone(states[1]), mcts_batch=32)
print("play_one_game(round_5, max_moves=6): tuples=%d values=%s"
      % (len(tup), [round(v, 3) for _, _, v in tup]))

for ck in ("azero/checkpoints/round_5.pt", "azero/checkpoints/round_6.pt"):
    m = load(os.path.join(os.path.dirname(__file__), ck), device)
    fr = finish_rate(m, n_positions=4, full=8, max_moves=6, n_sims=30,
                     device=device, seed=999, mcts_batch=32)
    print("%s  finish_rate(4 positions, max_moves=6) = %.3f" % (os.path.basename(ck), fr))
