import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from azero.endgame import make_endgame_positions, immediate_win_exists, serialize, deserialize
from azero.board import legal_action_indices
from halma8 import P1, P2

states = make_endgame_positions(20, full=8, attacker_min=1, seed=1)
print("n states:", len(states))
bad = 0
immediate = 0
for i, s in enumerate(states):
    ok = s.winner is None and len(legal_action_indices(s)) >= 1
    if not ok:
        bad += 1
    if immediate_win_exists(s):
        immediate += 1
print("bad positions:", bad)
print("immediate win exists:", immediate, "/", len(states))
# serialize round-trip
d = serialize(states[0])
s2 = deserialize(d)
print("round-trip grid equal:", s2.grid == states[0].grid)
print("round-trip turn equal:", s2.turn == states[0].turn)
# show one
print("sample turn:", states[0].turn, "winner:", states[0].winner)
