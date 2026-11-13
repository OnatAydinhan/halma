"""T-10 throughput probe (report-only, not a gate).

Compares the self-play phase at mcts_batch=1 (original) vs mcts_batch=32
(batched leaf evals): forward-call count and wall-clock per (move x sim) on
CUDA. Expect: far fewer forward calls at batch=32, and a wall-clock reduction
in the (Python-MCTS-bound) self-play phase where the batch finally loads the GPU.
"""

import time

import torch

from azero.model import HalmaNet
from azero.selfplay import self_play_batch


class CountingNet(HalmaNet):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.rows = 0  # total tensor rows (sum of batch dims) forwarded

    def forward(self, x):
        self.calls += 1
        self.rows += x.shape[0]
        return super().forward(x)


def run(model, batch, games=6, sims=30, max_moves=40, seed=1):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    data = self_play_batch(
        model, n_games=games, device=DEVICE, n_sims=sims,
        max_moves=max_moves, seed=seed, mcts_batch=batch,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.time() - t0
    per_move_sim = dt / (games * max_moves * sims)
    print(f"batch={batch:<3} samples={len(data):<5} wall={dt:6.2f}s  "
          f"fwd_calls={model.calls:<7} fwd_rows={model.rows:<8} "
          f"rows/call={model.rows/max(model.calls,1):.1f}  "
          f"per(move-sim)={per_move_sim*1e3:.3f}ms")
    return dt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("device:", DEVICE)
    for batch in (1, 32):
        torch.manual_seed(0)
        model = CountingNet()
        model.eval()
        model.to(DEVICE)
        run(model, batch, games=1, sims=10, max_moves=10, seed=0)  # warmup
        torch.manual_seed(0)
        model = CountingNet()
        model.eval()
        model.to(DEVICE)
        run(model, batch)
    print("expect: batch=32 -> fewer fwd_calls, similar/lower per(move-sim)")


if __name__ == "__main__":
    main()
