"""Time self-play throughput on CUDA so we can estimate the full-run wall-clock.

Measures seconds per (game x sim) unit, then extrapolates to candidate configs.
Self-play cost ~= games x moves x sims x forward_pass_cost; moves/game ~ constant,
so rate scales with (games x sims).
"""

import time

import torch

from azero.model import HalmaNet
from azero.selfplay import self_play_batch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(0)
    model = HalmaNet()
    model.eval()
    model.to(device)

    # Warm up (compile-ish, allocator, etc.)
    self_play_batch(model, n_games=1, device=device, n_sims=10, max_moves=10, seed=0)
    torch.cuda.synchronize() if device == "cuda" else None

    games, sims = 8, 40
    t0 = time.time()
    data = self_play_batch(model, n_games=games, device=device, n_sims=sims, max_moves=100, seed=1)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print("measured: games=%d sims=%d  samples=%d  wall=%.1fs" % (games, sims, len(data), dt))
    per_game_sim = dt / (games * sims)          # seconds per (game x sim) unit
    per_move_sim = dt / (games * 100 * sims)    # seconds per (move x sim)
    print("rate: %.4fs per (game-sim)   %.5fs per (move-sim)" % (per_game_sim, per_move_sim))

    for g, s in [(1000, 300), (1000, 200), (500, 200), (500, 100), (200, 100)]:
        secs = per_game_sim * g * s
        print("  games=%d sims=%d -> est self-play ~ %.0f min" % (g, s, secs / 60.0))


if __name__ == "__main__":
    main()
