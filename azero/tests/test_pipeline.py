"""Smoke tests for the full pipeline (T-6)."""

import os

import numpy as np
import pytest
import torch

from azero.elo import K, evaluate_elo, score_to_elo
from azero.model import HalmaNet
from azero.pipeline import BASELINE_PATH, LOG_PATH, run_pipeline
from azero.train import load_checkpoint


class TestEloFormula:
    """The Elo formula must be exact (no floating-point surprises)."""

    def test_mid(self):
        assert score_to_elo(0.5) == 1000.0

    def test_win(self):
        assert score_to_elo(1.0) == 1032.0

    def test_loss(self):
        assert score_to_elo(0.0) == 968.0

    def test_K(self):
        assert K == 64


class TestPipelineSmoke:
    """Run a minimal 1-round pipeline and verify the artifacts."""

    def test_runs_and_produces_artifacts(self):
        model = HalmaNet()
        run_pipeline(
            n_rounds=1,
            games_per_round=2,
            n_sims=2,
            n_train_epochs=1,
            batch_size=4,
            device="cpu",
            model=model,
            max_moves=8,
            elo_n_games=2,
            elo_n_sims=2,
        )

        # A round checkpoint must exist
        ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "round_1.pt")
        assert os.path.isfile(ckpt_path), f"Expected checkpoint at {ckpt_path}"

        # The training log must exist and contain at least one line with elo= and score=
        assert os.path.isfile(LOG_PATH), f"Expected log at {LOG_PATH}"
        with open(LOG_PATH) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) >= 1, "Training log is empty"
        last = lines[-1]
        assert "elo=" in last, f"Log line missing 'elo=': {last}"
        assert "score=" in last, f"Log line missing 'score=': {last}"

        # Checkpoint loads and weights match the trained model
        model2 = HalmaNet()
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        load_checkpoint(model2, opt2, ckpt_path, "cpu")
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu(), atol=1e-7)


class TestEloSelfMatch:
    """evaluate_elo on a self-match should return a bounded score and finite elo."""

    def test_self_match_bounded(self):
        model = HalmaNet()
        score, elo = evaluate_elo(
            model, model, n_games=4, n_sims=3, device="cpu"
        )
        assert 0.0 <= score <= 1.0, f"score={score} out of [0,1]"
        assert np.isfinite(elo)
        assert 968.0 <= elo <= 1032.0, f"elo={elo} out of [968, 1032]"


class TestEndgameCurriculum:
    """T-14 gate: endgame games mixed into a round + the finish-rate metric."""

    def test_collect_selfplay_endgame_adds_terminal_samples(self):
        """endgame_n>0 returns MORE samples than the same round without endgames,
        and the endgame games contribute a true terminal +/-1 value (the signal
        the all-draw fresh games never produce)."""
        from azero.pipeline import _collect_selfplay

        torch.manual_seed(1234)
        model = HalmaNet()
        model.eval()
        # n_sims high enough that MCTS finds the 1-move endgame win (the real
        # run uses --sims 100); the base fresh games are all draws within the cap.
        base = _collect_selfplay(model, n_games=2, device="cpu", n_sims=40,
                                 max_moves=8, seed=7)
        end = _collect_selfplay(model, n_games=2, device="cpu", n_sims=40,
                                max_moves=8, seed=7, endgame_n=3, endgame_full=8)
        assert len(end) > len(base), "endgame games should add samples"
        assert any(abs(v) == 1.0 for _e, _p, v in end), \
            "expected at least one true terminal +/-1 from the endgame games"

    def test_endgame_uses_dedicated_budget_and_noise_off(self):
        """T-15 fix: the endgame near-win games run noise-OFF at a dedicated
        (higher) sim budget, not the fresh-board low-sims + Dirichlet-noise
        default. The net's prior on the winning move is near-uniform, so the
        low-prior winning action gets starved to ~0 visits under noise at a low
        budget and the game never converts -> the true +/-1 target (the whole
        point of the curriculum) is never generated. Pin the fixed path: with
        `endgame_add_noise=False` + a workable `endgame_n_sims`, the endgame
        path MUST yield a genuine terminal +1 (attacker is to move and wins)."""
        from azero.pipeline import _collect_selfplay

        torch.manual_seed(1234)
        model = HalmaNet()
        model.eval()
        data = _collect_selfplay(
            model, n_games=0, device="cpu", n_sims=4, max_moves=8, seed=7,
            endgame_n=8, endgame_full=8,
            endgame_n_sims=80, endgame_add_noise=False,
        )
        assert data, "endgame path returned no samples"
        # At least one game must have decided in the attacker's favour, giving a
        # true +1 value target (the base fresh games are all draws -> never +1).
        assert any(v >= 0.99 for _e, _p, v in data), \
            "expected a true terminal +1 from the noise-off/dedicated-sims endgame path"

    def test_endgame_defaults_are_noise_off(self):
        """The endgame path must default to noise OFF (the fix); a regression
        that reverts endgame games to the fresh-board noise-on default would
        starve the low-prior winning move and lose the true +/-1 signal."""
        import inspect
        from azero.pipeline import _collect_selfplay, run_pipeline

        for fn in (_collect_selfplay, run_pipeline):
            sig = inspect.signature(fn)
            default = sig.parameters["endgame_add_noise"].default
            assert default is False, \
                f"{fn.__name__}: endgame_add_noise default must be False, got {default!r}"

    def test_run_pipeline_round_with_endgames_and_finish_rate(self):
        """A full round with endgames + the finish-rate metric completes without
        crashing and logs a finish_rate field (the integration path for T-14)."""
        model = HalmaNet()
        run_pipeline(
            n_rounds=1, games_per_round=2, n_sims=2, n_train_epochs=1,
            batch_size=4, device="cpu", model=model, max_moves=8,
            elo_n_games=2, elo_n_sims=2,
            endgame_n=2, endgame_full=8,
            finish_rate_n=6, finish_rate_sims=10,
        )
        assert os.path.isfile(LOG_PATH)
        with open(LOG_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert lines and "finish_rate=" in lines[-1], \
            f"log line missing 'finish_rate=': {lines[-1] if lines else None}"
