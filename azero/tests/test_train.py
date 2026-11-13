"""Tests for azero/train.py (T-5)."""

import os
import tempfile

import numpy as np
import pytest
import torch

from azero.model import HalmaNet
from azero.train import load_checkpoint, save_checkpoint, train, train_step

N_CHANNELS = 7
BOARD = 8
N_ACTIONS = 417


def _make_batch(n=64):
    """Synthetic training batch: random states, normalized policies, +/-1 values."""
    rng = np.random.default_rng(42)
    batch = []
    for _ in range(n):
        enc = torch.randn(N_CHANNELS, BOARD, BOARD)
        policy = rng.random(N_ACTIONS).astype(np.float32)
        policy /= policy.sum()
        value = float(rng.choice([-1.0, 1.0]))
        batch.append((enc, policy, value))
    return batch


class TestTrainStep:
    def test_returns_finite_float(self):
        model = HalmaNet()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        batch = _make_batch(16)
        loss = train_step(model, opt, batch, "cpu")
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_gradients_flow(self):
        model = HalmaNet()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        batch = _make_batch(8)
        train_step(model, opt, batch, "cpu")
        for p in model.parameters():
            assert p.grad is not None


class TestTrain:
    def test_completes_and_finite(self):
        model = HalmaNet()
        data = _make_batch(64)
        opt, avg_loss = train(
            model, data, n_epochs=2, batch_size=16, lr=0.001, device="cpu"
        )
        assert isinstance(opt, torch.optim.Adam)
        import numpy as _np
        assert _np.isfinite(avg_loss)
        # After training the model should still produce finite outputs.
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, N_CHANNELS, BOARD, BOARD)
            logits, val = model(x)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(val).all()

    def test_loss_finite_both_epochs(self):
        model = HalmaNet()
        data = _make_batch(64)
        losses = []
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        import random
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
        for epoch in range(2):
            random.shuffle(data)
            epoch_losses = []
            for i in range(0, len(data), 16):
                batch = data[i : i + 16]
                l = train_step(model, opt, batch, "cpu")
                epoch_losses.append(l)
            losses.append(sum(epoch_losses) / len(epoch_losses))
            scheduler.step()
        # Both epoch losses must be finite (random data has no learnable signal,
        # so we can't assert strict decrease).
        for l in losses:
            assert np.isfinite(l)


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        model = HalmaNet()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        # Run one step to give the optimizer some state.
        train_step(model, opt, _make_batch(8), "cpu")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(model, opt, round_no=7, path=path)
            assert os.path.isfile(path)

            # Create a fresh model and load.
            model2 = HalmaNet()
            opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            round_no = load_checkpoint(model2, opt2, path, "cpu")
            assert round_no == 7
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2, atol=1e-7)


class TestPostTrain:
    def test_eval_forward_after_train(self):
        model = HalmaNet()
        data = _make_batch(32)
        train(model, data, n_epochs=1, batch_size=16, device="cpu")
        model.eval()
        with torch.no_grad():
            x = torch.randn(4, N_CHANNELS, BOARD, BOARD)
            logits, val = model(x)
        assert logits.shape == (4, N_ACTIONS)
        assert val.shape == (4, 1)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(val).all()
