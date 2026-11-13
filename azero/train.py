"""Training loop for the AlphaZero NN (Part 3, T-5).

Trains HalmaNet on (state, policy, value) tuples produced by self-play.
Supports checkpointing so a long run can be resumed after interruption.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from azero.model import policy_loss, value_loss


def train_step(model, optimizer, batch, device):
    """One training step on a list of (enc_tensor, policy_np, value_float).

    Returns the scalar loss as a float.
    """
    model.train()

    states = torch.stack([enc for enc, _, _ in batch]).to(device)
    policies = torch.tensor(
        np.stack([p for _, p, _ in batch]), dtype=torch.float32
    ).to(device)
    values = torch.tensor(
        [v for _, _, v in batch], dtype=torch.float32
    ).unsqueeze(1).to(device)

    logits, value_pred = model(states)
    loss = policy_loss(logits, policies) + value_loss(value_pred, values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


def train(
    model,
    data,
    n_epochs=10,
    batch_size=256,
    lr=0.002,
    device="cpu",
    optimizer=None,
):
    """Train for `n_epochs` on the data list.

    Returns (optimizer, avg_loss) where avg_loss is the mean loss across all
    epochs, so the caller can save the optimizer and log the loss.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    all_losses = []
    for epoch in range(n_epochs):
        random.shuffle(data)
        epoch_losses = []
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            loss = train_step(model, optimizer, batch, device)
            epoch_losses.append(loss)
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        all_losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{n_epochs}  avg_loss={avg_loss:.6f}")
        scheduler.step()

    overall_avg = sum(all_losses) / max(len(all_losses), 1)
    return optimizer, overall_avg


def save_checkpoint(model, optimizer, round_no, path):
    """Save model + optimizer state to `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "round": round_no,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    """Load model + optimizer state from `path`. Returns the saved round_no."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # The optimizer's moving buffers (Adam exp_avg / exp_avg_sq) load on CPU,
    # but the model + grads train on `device`. If left on CPU, Adam's first
    # step raises a mixed-device RuntimeError. Relocate every buffer tensor.
    dev = device if isinstance(device, torch.device) else torch.device(device)
    for state in optimizer.state.values():
        for key in list(state.keys()):
            if torch.is_tensor(state[key]) and state[key].device.type != dev.type:
                state[key] = state[key].to(dev)
    return ckpt["round"]
