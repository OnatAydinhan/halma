"""Neural network for the AlphaZero NN (Part 3).

A CNN trunk with a shared representation feeding a policy head (417 raw
logits) and a value head (tanh scalar). Value convention is negamax:
+1 = certain win for the player to move, -1 = certain loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 7
BOARD = 8
N_ACTIONS = 417


class HalmaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * BOARD * BOARD, 256)
        self.policy_head = nn.Linear(256, N_ACTIONS)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 7, 8, 8)
        h = self.trunk(x)
        h = h.flatten(1)          # (B, 4096)
        h = F.relu(self.fc(h))    # (B, 256)
        logits = self.policy_head(h)   # (B, 417) raw logits
        value = torch.tanh(self.value_head(h))  # (B, 1) in [-1, 1]
        return logits, value

    def policy_and_value(self, x):
        """Alias for forward."""
        return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def policy_loss(logits, target):
    """Soft-target cross-entropy over the full 417-dim distribution.

    logits: (B, 417), target: (B, 417) probability distribution.
    Returns a scalar tensor.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def value_loss(pred, target):
    """MSE between predicted value and target. Both (B, 1)."""
    return F.mse_loss(pred.squeeze(-1), target.squeeze(-1))


def total_loss(logits, policy_target, value_pred, value_target):
    """policy_loss + value_loss, a scalar tensor."""
    return policy_loss(logits, policy_target) + value_loss(value_pred, value_target)
