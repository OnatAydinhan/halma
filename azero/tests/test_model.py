"""Tests for azero/model.py (T-2)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from azero.model import HalmaNet, policy_loss, value_loss, total_loss

N_ACTIONS = 417
torch.manual_seed(0)


def _random_input(batch=2):
    return torch.randn(batch, 7, 8, 8, dtype=torch.float32)


def test_instantiation():
    net = HalmaNet()
    assert isinstance(net, torch.nn.Module)


def test_forward_cpu_shapes_and_finite():
    net = HalmaNet().eval()
    x = _random_input(2)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (2, N_ACTIONS)
    assert value.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    # value in [-1, 1]
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_forward_gpu():
    if not torch.cuda.is_available():
        return  # skip gracefully
    net = HalmaNet().to("cuda").eval()
    x = _random_input(2).to("cuda")
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (2, N_ACTIONS)
    assert value.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_count_parameters_range():
    net = HalmaNet()
    n = net.count_parameters()
    assert 1_000_000 < n < 3_000_000, f"param count {n} out of range"


def test_backward_gradients_populated():
    net = HalmaNet()
    x = _random_input(2)
    policy_target = torch.softmax(torch.randn(2, N_ACTIONS), dim=-1)
    value_target = torch.tensor([[1.0], [-1.0]])
    logits, value = net(x)
    loss = total_loss(logits, policy_target, value, value_target)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_policy_loss_one_hot_uniform_finite_positive():
    logits = torch.zeros(1, N_ACTIONS)  # uniform logits
    target = torch.zeros(1, N_ACTIONS)
    target[0, 0] = 1.0
    loss = policy_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_value_loss_mse():
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[-1.0]])
    loss = value_loss(pred, target)
    # (1 - (-1))^2 = 4
    assert abs(loss.item() - 4.0) < 1e-5


def test_save_load_state_dict():
    net1 = HalmaNet().eval()
    net2 = HalmaNet()
    net2.load_state_dict(net1.state_dict())
    net2.eval()
    x = _random_input(2)
    with torch.no_grad():
        l1, v1 = net1(x)
        l2, v2 = net2(x)
    assert torch.allclose(l1, l2)
    assert torch.allclose(v1, v2)
