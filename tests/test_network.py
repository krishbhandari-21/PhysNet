"""Unit tests for core.network module."""

import pytest
import torch
import torch.nn as nn
from core.network import FeedForwardNN


def test_output_shape():
    """Network output shape should match the last layer size."""
    net = FeedForwardNN([2, 32, 32, 1])
    x = torch.rand(50, 2)
    out = net(x)
    assert out.shape == (50, 1)


def test_custom_activation():
    """Network should accept a custom activation function."""
    net = FeedForwardNN([2, 16, 1], activation=nn.ReLU)
    x = torch.rand(10, 2)
    out = net(x)
    assert out.shape == (10, 1)


def test_weights_initialized():
    """Weights should not all be zero after initialization."""
    net = FeedForwardNN([3, 64, 1])
    for param in net.parameters():
        assert not torch.all(param == 0)


def test_forward_differentiable():
    """Network output should support autograd."""
    net = FeedForwardNN([2, 16, 1])
    x = torch.rand(10, 2, requires_grad=True)
    out = net(x)
    grad = torch.autograd.grad(out.sum(), x)[0]
    assert grad.shape == x.shape


def test_deep_network():
    """A deeper network should still produce correct output shape."""
    layers = [2, 64, 64, 64, 64, 1]
    net = FeedForwardNN(layers)
    x = torch.rand(100, 2)
    out = net(x)
    assert out.shape == (100, 1)
