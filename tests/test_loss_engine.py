"""Unit tests for core.loss_engine module."""

import pytest
import torch
from core.loss_engine import compute_grad


def test_first_order_derivative():
    """Gradient of x^2 w.r.t x should be 2x."""
    x = torch.linspace(-1, 1, 50).unsqueeze(1).requires_grad_(True)
    y = x ** 2
    grad = compute_grad(y, x)
    expected = 2 * x.detach()
    assert torch.allclose(grad.detach(), expected, atol=1e-5)


def test_gradient_shape():
    """Gradient shape should match input shape."""
    x = torch.rand(20, 2, requires_grad=True)
    y = (x ** 2).sum(dim=1, keepdim=True)
    grad = compute_grad(y, x)
    assert grad.shape == x.shape


def test_partial_derivative_multidim():
    """For y = x0^2 + x1^3, dy/dx0 = 2*x0 and dy/dx1 = 3*x1^2."""
    x = torch.rand(30, 2, requires_grad=True)
    y = x[:, 0:1] ** 2 + x[:, 1:2] ** 3
    grad = compute_grad(y, x)
    expected_dx0 = 2 * x[:, 0:1].detach()
    expected_dx1 = 3 * x[:, 1:2].detach() ** 2
    assert torch.allclose(grad[:, 0:1].detach(), expected_dx0, atol=1e-5)
    assert torch.allclose(grad[:, 1:2].detach(), expected_dx1, atol=1e-5)


def test_second_order_derivative():
    """Second derivative of x^3 w.r.t x should be 6x."""
    x = torch.linspace(-1, 1, 50).unsqueeze(1).requires_grad_(True)
    y = x ** 3
    dy = compute_grad(y, x)
    d2y = compute_grad(dy, x)
    expected = 6 * x.detach()
    assert torch.allclose(d2y.detach(), expected, atol=1e-5)


def test_unused_returns_zeros():
    """If output does not depend on input, gradient should be zeros."""
    x = torch.rand(10, 2, requires_grad=True)
    y = torch.ones(10, 1)  # constant, no dependency on x
    grad = compute_grad(y, x)
    assert torch.all(grad == 0)
