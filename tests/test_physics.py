"""Unit tests for physics modules."""

import pytest
import torch
import math
from core.network import FeedForwardNN
from physics.heat import HeatPINN
from physics.burgers import BurgersPINN
from physics.poisson import PoissonPINN


@pytest.fixture
def heat_pinn():
    net = FeedForwardNN([2, 16, 16, 1])
    return HeatPINN(net, alpha=1.0)


@pytest.fixture
def burgers_pinn():
    net = FeedForwardNN([2, 16, 16, 1])
    return BurgersPINN(net, nu=(0.01 / math.pi))


@pytest.fixture
def poisson_pinn():
    net = FeedForwardNN([2, 16, 16, 1])
    forcing = lambda x: -2.0 * (torch.sin(x[:, 0:1]) + torch.sin(x[:, 1:2]))
    return PoissonPINN(net, forcing_fn=forcing)


# ---- HeatPINN ----

def test_heat_forward_shape(heat_pinn):
    x = torch.rand(20, 2)
    out = heat_pinn(x)
    assert out.shape == (20, 1)


def test_heat_pde_residual_is_tensor(heat_pinn):
    x = torch.rand(10, 2, requires_grad=True)
    loss = heat_pinn.compute_pde_residual(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_heat_pde_residual_finite(heat_pinn):
    x = torch.rand(50, 2, requires_grad=True)
    loss = heat_pinn.compute_pde_residual(x)
    assert torch.isfinite(loss)


# ---- BurgersPINN ----

def test_burgers_forward_shape(burgers_pinn):
    x = torch.rand(20, 2)
    out = burgers_pinn(x)
    assert out.shape == (20, 1)


def test_burgers_pde_residual_is_tensor(burgers_pinn):
    x = torch.rand(10, 2, requires_grad=True)
    loss = burgers_pinn.compute_pde_residual(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_burgers_pde_residual_finite(burgers_pinn):
    x = torch.rand(50, 2, requires_grad=True)
    loss = burgers_pinn.compute_pde_residual(x)
    assert torch.isfinite(loss)


# ---- PoissonPINN ----

def test_poisson_forward_shape(poisson_pinn):
    x = torch.rand(20, 2)
    out = poisson_pinn(x)
    assert out.shape == (20, 1)


def test_poisson_pde_residual_is_tensor(poisson_pinn):
    x = torch.rand(10, 2, requires_grad=True)
    loss = poisson_pinn.compute_pde_residual(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_poisson_pde_residual_finite(poisson_pinn):
    x = torch.rand(50, 2, requires_grad=True)
    loss = poisson_pinn.compute_pde_residual(x)
    assert torch.isfinite(loss)


# ---- BasePINN ----

def test_base_pinn_raises_not_implemented():
    """BasePINN.compute_pde_residual must be overridden."""
    from physics.base import BasePINN
    net = FeedForwardNN([2, 8, 1])
    pinn = BasePINN(net)
    with pytest.raises(NotImplementedError):
        pinn.compute_pde_residual(torch.rand(5, 2))
