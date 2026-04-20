"""Unit tests for core.conditions module."""

import pytest
import torch
import torch.nn as nn
from core.conditions import DirichletCondition, NeumannCondition, InitialCondition
from core.network import FeedForwardNN


@pytest.fixture
def simple_model():
    """A small network for testing conditions."""
    return FeedForwardNN([2, 16, 1])


def test_dirichlet_zero_loss_for_exact(simple_model):
    """Dirichlet loss should be near zero when model matches exactly."""
    x = torch.rand(20, 2)
    with torch.no_grad():
        u_exact = simple_model(x)
    cond = DirichletCondition(x.clone(), u_exact.clone())
    loss = cond.compute_loss(simple_model)
    assert loss.item() < 1e-10


def test_dirichlet_nonzero_loss(simple_model):
    """Dirichlet loss should be positive when model doesn't match."""
    x = torch.rand(20, 2)
    values = torch.ones(20, 1) * 999.0  # far from any sane output
    cond = DirichletCondition(x, values)
    loss = cond.compute_loss(simple_model)
    assert loss.item() > 0


def test_initial_condition_is_dirichlet(simple_model):
    """InitialCondition should behave identically to DirichletCondition."""
    x = torch.zeros(10, 2)
    values = torch.zeros(10, 1)
    ic = InitialCondition(x, values)
    loss = ic.compute_loss(simple_model)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_neumann_condition(simple_model):
    """NeumannCondition should compute a gradient-based loss."""
    x = torch.rand(10, 2)
    x.requires_grad_(True)
    values = torch.zeros(10, 1)
    cond = NeumannCondition(x, values, dim=1)
    loss = cond.compute_loss(simple_model)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_condition_requires_grad_enabled():
    """Condition should automatically enable requires_grad on x."""
    x = torch.rand(5, 2)
    assert not x.requires_grad
    cond = DirichletCondition(x, torch.zeros(5, 1))
    assert cond.x.requires_grad
