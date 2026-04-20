"""Unit tests for utils.sampler module."""
import pytest
import torch
from utils.sampler import Sampler


def test_lhs_shape():
    """LHS should return (num_samples, d) tensor."""
    bounds = [(0.0, 1.0), (-1.0, 1.0)]
    pts = Sampler.latin_hypercube(bounds, 100)
    assert pts.shape == (100, 2)


def test_lhs_within_bounds():
    """All LHS samples should lie within the specified bounds."""
    bounds = [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]
    pts = Sampler.latin_hypercube(bounds, 500)
    for i, (lo, hi) in enumerate(bounds):
        assert pts[:, i].min().item() >= lo - 1e-6
        assert pts[:, i].max().item() <= hi + 1e-6


def test_uniform_shape():
    """Uniform sampler should return (num_samples, d) tensor."""
    bounds = [(0.0, 1.0), (-1.0, 1.0)]
    pts = Sampler.uniform(bounds, 200)
    assert pts.shape == (200, 2)


def test_uniform_within_bounds():
    """All uniform samples should lie within specified bounds."""
    bounds = [(0.5, 1.5), (-2.0, -1.0)]
    pts = Sampler.uniform(bounds, 500)
    for i, (lo, hi) in enumerate(bounds):
        assert pts[:, i].min().item() >= lo - 1e-6
        assert pts[:, i].max().item() <= hi + 1e-6


def test_uniform_degenerate_range():
    """Degenerate range (lo==hi) should produce all-equal values — used for BCs."""
    bounds = [(0.0, 0.0), (-1.0, 1.0)]
    pts = Sampler.uniform(bounds, 50)
    assert torch.all(pts[:, 0] == 0.0)


def test_lhs_dtype():
    """LHS output should be float32."""
    pts = Sampler.latin_hypercube([(0.0, 1.0)], 10)
    assert pts.dtype == torch.float32


def test_uniform_dtype():
    """Uniform output should be float32."""
    pts = Sampler.uniform([(0.0, 1.0)], 10)
    assert pts.dtype == torch.float32
