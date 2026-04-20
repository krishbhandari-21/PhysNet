"""
End-to-end demo: 2D Poisson Equation PINN.

PDE:  u_xx + u_yy = f(x, y)
where  f(x,y) = -2 * pi^2 * sin(pi*x) * sin(pi*y)
Exact:  u(x, y) = sin(pi*x) * sin(pi*y)

Domain: (x, y) in [0, 1]^2
BC:  u = 0 on all four edges (Dirichlet)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np

from core.network import FeedForwardNN
from core.conditions import DirichletCondition
from core.trainer import Trainer
from physics.poisson import PoissonPINN
from utils.sampler import Sampler
from utils.visualizer import Visualizer


def forcing_fn(x: torch.Tensor) -> torch.Tensor:
    """Right-hand side f(x,y) = -2π² sin(πx) sin(πy)."""
    return -2.0 * math.pi ** 2 * torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    """Analytical solution u(x,y) = sin(πx) sin(πy)."""
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


def boundary_points(n: int, edge_val: float, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: sample `n` boundary points on one edge, return (coords, values)."""
    other = torch.rand(n, 1)
    if axis == 0:
        pts = torch.cat([torch.full((n, 1), edge_val), other], dim=1)
    else:
        pts = torch.cat([other, torch.full((n, 1), edge_val)], dim=1)
    vals = exact_solution(pts)
    return pts, vals


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # Network: 2 inputs (x, y) → 4 hidden layers → 1 output
    net = FeedForwardNN([2, 64, 64, 64, 64, 1], activation=torch.nn.Tanh)
    pinn = PoissonPINN(net, forcing_fn=forcing_fn)

    # ── Collocation points ──────────────────────────────────────────────
    x_col = Sampler.latin_hypercube([(0.0, 1.0), (0.0, 1.0)], 2000)

    # ── Boundary Conditions (all four edges, u = 0) ─────────────────────
    conditions = []
    n_bc = 100
    for axis, val in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)]:
        pts, vals = boundary_points(n_bc, val, axis)
        conditions.append(DirichletCondition(pts, vals))

    # ── Train ────────────────────────────────────────────────────────────
    print("Training Poisson PINN …")
    trainer = Trainer(pinn, pinn.compute_pde_residual, conditions)
    loss_history = trainer.train(x_col, epochs_adam=3000, epochs_lbfgs=500, lr_adam=1e-3)

    # ── Evaluate ─────────────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    Visualizer.plot_loss(loss_history, filename=os.path.join(out_dir, "poisson_loss.png"))

    x_vals = torch.linspace(0, 1, 100)
    y_vals = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    with torch.no_grad():
        u_pred = net(XY).reshape(100, 100)
        u_exact_grid = exact_solution(XY).reshape(100, 100)

    error = torch.norm(u_exact_grid - u_pred) / torch.norm(u_exact_grid)
    print(f"Relative L2 Error: {error.item():.4e}")

    Visualizer.plot_2d_heatmap(
        X, Y, u_pred,
        xlabel="x", ylabel="y",
        title="Poisson Equation — PINN Prediction",
        filename=os.path.join(out_dir, "poisson_heatmap_pred.png"),
    )
    Visualizer.plot_2d_heatmap(
        X, Y, u_exact_grid,
        xlabel="x", ylabel="y",
        title="Poisson Equation — Exact Solution",
        filename=os.path.join(out_dir, "poisson_heatmap_exact.png"),
    )
    Visualizer.plot_2d_heatmap(
        X, Y, torch.abs(u_exact_grid - u_pred),
        xlabel="x", ylabel="y",
        title="Poisson Equation — Absolute Error",
        filename=os.path.join(out_dir, "poisson_heatmap_error.png"),
    )
    print("Done. Plots saved to examples/")


if __name__ == "__main__":
    main()
