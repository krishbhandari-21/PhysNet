"""
End-to-end demo: Burgers' Equation PINN.

PDE:  u_t + u * u_x = nu * u_xx
Domain: t in [0, 1],  x in [-1, 1]
IC:   u(0, x) = -sin(pi * x)
BC:   u(t, -1) = u(t,  1) = 0
nu  = 0.01 / pi  (standard benchmark)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np

from core.network import FeedForwardNN
from core.conditions import DirichletCondition, InitialCondition
from core.trainer import Trainer
from physics.burgers import BurgersPINN
from utils.sampler import Sampler
from utils.visualizer import Visualizer


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    nu = 0.01 / math.pi

    # Network: 2 inputs (t, x) → 4 hidden layers → 1 output
    net = FeedForwardNN([2, 64, 64, 64, 64, 1], activation=torch.nn.Tanh)
    pinn = BurgersPINN(net, nu=nu)

    # ── Collocation points ──────────────────────────────────────────────
    bounds = [(0.0, 1.0), (-1.0, 1.0)]
    x_col = Sampler.latin_hypercube(bounds, 2000)

    # ── Initial Condition: u(0, x) = -sin(pi * x) ───────────────────────
    x_ic = Sampler.uniform([(0.0, 0.0), (-1.0, 1.0)], 200)
    u_ic = -torch.sin(math.pi * x_ic[:, 1:2])
    ic = InitialCondition(x_ic, u_ic)

    # ── Boundary Conditions: u(t, ±1) = 0 ───────────────────────────────
    x_bc_l = Sampler.uniform([(0.0, 1.0), (-1.0, -1.0)], 100)
    x_bc_r = Sampler.uniform([(0.0, 1.0), (1.0,  1.0)], 100)
    bc_l = DirichletCondition(x_bc_l, torch.zeros(100, 1))
    bc_r = DirichletCondition(x_bc_r, torch.zeros(100, 1))

    # ── Train ────────────────────────────────────────────────────────────
    print("Training Burgers PINN …")
    trainer = Trainer(pinn, pinn.compute_pde_residual, [ic, bc_l, bc_r])
    loss_history = trainer.train(x_col, epochs_adam=3000, epochs_lbfgs=500, lr_adam=1e-3)

    # ── Evaluate ─────────────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    Visualizer.plot_loss(loss_history, filename=os.path.join(out_dir, "burgers_loss.png"))

    t_vals = torch.linspace(0, 1, 100)
    x_vals = torch.linspace(-1, 1, 100)
    T, X = torch.meshgrid(t_vals, x_vals, indexing="ij")
    TX = torch.stack([T.flatten(), X.flatten()], dim=-1)

    with torch.no_grad():
        u_pred = net(TX).reshape(100, 100)

    Visualizer.plot_2d_heatmap(
        T, X, u_pred,
        xlabel="t", ylabel="x",
        title="Burgers' Equation Prediction",
        filename=os.path.join(out_dir, "burgers_heatmap.png"),
    )

    # 1-D slice at t = 0.25
    idx = 25
    Visualizer.plot_1d_comparison(
        x_vals, u_pred[idx, :], u_pred[idx, :],  # no closed form; compare to self
        title="Burgers' Equation at t=0.25",
        filename=os.path.join(out_dir, "burgers_1d_t0.25.png"),
    )
    print("Done. Plots saved to examples/")


if __name__ == "__main__":
    main()
