import sys
import os
# Add parent directory to sys.path to allow imports from PhysNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from core.network import FeedForwardNN
from core.conditions import DirichletCondition, InitialCondition
from core.trainer import Trainer
from physics.heat import HeatPINN
from utils.sampler import Sampler
from utils.visualizer import Visualizer

def exact_solution(t, x, alpha):
    """Analytical solution for the specific boundary and initial conditions."""
    return torch.exp(-alpha * (torch.pi ** 2) * t) * torch.sin(torch.pi * x)

def main():
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    alpha = 0.01
    
    # 1. Define Network & Physics
    # 2 inputs (t, x) -> 3 hidden layers of 64 -> 1 output u(t, x)
    layers = [2, 64, 64, 64, 1]
    net = FeedForwardNN(layers, activation=torch.nn.Tanh)
    pinn = HeatPINN(net, alpha=alpha)
    
    # 2. Sample Collocation Points
    # Domain: t in [0, 1], x in [-1, 1]
    bounds = [(0.0, 1.0), (-1.0, 1.0)]
    n_collocation = 2000
    x_collocation = Sampler.latin_hypercube(bounds, n_collocation)
    
    # 3. Define Boundary and Initial Conditions
    # Initial Condition: t=0, x in [-1, 1] -> u(0, x) = sin(pi * x)
    n_ic = 100
    x_ic = Sampler.uniform([(0.0, 0.0), (-1.0, 1.0)], n_ic)
    u_ic = exact_solution(x_ic[:, 0:1], x_ic[:, 1:2], alpha)
    ic = InitialCondition(x_ic, u_ic)
    
    # Boundary Conditions: x=-1 and x=1 -> u(t, +/-1) = 0
    n_bc = 100
    x_bc_left = Sampler.uniform([(0.0, 1.0), (-1.0, -1.0)], n_bc)
    u_bc_left = torch.zeros((n_bc, 1))
    bc_left = DirichletCondition(x_bc_left, u_bc_left)
    
    x_bc_right = Sampler.uniform([(0.0, 1.0), (1.0, 1.0)], n_bc)
    u_bc_right = torch.zeros((n_bc, 1))
    bc_right = DirichletCondition(x_bc_right, u_bc_right)
    
    conditions = [ic, bc_left, bc_right]
    
    # 4. Train
    trainer = Trainer(pinn, pinn.compute_pde_residual, conditions)
    print("Training Heat PINN...")
    loss_history = trainer.train(x_collocation, epochs_adam=2000, epochs_lbfgs=1000, lr_adam=1e-3)
    
    # 5. Evaluate and Visualize
    print("Evaluating...")
    Visualizer.plot_loss(loss_history, filename=os.path.join(os.path.dirname(__file__), "heat_loss.png"))
    
    # Grid for evaluation
    t_test = torch.linspace(0, 1, 100)
    x_test = torch.linspace(-1, 1, 100)
    T, X = torch.meshgrid(t_test, x_test, indexing='ij')
    
    TX = torch.stack([T.flatten(), X.flatten()], dim=-1)
    with torch.no_grad():
        u_pred = net(TX).reshape(100, 100)
        
    u_exact = exact_solution(T, X, alpha)
    
    error = torch.norm(u_exact - u_pred, 2) / torch.norm(u_exact, 2)
    print(f"Relative L2 Error: {error.item():.4e}")
    
    # Heatmap
    Visualizer.plot_2d_heatmap(
        T, X, u_pred, 
        xlabel="t", ylabel="x", 
        title="Heat Equation Prediction", 
        filename=os.path.join(os.path.dirname(__file__), "heat_heatmap.png")
    )
    
    # 1D comparison at t=0.5
    idx_t = 50 # corresponds to t=0.5
    Visualizer.plot_1d_comparison(
        x_test, u_pred[idx_t, :], u_exact[idx_t, :], 
        title="Heat Equation at t=0.5", 
        filename=os.path.join(os.path.dirname(__file__), "heat_1d_t0.5.png")
    )

if __name__ == "__main__":
    main()
