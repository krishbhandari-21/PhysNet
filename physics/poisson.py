import torch
from typing import Callable
from .base import BasePINN
from core.loss_engine import compute_grad

class PoissonPINN(BasePINN):
    """
    PINN for the 2D Poisson Equation: u_xx + u_yy = f(x, y).
    Inputs to the network are (x, y).
    """
    def __init__(self, network: torch.nn.Module, forcing_fn: Callable):
        super().__init__(network)
        self.forcing_fn = forcing_fn

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the PDE residual for Poisson's equation.
        """
        x = x.clone().requires_grad_(True)
        u = self.forward(x)
        
        grad_u = compute_grad(u, x)
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        grad_u_x = compute_grad(u_x, x)
        u_xx = grad_u_x[:, 0:1]
        
        grad_u_y = compute_grad(u_y, x)
        u_yy = grad_u_y[:, 1:2]
        
        f = self.forcing_fn(x)
        
        residual = u_xx + u_yy - f
        return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
