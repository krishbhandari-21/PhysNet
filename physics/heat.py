import torch
from .base import BasePINN
from core.loss_engine import compute_grad

class HeatPINN(BasePINN):
    """
    PINN for the 1D Heat Equation: u_t - alpha * u_xx = 0.
    Inputs to the network are (t, x).
    """
    def __init__(self, network: torch.nn.Module, alpha: float = 1.0):
        super().__init__(network)
        self.alpha = alpha

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the PDE residual for the heat equation.
        """
        x = x.clone().requires_grad_(True)
        u = self.forward(x)
        
        grad_u = compute_grad(u, x)
        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]
        
        grad_u_x = compute_grad(u_x, x)
        u_xx = grad_u_x[:, 1:2]
        
        residual = u_t - self.alpha * u_xx
        return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
