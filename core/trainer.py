import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Callable
from .conditions import Condition

class Trainer:
    """
    Handles the training loop for a PINN.
    """
    def __init__(self, model: nn.Module, pde_residual_fn: Callable, conditions: List[Condition]):
        """
        Args:
            model (nn.Module): The neural network.
            pde_residual_fn (Callable): Function that takes collocation points and returns the PDE MSE loss.
            conditions (List[Condition]): List of boundary and initial conditions.
        """
        self.model = model
        self.pde_residual_fn = pde_residual_fn
        self.conditions = conditions
        
    def train(self, x_collocation: torch.Tensor, epochs_adam: int = 1000, epochs_lbfgs: int = 1000, lr_adam: float = 1e-3):
        """
        Trains the PINN using Adam followed by L-BFGS.
        
        Args:
            x_collocation (torch.Tensor): Collocation points.
            epochs_adam (int): Number of Adam epochs.
            epochs_lbfgs (int): Max number of L-BFGS iterations.
            lr_adam (float): Learning rate for Adam.
            
        Returns:
            List[float]: History of total loss.
        """
        loss_history = []
        
        if epochs_adam > 0:
            optimizer_adam = optim.Adam(self.model.parameters(), lr=lr_adam)
            
            # Adam phase
            for epoch in range(epochs_adam):
                optimizer_adam.zero_grad()
                loss = self._compute_total_loss(x_collocation)
                loss.backward()
                optimizer_adam.step()
                loss_history.append(loss.item())
                
                if epoch % 500 == 0 or epoch == epochs_adam - 1:
                    print(f"Adam Epoch {epoch}: Loss = {loss.item():.6e}")
                
        # L-BFGS phase
        if epochs_lbfgs > 0:
            optimizer_lbfgs = optim.LBFGS(
                self.model.parameters(), 
                max_iter=epochs_lbfgs, 
                tolerance_grad=1e-7, 
                tolerance_change=1e-9, 
                line_search_fn="strong_wolfe"
            )
            
            def closure():
                optimizer_lbfgs.zero_grad()
                loss = self._compute_total_loss(x_collocation)
                loss.backward()
                return loss
                
            optimizer_lbfgs.step(closure)
            final_loss = self._compute_total_loss(x_collocation).item()
            loss_history.append(final_loss)
            print(f"L-BFGS Phase complete: Final Loss = {final_loss:.6e}")
            
        return loss_history
        
    def _compute_total_loss(self, x_collocation: torch.Tensor) -> torch.Tensor:
        pde_loss = self.pde_residual_fn(x_collocation)
        # Start from zero that lives on the same device/dtype as pde_loss
        # and properly participates in the autograd graph.
        cond_loss = pde_loss.new_zeros(())
        for cond in self.conditions:
            cond_loss = cond_loss + cond.compute_loss(self.model)
        return pde_loss + cond_loss
