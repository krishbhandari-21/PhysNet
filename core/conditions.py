import torch
import torch.nn as nn

class Condition:
    """
    Base class for tracking boundary and initial conditions.
    """
    def __init__(self, x: torch.Tensor, values: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Coordinates for the condition.
            values (torch.Tensor): The prescribed values.
        """
        self.x = x
        self.x.requires_grad_(True)
        self.values = values

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        raise NotImplementedError

class DirichletCondition(Condition):
    """
    Dirichlet boundary condition (prescribed values).
    """
    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        pred = model(self.x)
        return torch.nn.functional.mse_loss(pred, self.values)

class NeumannCondition(Condition):
    """
    Neumann boundary condition (prescribed derivatives).
    """
    def __init__(self, x: torch.Tensor, values: torch.Tensor, dim: int = 1):
        """
        Args:
            dim (int): The dimension index along which to compute the derivative.
                       e.g. if x = (t, space_x), and we want du/dx, dim=1.
        """
        super().__init__(x, values)
        self.dim = dim

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        from core.loss_engine import compute_grad
        pred = model(self.x)
        grad = compute_grad(pred, self.x)
        pred_derivative = grad[:, self.dim:self.dim+1]
        return torch.nn.functional.mse_loss(pred_derivative, self.values)

class InitialCondition(DirichletCondition):
    """
    Initial condition, mathematically identical to Dirichlet condition.
    """
    pass
