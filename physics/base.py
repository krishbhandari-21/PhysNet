import torch
import torch.nn as nn

class BasePINN(nn.Module):
    """
    Base class for Physics-Informed Neural Networks.
    Subclasses must implement the compute_pde_residual method.
    """
    def __init__(self, network: nn.Module):
        """
        Args:
            network (nn.Module): The underlying neural network for predictions.
        """
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Coordinates tensor.
            
        Returns:
            torch.Tensor: Network predictions.
        """
        return self.network(x)

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the PDE residual loss.
        
        Args:
            x (torch.Tensor): Collocation points.
            
        Returns:
            torch.Tensor: MSE loss of the PDE residual.
        """
        raise NotImplementedError
