import torch
import torch.nn as nn
from typing import List, Type

class FeedForwardNN(nn.Module):
    """
    A configurable fully connected feedforward neural network.
    """
    def __init__(
        self,
        layers: List[int],
        activation: Type[nn.Module] = nn.Tanh
    ):
        """
        Initializes the feedforward neural network.

        Args:
            layers (List[int]): Number of neurons in each layer, including input and output.
                                Example: [2, 64, 64, 1] means 2 inputs, 2 hidden layers of 64, 1 output.
            activation (Type[nn.Module]): Activation function module to use between layers.
                                          Defaults to `nn.Tanh`.
        """
        super().__init__()
        self.layers = layers
        self.activation = activation()
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(self.activation)
                
        self.network = nn.Sequential(*modules)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initializes weights using Xavier Normal, which is typically good for PINNs with Tanh.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)
