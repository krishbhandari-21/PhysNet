import torch

def compute_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Computes the first derivative of outputs with respect to inputs using automatic differentiation.

    Args:
        outputs (torch.Tensor): The output tensor (e.g., neural network predictions).
                                Shape: (N, 1) or similar.
        inputs (torch.Tensor): The input tensor with requires_grad=True (e.g., coordinates).
                               Shape: (N, D).

    Returns:
        torch.Tensor: The gradient of outputs w.r.t inputs.
                      Shape: (N, D).
    """
    grad_outputs = torch.ones_like(outputs)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grad is None:
        return torch.zeros_like(inputs)
        
    return grad
