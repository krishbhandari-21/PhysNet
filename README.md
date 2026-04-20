# PhysNet 

A robust, modular **Physics-Informed Neural Network (PINN)** framework built from scratch in PyTorch.  
PhysNet trains neural networks to satisfy physics laws (PDEs) using automatic differentiation — no finite differences, no pre-computed data labels.

---

## Features

| Feature | Details |
|---|---|
| **Core NN** | Configurable feedforward MLP (depth, width, activation) with Xavier initialization |
| **Autograd derivatives** | All PDE residuals computed via `torch.autograd.grad` |
| **3 Physics domains** | Heat Equation, Burgers' Equation, Poisson Equation |
| **Flexible BCs/ICs** | Dirichlet, Neumann, and Initial condition handlers |
| **Optimizers** | Sequential Adam → L-BFGS training |
| **Visualizer** | Loss curves, prediction vs exact heatmaps, 1D comparison plots |
| **Sampler** | Latin Hypercube and uniform random sampling of collocation points |

---

## Theory

### What is a PINN?

A **Physics-Informed Neural Network** uses a neural network `u_θ(x, t)` as the trial solution to a PDE. The total training loss has three terms:

```
L_total = L_PDE + L_BC + L_IC
```

- **L_PDE**: Mean-squared residual of the governing equation at interior *collocation points*.
- **L_BC**: Mismatch between the network's boundary values and prescribed boundary conditions.
- **L_IC**: Mismatch between the network's initial state and the prescribed initial condition.

Minimizing `L_total` drives the network to satisfy the physics everywhere in the domain.

---

### Supported Equations

#### 1. 1D Heat Equation
```
∂u/∂t = α ∂²u/∂x²
```
- Domain: `t ∈ [0, T]`, `x ∈ [-1, 1]`
- Analytical solution (for `u(0,x) = sin(πx)`, zero Dirichlet BCs):
  `u(t,x) = exp(-α π² t) sin(πx)`

#### 2. Burgers' Equation
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```
- Domain: `t ∈ [0, 1]`, `x ∈ [-1, 1]`  
- Viscous shock formation; standard benchmark with `ν = 0.01/π`

#### 3. 2D Poisson Equation
```
∂²u/∂x² + ∂²u/∂y² = f(x, y)
```
- Domain: `(x, y) ∈ Ω ⊂ ℝ²`
- Forcing function `f` is user-specified

---

## Project Structure

```
PhysNet/
├── core/
│   ├── __init__.py
│   ├── network.py        # Configurable feedforward neural network
│   ├── loss_engine.py    # Autograd-based derivative computation
│   ├── conditions.py     # Dirichlet / Neumann / Initial condition handlers
│   └── trainer.py        # Adam + L-BFGS training loop
├── physics/
│   ├── __init__.py
│   ├── base.py           # BasePINN abstract class
│   ├── heat.py           # 1D Heat Equation PINN
│   ├── burgers.py        # Burgers' Equation PINN
│   └── poisson.py        # 2D Poisson Equation PINN
├── utils/
│   ├── __init__.py
│   ├── sampler.py        # Collocation point sampling (LHS & uniform)
│   └── visualizer.py     # Matplotlib-based plotting tools
├── examples/
│   └── run_heat.py       # End-to-end Heat Equation demo
├── tests/
│   ├── test_network.py
│   ├── test_loss_engine.py
│   ├── test_conditions.py
│   └── test_physics.py
├── AGENTS.md
├── requirements.txt
└── README.md             ← you are here
```

---

## Quick Start

### 1. Install dependencies
```bash
cd PhysNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Heat Equation example
```bash
# From the PhysNet/ directory
python examples/run_heat.py
```

This will:
- Sample 2000 Latin Hypercube collocation points in `[0,1] × [-1,1]`
- Apply `u(0,x) = sin(πx)` as the initial condition and zero Dirichlet BCs
- Train for 2000 Adam epochs + 1000 L-BFGS iterations
- Print the relative L2 error vs the analytical solution
- Save plots to `examples/`

### 3. Run the unit tests
```bash
pytest tests/ -v
```

---

## Usage Guide

### Custom Network
```python
from core.network import FeedForwardNN
import torch.nn as nn

# 2 inputs (t, x), 3 hidden layers of 128 neurons, 1 output
net = FeedForwardNN(layers=[2, 128, 128, 128, 1], activation=nn.Tanh)
```

### Custom Physics (plug in your own PDE)
```python
from physics.base import BasePINN
from core.loss_engine import compute_grad
import torch

class WaveEquationPINN(BasePINN):
    def __init__(self, network, c=1.0):
        super().__init__(network)
        self.c = c

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        u = self.forward(x)
        u_t = compute_grad(u, x)[:, 0:1]
        u_tt = compute_grad(u_t, x)[:, 0:1]
        u_x = compute_grad(u, x)[:, 1:2]
        u_xx = compute_grad(u_x, x)[:, 1:2]
        residual = u_tt - self.c**2 * u_xx
        return torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
```

### Conditions
```python
from core.conditions import DirichletCondition, NeumannCondition, InitialCondition
import torch

x_ic = torch.zeros(100, 2)   # t=0 points
u_ic = torch.sin(torch.pi * x_ic[:, 1:2])
ic = InitialCondition(x_ic, u_ic)

x_bc = torch.rand(100, 2); x_bc[:, 1] = -1.0  # left boundary
bc = DirichletCondition(x_bc, torch.zeros(100, 1))
```

### Training
```python
from core.trainer import Trainer

trainer = Trainer(pinn, pinn.compute_pde_residual, conditions=[ic, bc])
loss_history = trainer.train(x_collocation, epochs_adam=3000, epochs_lbfgs=2000)
```

---

## Results (Heat Equation)

After training, PhysNet produces plots in `examples/`:

| Plot | Description |
|---|---|
| `heat_loss.png` | Training loss (log scale) |
| `heat_heatmap.png` | 2D colormap of u(t,x) prediction |
| `heat_1d_t0.5.png` | PINN vs exact solution at t=0.5 |

Typical relative L2 error: **< 1%** after Adam + L-BFGS.

---

## Tests

```
tests/test_network.py       — forward shape, Xavier init, autograd support
tests/test_loss_engine.py   — first/second-order derivatives, multi-dim partials
tests/test_conditions.py    — Dirichlet/Neumann loss, requires_grad enforcement
tests/test_physics.py       — PDE residual shapes, finiteness, BasePINN contract
```

Run with: `pytest tests/ -v`

---

## Dependencies

```
torch      >= 2.0
numpy
scipy
matplotlib
pytest
```

---

## License
MIT
