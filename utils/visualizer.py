import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch

class Visualizer:
    @staticmethod
    def plot_loss(loss_history, filename="loss_curve.png"):
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, label="Total Loss", color='blue')
        plt.yscale("log")
        plt.xlabel("Epoch / Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_1d_comparison(x, u_pred, u_exact, title="PINN vs Exact Solution", filename="comparison_1d.png"):
        plt.figure(figsize=(8, 6))
        
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if isinstance(u_pred, torch.Tensor): u_pred = u_pred.detach().cpu().numpy()
        if isinstance(u_exact, torch.Tensor): u_exact = u_exact.detach().cpu().numpy()

        plt.plot(x, u_exact, 'b-', label="Exact Solution", linewidth=2)
        plt.plot(x, u_pred, 'r--', label="PINN Prediction", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    @staticmethod
    def plot_2d_heatmap(x, y, u, xlabel="t", ylabel="x", title="Prediction Heatmap", filename="heatmap.png"):
        plt.figure(figsize=(8, 6))
        
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
        if isinstance(u, torch.Tensor): u = u.detach().cpu().numpy()
            
        plt.tricontourf(x.flatten(), y.flatten(), u.flatten(), levels=100, cmap='viridis')
        plt.colorbar(label="u")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
