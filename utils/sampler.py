import torch
import numpy as np
from scipy.stats import qmc

class Sampler:
    """
    Utility class for sampling collocation points.
    """
    @staticmethod
    def latin_hypercube(bounds: list, num_samples: int) -> torch.Tensor:
        """
        Generates samples using Latin Hypercube Sampling.
        
        Args:
            bounds (list): List of tuples [(min_1, max_1), ..., (min_d, max_d)]
            num_samples (int): Number of points to sample.
            
        Returns:
            torch.Tensor: Sampled points of shape (num_samples, d).
        """
        d = len(bounds)
        sampler = qmc.LatinHypercube(d=d)
        sample = sampler.random(n=num_samples)
        
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        scaled_sample = lower_bounds + sample * (upper_bounds - lower_bounds)
        return torch.tensor(scaled_sample, dtype=torch.float32)

    @staticmethod
    def uniform(bounds: list, num_samples: int) -> torch.Tensor:
        """
        Generates samples uniformly.
        """
        d = len(bounds)
        samples = []
        for b in bounds:
            samples.append(torch.rand(num_samples, 1) * (b[1] - b[0]) + b[0])
        return torch.cat(samples, dim=1)
