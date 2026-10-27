import numpy as np
from pqblocks_qiea.qiea_kernel import _qiea_kernel
from typing import Optional, Tuple

def solve_ising(
    h: np.ndarray, 
    J: np.ndarray, 
    num_generations: int = 100, 
    population_size: Optional[int] = None, 
    theta_0: float = 0.05 * np.pi, 
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Public API for the Quantum-Inspired Evolutionary Algorithm (QIEA).
    Validates inputs, derives population_size if None, and invokes the optimized kernel.
    
    Args:
        h: Linear bias vector (shape: N,)
        J: Coupling matrix (shape: N x N, symmetric, zero diagonal)
        num_generations: Number of algorithmic generations (>= 1)
        population_size: Number of Q-bit individuals. If None, auto-derived as max(10, min(100, 2*N))
        theta_0: Base rotation step size (0, pi/2]
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (best_spin_vector, best_energy)
    """
    # 1. Edge case validations
    N = len(h)
    
    if N == 0:
        raise ValueError("Empty model: N=0 is not supported.")
    
    if not (np.isfinite(h).all() and np.isfinite(J).all()):
        raise ValueError("h and J must contain only finite numbers.")
    
    if J.shape != (N, N):
        raise ValueError("Dimension mismatch between h and J.")
    
    if not np.allclose(J, J.T):
        raise ValueError("Coupling matrix J must be symmetric.")
    
    if np.any(np.diag(J) != 0):
        raise ValueError("Coupling matrix J must have a zero diagonal.")
    
    # Auto-derive population size if not provided
    if population_size is None:
        population_size = max(10, min(100, 2 * N))
    elif population_size < 1:
        raise ValueError("Population size must be >= 1")
    
    # Validate hyperparameters
    if num_generations < 1:
        raise ValueError("Number of generations must be >= 1")
    if theta_0 <= 0:
        raise ValueError("Theta_0 must be > 0")
    if theta_0 > np.pi / 2:
        raise ValueError("Theta_0 must be <= pi/2")
    
    # 2. Invoke Kernel
    best_s, best_e = _qiea_kernel(h, J, num_generations, population_size, theta_0, seed)
    return best_s, best_e