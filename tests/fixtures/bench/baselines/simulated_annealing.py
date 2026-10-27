import numpy as np
from typing import Tuple, Optional


def simulated_annealing(
    h: np.ndarray,
    J: np.ndarray,
    num_sweeps: int = 1000,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Simulated Annealing for Ising/QUBO optimization.
    
    Args:
        h: Linear bias vector (shape: N,)
        J: Coupling matrix (shape: N x N, symmetric, zero diagonal)
        num_sweeps: Number of temperature sweeps
        initial_temp: Initial temperature
        final_temp: Final temperature
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_spin_vector, best_energy)
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(h)
    
    # Initialize random spin state
    s = np.random.choice([-1, 1], size=N)
    
    # Compute initial energy
    def compute_energy(spins):
        energy = np.dot(h, spins)
        for i in range(N):
            for j in range(i + 1, N):
                energy += J[i, j] * spins[i] * spins[j]
        return energy
    
    current_energy = compute_energy(s)
    best_s = s.copy()
    best_energy = current_energy
    
    # Geometric cooling schedule
    for sweep in range(num_sweeps):
        # Temperature decreases geometrically
        temp = initial_temp * (final_temp / initial_temp) ** (sweep / (num_sweeps - 1))
        
        # Try to flip each spin once per sweep (random order)
        for i in np.random.permutation(N):
            # Calculate energy change if we flip spin i
            delta_e = 2 * h[i] * s[i]
            for j in range(N):
                if i != j:
                    delta_e += 2 * J[i, j] * s[i] * s[j]
            
            # Accept if energy decreases or with Boltzmann probability
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                s[i] *= -1  # Flip the spin
                current_energy += delta_e
                
                # Update best solution
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_s = s.copy()
    
    return best_s, best_energy