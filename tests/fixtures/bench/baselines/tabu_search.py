import numpy as np
from typing import Tuple, Optional, List


def tabu_search(
    h: np.ndarray,
    J: np.ndarray,
    num_iterations: int = 1000,
    tabu_tenure: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Tabu Search for Ising/QUBO optimization.
    
    Args:
        h: Linear bias vector (shape: N,)
        J: Coupling matrix (shape: N x N, symmetric, zero diagonal)
        num_iterations: Number of iterations to run
        tabu_tenure: Length of tabu list (default: N/2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_spin_vector, best_energy)
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(h)
    
    if tabu_tenure is None:
        tabu_tenure = max(1, N // 2)
    
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
    
    # Tabu list: stores (position, flip_value) tuples with expiration time
    # We'll store iterations when each position was last flipped
    tabu_list = np.full(N, -1, dtype=int)  # -1 means never flipped or expired
    
    for iteration in range(num_iterations):
        # Find best non-tabu move
        best_delta_e = float('inf')
        best_pos = -1
        
        # Evaluate flipping each spin
        for i in range(N):
            # Check if move is tabu (flipped too recently)
            if tabu_list[i] > iteration - tabu_tenure:
                continue  # Skip tabu move
            
            # Calculate energy change if we flip spin i
            delta_e = 2 * h[i] * s[i]
            for j in range(N):
                if i != j:
                    delta_e += 2 * J[i, j] * s[i] * s[j]
            
            # Update best move if this is better
            if delta_e < best_delta_e:
                best_delta_e = delta_e
                best_pos = i
        
        # If no valid moves found (shouldn't happen with proper tabu tenure), 
        # just pick a random non-tabu move
        if best_pos == -1:
            # Find any non-tabu position
            non_tabu_positions = [i for i in range(N) if tabu_list[i] <= iteration - tabu_tenure]
            if non_tabu_positions:
                best_pos = np.random.choice(non_tabu_positions)
                # Recalculate delta_e for this position
                best_delta_e = 2 * h[best_pos] * s[best_pos]
                for j in range(N):
                    if best_pos != j:
                        best_delta_e += 2 * J[best_pos, j] * s[best_pos] * s[j]
            else:
                # All moves are tabu, pick the best anyway (aspiration criteria)
                best_delta_e = float('inf')
                for i in range(N):
                    delta_e = 2 * h[i] * s[i]
                    for j in range(N):
                        if i != j:
                            delta_e += 2 * J[i, j] * s[i] * s[j]
                    if delta_e < best_delta_e:
                        best_delta_e = delta_e
                        best_pos = i
        
        # Apply the best move
        if best_pos != -1:
            # Flip the spin
            s[best_pos] *= -1
            current_energy += best_delta_e
            
            # Update tabu list
            tabu_list[best_pos] = iteration
            
            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_s = s.copy()
    
    return best_s, best_energy