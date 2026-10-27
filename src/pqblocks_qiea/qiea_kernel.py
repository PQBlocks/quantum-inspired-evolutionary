import numpy as np
from numba import njit
from typing import Optional, Tuple

@njit(fastmath=False)
def get_rotation_sign_numba(x_j: int, b_j: int, f_ge_fb: bool, alpha: float, beta: float) -> int:
    """
    Numba-compiled version of get_rotation_sign.
    Returns the sign s (+1, -1, 0) for the rotation gate based on the QIEA lookup table.
    """
    # Cases where no rotation is needed
    if x_j == 0 and b_j == 0:
        return 0
    if x_j == 1 and b_j == 1:
        return 0
    
    ab = alpha * beta
    ab_gt_0 = ab > 0
    ab_lt_0 = ab < 0
    a_eq_0 = alpha == 0.0
    b_eq_0 = beta == 0.0
    
    # Case: x_j = 0, b_j = 1
    if x_j == 0 and b_j == 1:
        if not f_ge_fb:  # f(x) < f(b)
            if ab_gt_0:
                return 1
            if ab_lt_0:
                return -1
            if a_eq_0:
                return 0
            if b_eq_0:
                return 1
        else:  # f(x) >= f(b)
            if ab_gt_0:
                return -1
            if ab_lt_0:
                return 1
            if a_eq_0:
                return 1
            if b_eq_0:
                return 0
    
    # Case: x_j = 1, b_j = 0
    if x_j == 1 and b_j == 0:
        if not f_ge_fb:  # f(x) < f(b)
            if ab_gt_0:
                return -1
            if ab_lt_0:
                return 1
            if a_eq_0:
                return 1
            if b_eq_0:
                return 0
        else:  # f(x) >= f(b)
            if ab_gt_0:
                return 1
            if ab_lt_0:
                return -1
            if a_eq_0:
                return 0
            if b_eq_0:
                return 1
    
    return 0

@njit(fastmath=False)
def _compute_spin_energy_numba(x: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """
    Numba-compiled function to compute the Ising energy of a binary state.
    """
    # Map binary to spin
    s = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        s[i] = 2.0 * x[i] - 1.0
    
    # Linear term
    energy = 0.0
    for i in range(h.size):
        energy += h[i] * s[i]
    
    # Quadratic term (upper triangular only to avoid double counting)
    N = h.size
    for i in range(N):
        for j in range(i + 1, N):
            energy += J[i, j] * s[i] * s[j]
    
    return energy

@njit(fastmath=False)
def _qiea_kernel(
    h: np.ndarray,
    J: np.ndarray,
    num_generations: int,
    pop_size: int,
    theta_0: float,
    seed: Optional[int]
) -> Tuple[np.ndarray, float]:
    """
    Numba-optimized kernel for Quantum-Inspired Evolutionary Algorithm.
    
    Args:
        h: Linear bias vector (shape: N,)
        J: Coupling matrix (shape: N x N, symmetric, zero diagonal)
        num_generations: Number of generations to run
        pop_size: Population size (number of Q-bit individuals)
        theta_0: Base rotation step size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (best_spin_vector, best_energy)
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = h.shape[0]
    
    # Initialize Q-bit amplitudes to uniform superposition
    # alpha = 1/sqrt(2), beta = 1/sqrt(2)
    init_amplitude = 1.0 / np.sqrt(2.0)
    alphas = np.full((pop_size, N), init_amplitude, dtype=np.float64)
    betas = np.full((pop_size, N), init_amplitude, dtype=np.float64)
    
    # Track global best
    global_best_x = np.zeros(N, dtype=np.int8)
    global_best_fitness = -np.inf
    
    for gen in range(num_generations):
        # Observation step: sample binary states from Q-bit probabilities
        # x_j = 1 if r_j < |beta_j|^2, else 0
        rand_vals = np.random.random((pop_size, N))
        x_pop = (rand_vals < (betas * betas)).astype(np.int8)
        
        # Fitness evaluation for each individual
        fitnesses = np.zeros(pop_size, dtype=np.float64)
        for i in range(pop_size):
            # Compute Ising energy (we minimize E, so fitness = -E)
            energy = _compute_spin_energy_numba(x_pop[i], h, J)
            fitnesses[i] = -energy  # fitness = -E, we maximize fitness
            
            # Update global best if this is better
            if fitnesses[i] > global_best_fitness:
                global_best_fitness = fitnesses[i]
                global_best_x = x_pop[i].copy()
        
        # Rotation gate update for each individual
        for i in range(pop_size):
            f_x = fitnesses[i]
            f_ge_fb = f_x >= global_best_fitness
            
            for j in range(N):
                s_sign = get_rotation_sign_numba(
                    x_pop[i, j],
                    global_best_x[j],
                    f_ge_fb,
                    alphas[i, j],
                    betas[i, j]
                )
                
                d_theta = s_sign * theta_0
                
                # Apply rotation if needed
                if d_theta != 0.0:
                    a = alphas[i, j]
                    b = betas[i, j]
                    alphas[i, j] = a * np.cos(d_theta) - b * np.sin(d_theta)
                    betas[i, j] = a * np.sin(d_theta) + b * np.cos(d_theta)
    
    # Return best solution found
    best_s = np.empty(N, dtype=np.int8)
    for i in range(N):
        best_s[i] = 2 * global_best_x[i] - 1  # Map to Ising spins
    best_energy = -global_best_fitness  # Convert back to energy
    
    return best_s, best_energy