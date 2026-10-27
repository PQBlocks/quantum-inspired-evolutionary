"""
Example 02: Quantum-Inspired Evolutionary Algorithm Parameter Tuning.

Demonstrates the effect of the quantum rotation gate angle (theta_0)
and generation scaling on convergence across a frustrated spin system.
"""

import numpy as np
from pqblocks_qiea import solve_ising


def main():
    print("--- QIEA Quantum Rotation Gate Parameter Tuning ---")

    # Define 8-spin frustrated model
    N = 8
    rng = np.random.RandomState(42)
    h = rng.uniform(-0.5, 0.5, size=N)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = rng.uniform(-1.0, 1.0)

    for theta_mult in [0.01, 0.05, 0.10]:
        theta_0 = theta_mult * np.pi
        best_spins, best_energy = solve_ising(
            h=h,
            J=J,
            num_generations=150,
            population_size=30,
            theta_0=theta_0,
            seed=42
        )
        print(f"theta_0 = {theta_mult:.2f}*pi: Best Energy = {best_energy:.6f}, Solution = {best_spins.tolist()}")

    print("\nQIEA quantum rotation optimization completed successfully.")


if __name__ == "__main__":
    main()
