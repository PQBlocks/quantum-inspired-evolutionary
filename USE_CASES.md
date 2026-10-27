# Use Cases & Integration Patterns

> Practical applications and integration scenarios for `pqblocks-qiea`.

---

## 1. Exploring Rugged Solution Landscapes with Q-Bits

Classical genetic algorithms maintain discrete bitstrings $x \in \{0, 1\}^N$, requiring explicit mutation operators to escape local optima. In contrast, Quantum-Inspired Evolutionary Algorithms represent each variable via a continuous probability amplitude pair $[\alpha_i, \beta_i]^T$, enabling the population to implicitly represent all $2^N$ states simultaneously in superposition.

```python
import numpy as np
from pqblocks_qiea import solve_ising

# Solve dense Ising model
best_spins, best_energy = solve_ising(
    h=h,
    J=J,
    num_generations=200,
    population_size=50,
    seed=42
)

print(f"Optimal ground state energy: {best_energy:.4f}")
print(f"Spin configuration: {best_spins}")
```

---

## 2. Dynamic Rotation Gate Tuning ($\theta_0$)

The base rotation angle $\theta_0$ controls the trade-off between exploration and exploitation:
- **Small $\theta_0$ ($\approx 0.01\pi$)**: Smooth, gradual convergence suitable for fine-grained local refinement.
- **Large $\theta_0$ ($\approx 0.10\pi$)**: Fast, aggressive steering toward discovered candidates, suitable for rapid convergence on low-dimensional models.

```python
# Fine-grained local refinement
best_spins, best_energy = solve_ising(
    h=h,
    J=J,
    num_generations=300,
    theta_0=0.01 * np.pi,
    seed=42
)
```

---

## 3. Lossless Integration with QUBO Formulations

Binary quadratic problems ($x_i \in \{0, 1\}$) are easily transformed to Ising spin models ($s_i \in \{-1, +1\}$) via `pqblocks-qubo`:

```python
from pqblocks_qubo import qubo_to_ising
from pqblocks_qiea import solve_ising

# Convert QUBO to Ising
J, h, offset = qubo_to_ising(Q)

# Run QIEA
spins, ising_energy = solve_ising(h, J, num_generations=150, seed=42)

# Recover QUBO binary solution
binary_solution = (spins + 1) // 2
qubo_energy = ising_energy + offset
```
