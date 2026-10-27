# Autonomous Agent Integration Guide

> Operational instructions and best practices for AI and autonomous software agents integrating `pqblocks-qiea`.

---

## 1. Package Purpose

`pqblocks-qiea` provides a Quantum-Inspired Evolutionary Algorithm heuristic solver for Ising spin-glass and QUBO combinatorial optimization.

---

## 2. Key APIs

```python
from pqblocks_qiea import solve_ising

# Functional solver invocation
best_spins, best_energy = solve_ising(
    h=h_vector,                 # array-like: (N,) linear bias terms
    J=J_matrix,                 # array-like: (N, N) symmetric coupling matrix (zero diagonal)
    num_generations=100,        # int: evolutionary generations (default: 100)
    population_size=None,       # int, optional: auto-derived as max(10, min(100, 2*N)) if None
    theta_0=0.05 * np.pi,       # float: base rotation gate angle (default: 0.05 * pi)
    seed=42                     # int, optional: PRNG seed for deterministic runs
)
```

---

## 3. Best Practices for Agents

1. **Auto Population Sizing**: Allow `population_size=None` unless explicitly tuning memory or runtime constraints; the auto-heuristic `max(10, min(100, 2*N))` provides solid convergence across standard topologies.
2. **Determinism**: Always specify `seed` when verifying solution reproducibility.
3. **Spin Conventions**: Returns spin values $s_i \in \{-1, +1\}$. When working with binary $x_i \in \{0, 1\}$, use $(s_i + 1)//2$.
