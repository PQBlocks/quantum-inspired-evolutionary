# pqblocks-qiea

> Quantum-Inspired Evolutionary Algorithm (QIEA) with Q-bit representations and quantum rotation gates for Ising and QUBO optimization problems.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## What It Is

`pqblocks-qiea` implements the Quantum-Inspired Evolutionary Algorithm (QIEA) for combinatorial optimization on Ising spin systems and Quadratic Unconstrained Binary Optimization (QUBO) problems.

In QIEA, individuals in the population are represented as strings of **Q-bits** (quantum-inspired probabilistic bits) characterized by probability amplitudes $[\alpha_i, \beta_i]^T$ satisfying $|\alpha_i|^2 + |\beta_i|^2 = 1$. The state of a Q-bit represents the probability of collapsing to spin $+1$ or $-1$.

Evolution is driven by **quantum rotation gates** $U(\Delta \theta)$, which rotate individual Q-bit state vectors toward candidate best configurations found by the population. This mechanism enables a balance between exploratory global search (via superposition-like probability amplitudes) and exploitative local refinement (via targeted angular rotation).

---

## Key Features

- **Probabilistic Q-Bit Ensembles**: Each individual maintains $N$ Q-bit amplitudes $[\alpha_i, \beta_i]^T$ initialized in balanced superposition ($\alpha_i = \beta_i = 1/\sqrt{2}$).
- **Quantum Rotation Gate Operators**: Amplitudes are steered toward the best individual using rotation gates $U(\theta)$ with lookup-table rotation signs.
- **Auto-Derived Hyperparameters**: Population size auto-scales as $\max(10, \min(100, 2N))$, with default base rotation step $\theta_0 = 0.05\pi$.
- **High-Performance Numba JIT Kernel**: Fast inner evolutionary loop with pure-Python reference fallback.
- **Reference Instance Suite**: Includes planted golden instances with mathematically proven ground states for verification and benchmarking.

---

## Installation

```bash
pip install pqblocks-qiea
```

---

## Quickstart

```python
import numpy as np
from pqblocks_qiea import solve_ising

# 4-spin antiferromagnetic ring
N = 4
h = np.zeros(N)
J = np.zeros((N, N))
for i in range(N):
    j = (i + 1) % N
    J[i, j] = J[j, i] = 1.0

# Solve using QIEA
best_spins, best_energy = solve_ising(
    h=h,
    J=J,
    num_generations=100,
    seed=42
)

print("Best Spin State:", best_spins.tolist())
print(f"Best Energy: {best_energy:.6f}")
```

Execution output:

```text
Best Spin State: [1, -1, 1, -1]
Best Energy: -4.000000
```

---

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md) — Empirical performance comparisons, TTS evaluations, and winning/losing cases
- [TESTING.md](TESTING.md) — Test suite layout, coverage details, and validation procedures
- [EVIDENCE.md](EVIDENCE.md) — Clean-room execution logs and reproducibility artifacts
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) — Formal release verification gates and checks
- [USE_CASES.md](USE_CASES.md) — Q-bit rotation tuning and QUBO integration patterns
- [SECURITY_NOTES.md](SECURITY_NOTES.md) — Numerical safeguards, non-cryptographic scope, and stability bounds
- [AGENTS.md](AGENTS.md) — Machine-readable interfaces and guidelines for autonomous agents
- [CHANGELOG.md](CHANGELOG.md) — Version history and release notes
