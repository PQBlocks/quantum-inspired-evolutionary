# Security Notes & Posture

> Security scope, numerical safeguards, and execution boundaries for `pqblocks-qiea`.

---

## 1. Non-Cryptographic Classification

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

`pqblocks-qiea` is a classical evolutionary heuristic optimization solver. It does not implement, expose, or validate cryptographic keys, digital signatures, hash algorithms, or post-quantum cryptographic primitives.

The term "Quantum-Inspired" refers strictly to the algorithmic use of 2D rotation matrices and probability amplitudes analogous to qubit state vectors.

---

## 2. Numerical Safeguards & Trigonometric Stability

- **Amplitude Normalization Safeguards**:
  Following each quantum rotation step:
  $$\begin{bmatrix} \alpha' \\ \beta' \end{bmatrix} = \begin{bmatrix} \cos(\Delta \theta) & -\sin(\Delta \theta) \\ \sin(\Delta \theta) & \cos(\Delta \theta) \end{bmatrix} \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$$
  accumulated floating-point rounding errors are corrected by periodic re-normalization:
  $$\alpha_{\text{norm}} = \frac{\alpha'}{\sqrt{|\alpha'|^2 + |\beta'|^2}}, \quad \beta_{\text{norm}} = \frac{\beta'}{\sqrt{|\alpha'|^2 + |\beta'|^2}}$$
  preventing probability drift away from unity.
- **Angle Domain Bounding**:
  Base rotation step sizes $\theta_0$ are constrained to the interval $(0, \pi/2]$. Out-of-range values or non-positive parameters trigger immediate `ValueError` exceptions.
- **Input Validation**:
  Matrices and field vectors are validated for finiteness (rejecting NaN, positive and negative infinity), symmetry ($J = J^T$), and zero diagonal entries ($J_{ii} = 0$).
