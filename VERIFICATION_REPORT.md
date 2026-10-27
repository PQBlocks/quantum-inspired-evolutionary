# Verification Report: pqblocks-qiea

> Formal release verification report for `pqblocks-qiea`.

---

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Product** | `pqblocks-qiea` |
| **Package Name** | `pqblocks_qiea` |
| **Version** | `0.1.0` |
| **Classification** | Quantum-inspired classical optimisation: not cryptography, no security claim |
| **Verification Status** | **PASSED** |
| **Test Matrix** | Python 3.11, 3.12, 3.13, 3.14 |
| **Test Results** | 72 passed, 0 failed across all supported versions |
| **Leak Gate** | 0 hits |

---

## 2. Verification Gate Results

| Check ID | Verification Item | Target | Result | Status |
|---|---|---|---|---|
| **CHK-01** | Test Suite Completeness | 72 passing tests | 72 passed, 0 failed | **PASS** |
| **CHK-02** | Python Compatibility | Python 3.11–3.14 | Clean execution on all 4 versions | **PASS** |
| **CHK-03** | Source & Wheel Build | PEP 517 / PEP 621 | Valid sdist and wheel generated | **PASS** |
| **CHK-04** | Package Integrity | `twine check --strict` | Passed without warnings | **PASS** |
| **CHK-05** | Public Leak Gate | 0 internal leaks | 0 hits across all files | **PASS** |
| **CHK-06** | Example Verification | $\ge 2$ functioning examples | `quickstart.py` & `02_q_bit_rotation_optimization.py` passed | **PASS** |
| **CHK-07** | Statistical Equivalence | Two-tier KS test | Passed with $p > 0.05$ | **PASS** |

---

## 3. Mathematical & Algorithmic Verification

- **Q-Bit Unit Normalization**: Every individual satisfies $|\alpha_j|^2 + |\beta_j|^2 = 1$ strictly across all evolutionary generations.
- **Quantum Rotation Gate Mapping**:
  $$\begin{bmatrix} \alpha_j' \\ \beta_j' \end{bmatrix} = \begin{bmatrix} \cos(\Delta \theta_j) & -\sin(\Delta \theta_j) \\ \sin(\Delta \theta_j) & \cos(\Delta \theta_j) \end{bmatrix} \begin{bmatrix} \alpha_j \\ \beta_j \end{bmatrix}$$
  where $\Delta \theta_j = s(\alpha_j, \beta_j) \cdot \theta_0$ with rotation signs chosen according to the standard QIEA lookup table.
- **Statistical Equivalence**: Verified via two-sample Kolmogorov-Smirnov test comparing the Numba JIT-compiled inner loop against pure-Python reference calculations.

---

## 4. Conclusion

The package `pqblocks-qiea` has satisfied all release verification requirements and is verified for release.
