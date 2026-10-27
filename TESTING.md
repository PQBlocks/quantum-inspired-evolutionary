# Testing & Test Suite Documentation

> Test suite coverage, execution procedures, and verification breakdown for `pqblocks-qiea`.

---

## 1. Test Suite Summary

The test suite contains **72 comprehensive tests** covering input validation, Q-bit probability normalization, quantum rotation gate dynamics, Kolmogorov-Smirnov statistical equivalence between Numba and reference kernels, golden instance verification, and benchmark harness integrations.

- **Status**: 72 passed, 0 failed, 1 warning (runtime informational warning on asymptotic KS test)
- **Runtime**: ~29.0 seconds
- **Statement Coverage**: 73% across 393 statements
- **Python Matrix**: Fully verified on Python 3.11, 3.12, 3.13, and 3.14

---

## 2. Test File Breakdown

| Test File | Tests | Focus Area |
|---|---|---|
| `test_api.py` | 13 | Public API input validation: empty models, dimension mismatches, asymmetric coupling, non-zero diagonals, NaN/Inf bounds, parameter ranges |
| `test_benchmark.py` | 6 | TTS computation engine, solution quality measurement, statistical distribution checks |
| `test_golden.py` | 6 | Golden instance verification, schema validation, analytical brute-force comparisons |
| `test_golden_instances.py` | 21 | Independent first-principles energy oracles, brute-force search correctness, instance loading, tolerance checks |
| `test_kernel.py` | 10 | Numba JIT kernel verification, rotation sign agreement with specification, KS distribution equivalence, edge cases |
| `test_reference.py` | 4 | Pure-Python reference kernel, 4-spin worked example exact matches, amplitude normalization invariance ($|\alpha|^2 + |\beta|^2 = 1$) |
| `test_verification.py` | 12 | Scaled planted instances ($N=4, 6, 8, 10$), MQLib Rudy graph instances, seed determinism, convergence consistency |

---

## 3. Running the Tests

To run the complete test suite:

```bash
PYTHONPATH=src pytest -v
```

To run with coverage reporting:

```bash
PYTHONPATH=src pytest --cov=pqblocks_qiea --cov-report=term
```
