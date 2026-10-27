# Verification Evidence & Traceability

> Detailed execution evidence, command outputs, and reproducibility artifacts for `pqblocks-qiea`.

---

## 1. Clean-Room Test Execution Evidence

Full test execution output in a clean virtual environment running Python 3.14.8:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /tmp/pqblocks-gate-qiea/tests
configfile: pyproject.toml
testpaths: tests
plugins: hypothesis-6.168.0, cov-7.1.0, anyio-4.15.1
collected 72 items

tests/test_api.py .............                                          [ 18%]
tests/test_benchmark.py ......                                           [ 26%]
tests/test_golden.py ......                                              [ 34%]
tests/test_golden_instances.py .....................                     [ 63%]
tests/test_kernel.py ..........                                          [ 77%]
tests/test_reference.py ....                                             [ 83%]
tests/test_verification.py ............                                  [100%]

======================== 72 passed, 1 warning in 29.65s ========================
```

---

## 2. Example Execution Evidence

### Quickstart Execution:

```bash
$ python3 examples/quickstart.py
Best Spin State: [1, -1, 1, -1]
Best Energy: -4.000000
```

### Rotation Parameter Tuning Evaluation:

```bash
$ python3 examples/02_q_bit_rotation_optimization.py
--- QIEA Quantum Rotation Gate Parameter Tuning ---
theta_0 = 0.01*pi: Best Energy = -7.584828, Solution = [-1, -1, -1, -1, -1, 1, -1, -1]
theta_0 = 0.05*pi: Best Energy = -7.584828, Solution = [-1, -1, -1, -1, -1, 1, -1, -1]
theta_0 = 0.10*pi: Best Energy = -7.584828, Solution = [-1, -1, -1, -1, -1, 1, -1, -1]

QIEA quantum rotation optimization completed successfully.
```
