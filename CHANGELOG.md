# Changelog

All notable changes to `pqblocks-qiea` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- Quantum-Inspired Evolutionary Algorithm (QIEA) solver for Ising and QUBO combinatorial problems.
- Continuous Q-bit probability amplitude representations ($[\alpha_i, \beta_i]^T$).
- Dynamic quantum rotation gate steering mechanism with sign lookup tables.
- High-performance Numba JIT-compiled inner evolutionary loops with pure-Python reference fallback.
- Auto-scaling population heuristics and configurable rotation angle step sizes.
- Reference instance verification suite with planted and MQLib benchmark models.
- Comprehensive test suite with 72 tests passing across Python 3.11, 3.12, 3.13, and 3.14.
- Quickstart and quantum rotation angle parameter tuning examples.
