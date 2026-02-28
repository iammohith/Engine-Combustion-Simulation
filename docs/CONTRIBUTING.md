# Contributing Guidelines

This document defines the strict workflow required to maintain the physical and mathematical integrity of the framework.

## 1. Verification Protocol (Mandatory)

Before submitting any code, you MUST prove it does not break the simulation invariants.

### Command A: Scientific Integrity
```bash
PYTHONPATH=src python tests/validation_scientific.py
```
**Required Result**: `✅ ALL VALIDATION TESTS PASSED`. This verifies that energy conservation and kinematic limits are maintained.

### Command B: Logic & Boundaries
```bash
PYTHONPATH=src pytest
```
**Required Result**: `0 failures`. This executes the 90+ test suite, including randomized property-based searches.

## 2. Engineering Standards

1.  **Functional Purity**: Core physics functions in `kinematics` and `thermodynamics` must remain stateless. They take variables and return values; they do not access global state.
2.  **Immutability**: All output dataclasses (like `CycleResults`) must be decorated with `frozen=True`.
3.  **Performance**: Do not use Python `for` loops for time-series data or multi-cylinder summation. Use NumPy vectorized operations exclusively.
4.  **Numerical Hygiene**: Avoid floating-point cancellation. For example, use `math.expm1(x)` instead of `math.exp(x) - 1`.

## 3. Documentation Requirements

- **No Intent**: Only document what the code is currently doing.
- **No Aspiration**: Do not list "planned" features.
- **Verified Files**: Every file described in documentation must exist in the repository.

## 4. Pull Request Process

1.  Fork the repo and create your feature branch.
2.  Ensure linter (`ruff`) and formatter (`black`) pass.
3.  Execute the **Verification Protocol** (Section 1).
4.  Documentation: If adding a feature, update the relevant component description and `src/README.md`.

---

*Violations of physical laws or numerical hygiene will result in immediate PR rejection.*
