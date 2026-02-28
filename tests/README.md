# Verification Suite: `tests/`

This directory contains the rigorous validation logic required to certify the physical and numerical integrity of the simulation framework.

## Verification Components

### 1. Scientific Validation (`validation_scientific.py`)
**Purpose**: Proves the fundamental laws of physics are enforced.
- **Energy Conservation**: Verifies $\Delta U = Q - W$ across the entire cycle.
- **Kinematic Bounds**: Ensures piston displacement exactly matches analytical limits at TDC and BDC.
- **Heat Transfer**: Validates Woschni implementation vs expected scaling.

### 2. Logic Tests (`test_kinematics.py`, `test_thermodynamics.py`)
**Purpose**: Standard unit tests for edge-case coverage and function correctness.
- Ensures zero-volume guards work.
- Verifies valve overlap timing logic.

### 3. Property-Based Testing (`test_property_based.py`)
**Purpose**: Uses `Hypothesis` to perform randomized domain searches across thousands of engine configurations.
- Proves that no valid geometry input can lead to a crash or NaN output.

## Execution

Tests must be run from the repository root with `PYTHONPATH` set to `src`:

```bash
# Run all tests
PYTHONPATH=src pytest

# Run scientific validation alone
PYTHONPATH=src python tests/validation_scientific.py
```

## Quality Bar

A contribution is only valid if:
1.  **Pytest** results in `0 failures`.
2.  **Scientific Validation** outputs: `✅ ALL VALIDATION TESTS PASSED`.
3.  **Linter** (`ruff`) reports zero violations.
