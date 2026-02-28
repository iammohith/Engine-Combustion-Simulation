# Quick Start Guide

Follow these steps to verify and execute the Engine Simulation Framework locally in exactly three minutes.

## 1. Environment Setup

The framework is a standard Python package. It is recommended to use a virtual environment to isolate dependencies.

```bash
# Initialize environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install package with development dependencies
pip install -e ".[dev]"
```

## 2. Mandatory Verification

Before running simulations, prove the local environment correctly computes the analytical physical invariants.

```bash
# Run the scientific certification suite
PYTHONPATH=src python tests/validation_scientific.py

# Run the logic and property-based test suite
PYTHONPATH=src pytest
```

**Success Indicators**:
- Validation: `✅ ALL VALIDATION TESTS PASSED`
- Pytest: `94 passed` (Approximate; varies by suite updates)

## 3. Running Your First Simulation

### Option A: The Package CLI (Recommended)
Use the built-in CLI to run a standard V8 engine preset at 4500 RPM.
```bash
python -m engine_simulator --preset v8 --rpm 4500
```
This will print a summary performance table and export two files:
1.  `multi_cylinder_data.csv`: High-resolution time-series data.
2.  `multi_cylinder_metrics.json`: Aggregated performance constants.

### Option B: The Example Suite
Execute comprehensive demonstrations including parametric studies and combustion analysis.
```bash
PYTHONPATH=src python examples/example_script.py
```

## 4. Basic API Usage

Integrate the simulator into your own research scripts:

```python
from engine_simulator.engine_config import create_default_v8
from engine_simulator.single_cylinder_simulator import SingleCylinderSimulator

# 1. Define configuration
config = create_default_v8()

# 2. Initialize and run simulation
sim = SingleCylinderSimulator(config)
results = sim.simulate_cycle()

# 3. Access verified physics
print(f"Indicated Thermal Efficiency: {results.thermal_efficiency:.2%}")
```

## 5. Troubleshooting Common Failures

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| `ImportError` | Standard `pip install` skipped. | Run `pip install -e .` from the root. |
| `AssertionError` | Physics bounds violation. | Ensure geometry values are physically possible. |
| `ModuleNotFoundError` | `PYTHONPATH` not set for manual scripts. | Use `PYTHONPATH=src python <path>` or install the package. |
| No Plots Displayed | Running on headless server. | Check `examples/README.md` for `Agg` renderer notes. |
