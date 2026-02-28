# Source Package: `engine_simulator`

This directory contains the core physical modeling logic for the Engine Combustion Simulation Framework.

## Component Overview

| Module | Role | Key Abstractions |
| :--- | :--- | :--- |
| `engine_config.py` | Schema Definition | `EngineConfiguration`, `EngineType` |
| `kinematics.py` | Analytical Geometry | `SliderCrank`, `ValveTiming` |
| `thermodynamics.py` | Physics Invariants | `EngineCycle`, `ThermodynamicState` |
| `single_cylinder_simulator.py` | Integration Loop | `SingleCylinderSimulator` |
| `multi_cylinder_simulator.py` | Engine Assembly | `MultiCylinderSimulator` |
| `utilities.py` | Data IO | `DataExporter` |
| `visualization.py` | Plotting Library | `EnginePlotter` |

## Design Principles

1.  **Immutability**: Data structures representing physical states (e.g., `CycleResults`) use frozen dataclasses to prevent side effects during multi-cylinder aggregation.
2.  **Vectorization**: Cycle simulation data is stored in NumPy arrays. Multi-cylinder analysis avoids $O(N)$ Python loops by using `np.roll` and `np.interp` for signal phasing.
3.  **Functional Purity**: Core physics modules (`kinematics`, `thermodynamics`) are stateless and deterministic; they compute outputs purely from function arguments.

## How to Initialize

The package should be interacted with via the `EngineConfiguration` object:

```python
from engine_simulator.engine_config import create_default_inline_4
from engine_simulator.single_cylinder_simulator import SingleCylinderSimulator

config = create_default_inline_4()
sim = SingleCylinderSimulator(config)
results = sim.simulate_cycle()
```

## Failure Modes

- `ValueError`: Raised during configuration if geometry violates physical bounds (e.g., negative clearance volume).
- `AssertionError`: Raised if numerical convergence fails or energy conservation limits are exceeded during the integration loop.
