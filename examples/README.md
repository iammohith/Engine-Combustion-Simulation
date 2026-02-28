# Demonstrations: `examples/`

This directory provides executable scripts demonstrating the practical application of the simulation framework.

## Available Examples

### `example_script.py`
The primary entry point for exploring simulation capabilities. It contains four distinct studies:

1.  **Single Cylinder Baseline**: Standard 4-stroke cycle metrics.
2.  **Wiebe Combustion Analysis**: Impact of burn duration and phase on peak pressure.
3.  **Inline-4 Multi-Cylinder**: Torque and power delivery of a 4-cylinder assembly.
4.  **Parametric Study**: Sweep of compression ratios to identify the power-efficiency frontier.

## Execution

Run all examples from the repository root:

```bash
PYTHONPATH=src python examples/example_script.py
```

## Success Indicators

- **Console Stdout**: Tables summarizing power, torque, and efficiency.
- **Filesystem**: Generation of analysis plots (`.png`) and data summaries (`.csv`, `.json`).

## Requirements

Visualization examples require `matplotlib`. On headless systems, ensure the backend is configured or use the CLI commands provided in the root `README.md`.
