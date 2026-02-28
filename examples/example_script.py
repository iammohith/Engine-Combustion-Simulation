"""
Basic Engine Simulation Example
Demonstrates simple usage of the simulation framework.

Author: Mohith Sai Gorla
Date: 27-02-2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Headless environment check for observability stability
if "DISPLAY" not in os.environ and os.name != "nt":
    import matplotlib
    matplotlib.use("Agg")
    print("Physical display not detected. Using 'Agg' backend for plot exports.")

from engine_simulator.engine_config import (
    EngineConfiguration,
    GeometryParameters,
    ThermodynamicParameters,
    CombustionParameters,
    MultiCylinderParameters,
    OperatingConditions,
    SimulationParameters,
    CombustionType,
    EngineType,
)
from engine_simulator.single_cylinder_simulator import SingleCylinderSimulator
from engine_simulator.multi_cylinder_simulator import MultiCylinderSimulator
from engine_simulator.visualization import EnginePlotter
from engine_simulator.utilities import DataExporter


def example_1_single_cylinder():
    """Example 1: Single cylinder engine simulation"""

    print("=" * 70)
    print("EXAMPLE 1: Single Cylinder Engine Simulation")
    print("=" * 70)
    print()

    # Create custom engine configuration
    config = EngineConfiguration(
        geometry=GeometryParameters(
            bore=0.082,  # 82mm
            stroke=0.090,  # 90mm
            connecting_rod_length=0.150,  # 150mm
            compression_ratio=10.5,
            num_cylinders=1,
        ),
        thermodynamics=ThermodynamicParameters(
            intake_pressure=101325.0,  # 1 atm
            intake_temperature=298.15,  # 25°C
            peak_combustion_temperature=2500.0,  # K
        ),
        combustion=CombustionParameters(
            combustion_type=CombustionType.IDEAL_CONSTANT_VOLUME
        ),
        multi_cylinder=MultiCylinderParameters(),
        operating=OperatingConditions(rpm=3000.0),
        simulation=SimulationParameters(angular_resolution=1.0),
    )

    # Create simulator and run
    simulator = SingleCylinderSimulator(config)
    results = simulator.simulate_cycle()

    # Calculate metrics
    metrics = simulator.calculate_performance_metrics(results)

    # Display results
    print("\nPerformance Metrics:")
    print(
        f"  Indicated Power:    {metrics['indicated_power_kw']:.2f} kW ({metrics['indicated_power_hp']:.2f} HP)"
    )
    print(f"  Thermal Efficiency: {metrics['thermal_efficiency']*100:.2f}%")
    print(f"  IMEP:              {metrics['imep_bar']:.2f} bar")
    print(f"  Peak Pressure:     {metrics['peak_pressure_mpa']:.2f} MPa")
    print(f"  Peak Temperature:  {metrics['peak_temperature_k']:.0f} K")
    print()

    # Create plots
    plotter = EnginePlotter()
    plotter.plot_pv_diagram(results, save_path="./example1_pv_diagram.png")
    plotter.plot_pressure_angle(results, save_path="./example1_pressure.png")

    # Export data
    DataExporter.export_to_csv(results, "./example1_cycle_data.csv")

    return results, metrics


def example_2_wiebe_combustion():
    """Example 2: Realistic combustion with Wiebe function"""

    print("=" * 70)
    print("EXAMPLE 2: Wiebe Combustion Model")
    print("=" * 70)
    print()

    # Configuration with Wiebe combustion
    config = EngineConfiguration(
        geometry=GeometryParameters(
            bore=0.086,  # 86mm
            stroke=0.086,  # 86mm (square engine)
            connecting_rod_length=0.143,
            compression_ratio=11.0,
            num_cylinders=1,
        ),
        thermodynamics=ThermodynamicParameters(peak_combustion_temperature=2700.0),
        combustion=CombustionParameters(
            combustion_type=CombustionType.WIEBE,
            wiebe_a=5.0,
            wiebe_m=2.0,
            combustion_start_btdc=20.0,  # Ignition 20° BTDC
            combustion_duration=50.0,  # 50° burn duration
        ),
        multi_cylinder=MultiCylinderParameters(),
        operating=OperatingConditions(rpm=5000.0),
        simulation=SimulationParameters(angular_resolution=0.5),  # Higher resolution
    )

    simulator = SingleCylinderSimulator(config)
    results = simulator.simulate_cycle()
    metrics = simulator.calculate_performance_metrics(results)

    print("\nWiebe Combustion Results:")
    print(f"  Power:             {metrics['indicated_power_kw']:.2f} kW")
    print(f"  Efficiency:        {metrics['thermal_efficiency']*100:.2f}%")
    print(f"  IMEP:              {metrics['imep_bar']:.2f} bar")
    print()

    # Comprehensive plot
    plotter = EnginePlotter()
    plotter.plot_comprehensive_analysis(
        results, save_path="./example2_comprehensive.png"
    )

    return results, metrics


def example_3_inline_4_engine():
    """Example 3: Inline-4 multi-cylinder engine"""

    print("=" * 70)
    print("EXAMPLE 3: Inline-4 Multi-Cylinder Engine")
    print("=" * 70)
    print()

    # Inline-4 configuration
    config = EngineConfiguration(
        geometry=GeometryParameters(
            bore=0.082,
            stroke=0.090,
            connecting_rod_length=0.150,
            compression_ratio=10.5,
            num_cylinders=4,
        ),
        thermodynamics=ThermodynamicParameters(),
        combustion=CombustionParameters(),
        multi_cylinder=MultiCylinderParameters(
            engine_type=EngineType.INLINE,
            firing_order=[1, 3, 4, 2],
            crank_phases=[0.0, 180.0, 180.0, 0.0],
        ),
        operating=OperatingConditions(rpm=4000.0),
        simulation=SimulationParameters(angular_resolution=1.0),
    )

    simulator = MultiCylinderSimulator(config)
    results = simulator.simulate_engine()
    balance = simulator.analyze_balance(results)

    print("\nMulti-Cylinder Results:")
    print(f"  Total Power:       {results.total_indicated_power_kw:.2f} kW")
    print(f"  Mean Torque:       {results.mean_torque:.2f} N·m")
    print(f"  Torque Variation:  {results.torque_variation:.2f}%")
    print(f"  Smoothness Factor: {results.smoothness_factor:.3f}")
    print()

    print("Force Balance:")
    print(f"  Primary Balanced:  {'Yes' if balance['is_primary_balanced'] else 'No'}")
    print(
        f"  Secondary Balanced: {'Yes' if balance['is_secondary_balanced'] else 'No'}"
    )
    print(f"  Balance Quality:   {balance['balance_quality_percent']:.2f}%")
    print()

    # Plot multi-cylinder torque
    plotter = EnginePlotter()
    plotter.plot_multi_cylinder_torque(results, save_path="./example3_torque.png")

    return results, balance


def example_4_parametric_study():
    """Example 4: Parametric study of compression ratio"""

    print("=" * 70)
    print("EXAMPLE 4: Parametric Study - Compression Ratio Effect")
    print("=" * 70)
    print()

    compression_ratios = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    efficiencies = []
    powers = []

    for cr in compression_ratios:
        config = EngineConfiguration(
            geometry=GeometryParameters(
                bore=0.086,
                stroke=0.086,
                connecting_rod_length=0.143,
                compression_ratio=cr,
                num_cylinders=1,
            ),
            thermodynamics=ThermodynamicParameters(),
            combustion=CombustionParameters(),
            multi_cylinder=MultiCylinderParameters(),
            operating=OperatingConditions(rpm=3000.0),
            simulation=SimulationParameters(angular_resolution=2.0),  # Faster
        )

        simulator = SingleCylinderSimulator(config)
        results = simulator.simulate_cycle()
        metrics = simulator.calculate_performance_metrics(results)

        efficiencies.append(metrics["thermal_efficiency"] * 100)
        powers.append(metrics["indicated_power_kw"])

        print(f"  CR {cr:.1f}: η={efficiencies[-1]:.2f}%, P={powers[-1]:.2f}kW")

    print()

    # Plot results
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(compression_ratios, efficiencies, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Compression Ratio", fontweight="bold")
    ax1.set_ylabel("Thermal Efficiency (%)", fontweight="bold")
    ax1.set_title("Efficiency vs Compression Ratio", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(compression_ratios, powers, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Compression Ratio", fontweight="bold")
    ax2.set_ylabel("Indicated Power (kW)", fontweight="bold")
    ax2.set_title("Power vs Compression Ratio", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./example4_parametric.png", dpi=300)
    plt.show()

    print("Parametric study complete!")


def main():
    """Run all examples"""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ENGINE SIMULATION EXAMPLES" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    # Run examples
    example_1_single_cylinder()
    print("\n" + "─" * 70 + "\n")

    example_2_wiebe_combustion()
    print("\n" + "─" * 70 + "\n")

    example_3_inline_4_engine()
    print("\n" + "─" * 70 + "\n")

    example_4_parametric_study()

    print("\n" + "═" * 70)
    print("All examples completed successfully!")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
