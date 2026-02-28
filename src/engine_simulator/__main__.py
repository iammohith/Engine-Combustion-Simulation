"""
CLI entry point for engine_simulator.
"""
import argparse
import sys
from .engine_config import create_default_inline_4, create_default_v8
from .single_cylinder_simulator import SingleCylinderSimulator
from .multi_cylinder_simulator import MultiCylinderSimulator
from .utilities import DataExporter

def run_preset(preset, rpm):
    if preset == "inline4":
        config = create_default_inline_4()
    elif preset == "v8":
        config = create_default_v8()
    else:
        print(f"Error: Unknown preset '{preset}'")
        sys.exit(1)
    
    config.rpm = rpm
    
    print("\n" + "═" * 50)
    print(f"  SIMULATION: {preset.upper()} at {rpm:.0f} RPM")
    print("═" * 50)
    
    # Run single cylinder reference
    sc_sim = SingleCylinderSimulator(config)
    reference_results = sc_sim.simulate_cycle()
    
    # Run multi-cylinder
    mc_sim = MultiCylinderSimulator(config)
    mc_results = mc_sim.simulate_engine()
    
    print(f"Status: Complete")
    print(f"Mean Torque: {mc_results.mean_torque:.2f} Nm")
    print(f"Indicated Power: {mc_results.total_indicated_power_kw:.2f} kW")
    print("─" * 50)
    
    # Export results
    exporter = DataExporter()
    exporter.export_to_csv(mc_results, "multi_cylinder_data.csv")
    
    summary = {
        "preset": preset,
        "rpm": rpm,
        "mean_torque_nm": mc_results.mean_torque,
        "indicated_power_kw": mc_results.total_indicated_power_kw,
        "torque_variation_pct": mc_results.torque_variation,
        "smoothness_factor": mc_results.smoothness_factor
    }
    exporter.export_to_json(summary, "multi_cylinder_metrics.json")
    print("Results exported to:")
    print(" - multi_cylinder_data.csv")
    print(" - multi_cylinder_metrics.json")
    print("═" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Engine Combustion Simulator CLI")
    parser.add_argument("--preset", choices=["inline4", "v8"], default="inline4", help="Engine preset to run (default: inline4)")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed in RPM (default: 3000.0)")
    
    args = parser.parse_args()
    run_preset(args.preset, args.rpm)

if __name__ == "__main__":
    main()
