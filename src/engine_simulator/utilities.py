"""
Utilities Module
Helper functions for unit conversion, validation, and data export.

Author: Mohith Sai Gorla
Date:   27-02-2026
"""

import json
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class UnitSystem(Enum):
    """Unit system enumeration"""

    SI = "si"
    IMPERIAL = "imperial"


class UnitConverter:
    """
    Unit conversion utilities.

    Converts between SI and Imperial units for engine parameters.
    """

    # Conversion factors (to SI)
    CONVERSIONS = {
        # Length
        "inch_to_meter": 0.0254,
        "meter_to_inch": 1.0 / 0.0254,  # exact
        # Volume
        "liter_to_m3": 1.0e-3,
        "m3_to_liter": 1.0e3,
        "cubic_inch_to_m3": 1.6387064e-5,  # exact per NIST
        "m3_to_cubic_inch": 1.0 / 1.6387064e-5,
        # Pressure
        "psi_to_pa": 6894.757293168,  # exact
        "pa_to_psi": 1.0 / 6894.757293168,
        "bar_to_pa": 1.0e5,
        "pa_to_bar": 1.0e-5,
        # Power  (1 hp = 550 ft·lbf/s, exact by US definition)
        "hp_to_w": 745.69987158227022,  # NIST exact
        "w_to_hp": 1.0 / 745.69987158227022,
        "kw_to_hp": 1.0 / 0.74569987158227022,
        "hp_to_kw": 0.74569987158227022,
        # Torque
        "lbft_to_nm": 1.3558179483314004,  # exact
        "nm_to_lbft": 1.0 / 1.3558179483314004,
    }

    @staticmethod
    def length_to_si(value: float, from_unit: str) -> float:
        """Convert length to meters"""
        if from_unit == "in" or from_unit == "inch":
            return value * UnitConverter.CONVERSIONS["inch_to_meter"]
        elif from_unit == "m" or from_unit == "meter":
            return value
        else:
            raise ValueError(f"Unknown length unit: {from_unit}")

    @staticmethod
    def length_from_si(value: float, to_unit: str) -> float:
        """Convert length from meters"""
        if to_unit == "in" or to_unit == "inch":
            return value * UnitConverter.CONVERSIONS["meter_to_inch"]
        elif to_unit == "m" or to_unit == "meter":
            return value
        else:
            raise ValueError(f"Unknown length unit: {to_unit}")

    @staticmethod
    def pressure_to_si(value: float, from_unit: str) -> float:
        """Convert pressure to Pascals"""
        if from_unit == "psi":
            return value * UnitConverter.CONVERSIONS["psi_to_pa"]
        elif from_unit == "bar":
            return value * UnitConverter.CONVERSIONS["bar_to_pa"]
        elif from_unit == "pa":
            return value
        else:
            raise ValueError(f"Unknown pressure unit: {from_unit}")

    @staticmethod
    def pressure_from_si(value: float, to_unit: str) -> float:
        """Convert pressure from Pascals"""
        if to_unit == "psi":
            return value * UnitConverter.CONVERSIONS["pa_to_psi"]
        elif to_unit == "bar":
            return value * UnitConverter.CONVERSIONS["pa_to_bar"]
        elif to_unit == "pa":
            return value
        else:
            raise ValueError(f"Unknown pressure unit: {to_unit}")

    @staticmethod
    def fahrenheit_to_kelvin(temp_f: float) -> float:
        """Convert Fahrenheit to Kelvin"""
        return (temp_f - 32.0) * 5.0 / 9.0 + 273.15

    @staticmethod
    def kelvin_to_fahrenheit(temp_k: float) -> float:
        """Convert Kelvin to Fahrenheit"""
        return (temp_k - 273.15) * 9.0 / 5.0 + 32.0


class EngineeringValidator:
    """
    Validates engine parameters against engineering constraints.

    Provides warnings for parameters outside typical ranges.
    """

    # Valid ranges for key parameters
    VALID_RANGES = {
        "bore": (0.020, 0.500),  # 20mm to 500mm
        "stroke": (0.020, 0.500),
        "rod_ratio": (2.0, 10.0),
        "compression_ratio": (6.0, 25.0),
        "rpm": (100, 15000),
        "peak_temperature": (1500, 4000),  # K
        "intake_pressure": (50000, 300000),  # Pa (0.5 to 3 bar)
    }

    @staticmethod
    def validate_geometry(
        bore: float, stroke: float, rod_length: float, compression_ratio: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate geometric parameters.

        Args:
            bore: Cylinder bore in meters
            stroke: Piston stroke in meters
            rod_length: Connecting rod length in meters
            compression_ratio: Compression ratio

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check bore
        if not (
            EngineeringValidator.VALID_RANGES["bore"][0]
            <= bore
            <= EngineeringValidator.VALID_RANGES["bore"][1]
        ):
            warnings.append(f"Bore {bore*1000:.1f}mm outside typical range [20, 500]mm")

        # Check stroke
        if not (
            EngineeringValidator.VALID_RANGES["stroke"][0]
            <= stroke
            <= EngineeringValidator.VALID_RANGES["stroke"][1]
        ):
            warnings.append(
                f"Stroke {stroke*1000:.1f}mm outside typical range [20, 500]mm"
            )

        # Check bore/stroke ratio
        bore_stroke_ratio = bore / stroke
        if bore_stroke_ratio > 1.3:
            warnings.append(
                f"Oversquare engine (B/S={bore_stroke_ratio:.2f}): High RPM capability"
            )
        elif bore_stroke_ratio < 0.8:
            warnings.append(
                f"Undersquare engine (B/S={bore_stroke_ratio:.2f}): High torque bias"
            )

        # Check rod ratio
        rod_ratio = rod_length / (stroke / 2.0)
        if not (
            EngineeringValidator.VALID_RANGES["rod_ratio"][0]
            <= rod_ratio
            <= EngineeringValidator.VALID_RANGES["rod_ratio"][1]
        ):
            warnings.append(
                f"Rod ratio {rod_ratio:.2f} outside typical range [2.0, 10.0]"
            )

        if rod_ratio < 3.0:
            warnings.append("Low rod ratio: Expect higher side loading on piston")

        # Check compression ratio
        if not (
            EngineeringValidator.VALID_RANGES["compression_ratio"][0]
            <= compression_ratio
            <= EngineeringValidator.VALID_RANGES["compression_ratio"][1]
        ):
            warnings.append(
                f"Compression ratio {compression_ratio:.1f} outside typical range [6, 25]"
            )

        if compression_ratio > 15.0:
            warnings.append(
                "High compression ratio: Suitable for diesel, risk of knock for SI"
            )

        is_valid = len(warnings) == 0
        return is_valid, warnings

    @staticmethod
    def validate_operating_conditions(
        rpm: float, peak_temp: float, intake_pressure: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate operating conditions.

        Args:
            rpm: Engine speed in RPM
            peak_temp: Peak combustion temperature in K
            intake_pressure: Intake manifold pressure in Pa

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check RPM
        if not (
            EngineeringValidator.VALID_RANGES["rpm"][0]
            <= rpm
            <= EngineeringValidator.VALID_RANGES["rpm"][1]
        ):
            warnings.append(f"RPM {rpm:.0f} outside typical range [100, 15000]")

        # Check peak temperature
        if not (
            EngineeringValidator.VALID_RANGES["peak_temperature"][0]
            <= peak_temp
            <= EngineeringValidator.VALID_RANGES["peak_temperature"][1]
        ):
            warnings.append(
                f"Peak temperature {peak_temp:.0f}K outside typical range [1500, 4000]K"
            )

        # Check mean piston speed (should be < 20 m/s for reliability)
        # This requires stroke, so we'll skip for now or pass it in

        is_valid = len(warnings) == 0
        return is_valid, warnings


class DataExporter:
    """
    Export simulation results to various formats.

    Supports: CSV, JSON, custom report formats
    """

    @staticmethod
    def export_to_csv(
        results: Any, filepath: str, variables: Optional[List[str]] = None
    ):
        """
        Export cycle results to CSV file.

        Args:
            results: CycleResults or MultiCylinderResults object
            filepath: Output file path
            variables: List of variable names to export (None = all)
        """

        # Extract data arrays
        data_dict = {}

        if hasattr(results, "crank_angles_deg"):
            data_dict["crank_angle_deg"] = results.crank_angles_deg

        if hasattr(results, "volume"):
            data_dict["volume_m3"] = results.volume

        if hasattr(results, "pressure"):
            data_dict["pressure_pa"] = results.pressure
            data_dict["pressure_bar"] = results.pressure / 1e5

        if hasattr(results, "temperature"):
            data_dict["temperature_k"] = results.temperature

        if hasattr(results, "torque"):
            data_dict["torque_nm"] = results.torque

        if hasattr(results, "total_torque"):
            data_dict["total_torque_nm"] = results.total_torque

        # Filter by requested variables
        if variables:
            data_dict = {k: v for k, v in data_dict.items() if k in variables}

        # Write to CSV
        if len(data_dict) == 0:
            raise ValueError("No data to export")

        # Get length of first array
        num_rows = len(next(iter(data_dict.values())))

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(data_dict.keys())

            # Write data rows
            for i in range(num_rows):
                row = [data_dict[key][i] for key in data_dict.keys()]
                writer.writerow(row)

        print(f"Data exported to {filepath}")

    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: str):
        """
        Export data dictionary to JSON file.

        Args:
            data: Dictionary of data to export
            filepath: Output file path
        """
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Data exported to {filepath}")

    @staticmethod
    def create_performance_report(metrics: Dict[str, float]) -> str:
        """
        Create formatted performance report string.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("ENGINE PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        # Power and efficiency
        report.append("POWER & EFFICIENCY:")
        report.append("-" * 60)
        if "indicated_power_kw" in metrics:
            report.append(
                f"  Indicated Power:        {metrics['indicated_power_kw']:.2f} kW "
                f"({metrics.get('indicated_power_hp', 0):.2f} HP)"
            )
        if "thermal_efficiency" in metrics:
            report.append(
                f"  Thermal Efficiency:     {metrics['thermal_efficiency']*100:.2f}%"
            )
        if "theoretical_efficiency" in metrics:
            report.append(
                f"  Theoretical Efficiency: {metrics['theoretical_efficiency']*100:.2f}%"
            )
        report.append("")

        # Pressure and temperature
        report.append("PEAK VALUES:")
        report.append("-" * 60)
        if "peak_pressure_mpa" in metrics:
            report.append(
                f"  Peak Pressure:          {metrics['peak_pressure_mpa']:.2f} MPa "
                f"({metrics['peak_pressure_mpa']*10:.1f} bar)"
            )
        if "peak_temperature_k" in metrics:
            report.append(
                f"  Peak Temperature:       {metrics['peak_temperature_k']:.0f} K "
                f"({metrics['peak_temperature_k']-273.15:.0f} °C)"
            )
        if "peak_torque_nm" in metrics:
            report.append(
                f"  Peak Torque:            {metrics['peak_torque_nm']:.2f} N·m"
            )
        report.append("")

        # IMEP
        report.append("INDICATED MEAN EFFECTIVE PRESSURE:")
        report.append("-" * 60)
        if "imep_bar" in metrics:
            report.append(
                f"  IMEP:                   {metrics['imep_bar']:.2f} bar "
                f"({metrics.get('imep_pa', 0)/1000:.0f} kPa)"
            )
        report.append("")

        # Specific outputs
        report.append("SPECIFIC OUTPUTS:")
        report.append("-" * 60)
        if "specific_power_kw_per_liter" in metrics:
            report.append(
                f"  Specific Power:         {metrics['specific_power_kw_per_liter']:.2f} kW/L"
            )
        if "displacement_liters" in metrics:
            report.append(
                f"  Total Displacement:     {metrics['displacement_liters']:.3f} L"
            )
        if "mean_piston_speed_ms" in metrics:
            report.append(
                f"  Mean Piston Speed:      {metrics['mean_piston_speed_ms']:.2f} m/s"
            )
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a data series.

    Args:
        data: List or array of numerical data

    Returns:
        Dictionary of statistics
    """
    data_array = np.array(data, dtype=float)

    stats = {
        "mean": float(np.mean(data_array)),
        "std": float(np.std(data_array, ddof=0)),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "median": float(np.median(data_array)),
        "range": float(np.max(data_array) - np.min(data_array)),  # peak-to-peak
    }

    return stats
