"""
Visualization Module
Creates plots and animations for engine simulation results.

Author: Mohith Sai Gorla
Date: 27-02-2026
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Optional

from .single_cylinder_simulator import CycleResults
from .multi_cylinder_simulator import MultiCylinderResults


class EnginePlotter:
    """
    Creates publication-quality plots for engine simulation results.

    Supports:
    - P-V diagrams
    - Pressure vs crank angle
    - Temperature vs crank angle
    - Torque curves
    - Multi-cylinder comparisons
    """

    def __init__(self, style: str = "default"):
        """
        Initialize plotter with specified style.

        Args:
            style: Matplotlib style ('default', 'seaborn', 'ggplot')
        """
        if style != "default":
            try:
                plt.style.use(style)
            except Exception as e:
                print(f"Warning: Style '{style}' not found, using default. Error: {e}")

        self.fig_size = (12, 8)
        self.dpi = 100

    def plot_pv_diagram(self, results: CycleResults, save_path: Optional[str] = None):
        """
        Create P-V (Pressure-Volume) diagram.

        Args:
            results: Single cylinder simulation results
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert to more convenient units
        volume_cm3 = results.volume * 1e6  # m³ to cm³
        pressure_bar = results.pressure / 1e5  # Pa to bar

        # Plot cycle
        ax.plot(volume_cm3, pressure_bar, "b-", linewidth=2, label="Otto Cycle")

        # Mark key points
        # TDC compression (min volume, high pressure)
        tdc_idx = np.argmin(results.volume)
        ax.plot(
            volume_cm3[tdc_idx],
            pressure_bar[tdc_idx],
            "ro",
            markersize=10,
            label="TDC Firing",
        )

        # BDC (max volume)
        bdc_idx = np.argmax(results.volume)
        ax.plot(
            volume_cm3[bdc_idx], pressure_bar[bdc_idx], "go", markersize=10, label="BDC"
        )

        ax.set_xlabel("Volume (cm³)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pressure (bar)", fontsize=12, fontweight="bold")
        ax.set_title("P-V Diagram (Indicator Diagram)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add IMEP annotation
        imep_bar = results.imep / 1e5
        ax.text(
            0.05,
            0.95,
            f"IMEP = {imep_bar:.2f} bar",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"P-V diagram saved to {save_path}")

        plt.show()

    def plot_pressure_angle(
        self, results: CycleResults, save_path: Optional[str] = None
    ):
        """
        Plot pressure vs crank angle.

        Args:
            results: Single cylinder simulation results
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        pressure_bar = results.pressure / 1e5

        ax.plot(results.crank_angles_deg, pressure_bar, "b-", linewidth=2)

        # Mark cycle phases
        colors = {
            "INTAKE": "green",
            "COMPRESSION": "orange",
            "EXPANSION": "red",
            "EXHAUST": "gray",
        }

        current_phase = results.phases[0]
        phase_start = 0

        for i, phase in enumerate(results.phases):
            if phase != current_phase or i == len(results.phases) - 1:
                ax.axvspan(
                    results.crank_angles_deg[phase_start],
                    results.crank_angles_deg[i],
                    alpha=0.2,
                    color=colors.get(current_phase, "white"),
                    label=(
                        current_phase
                        if phase_start == 0
                        or results.phases[phase_start - 1] != current_phase
                        else ""
                    ),
                )
                current_phase = phase
                phase_start = i

        ax.set_xlabel("Crank Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pressure (bar)", fontsize=12, fontweight="bold")
        ax.set_title("Cylinder Pressure vs Crank Angle", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add vertical lines for TDC/BDC
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="TDC Firing")
        ax.axvline(180, color="blue", linestyle="--", alpha=0.5, label="BDC")
        ax.axvline(360, color="red", linestyle="--", alpha=0.5, label="TDC Exhaust")
        ax.axvline(540, color="blue", linestyle="--", alpha=0.5, label="BDC Intake")

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Pressure-angle plot saved to {save_path}")

        plt.show()

    def plot_temperature_angle(
        self, results: CycleResults, save_path: Optional[str] = None
    ):
        """
        Plot temperature vs crank angle.

        Args:
            results: Single cylinder simulation results
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(results.crank_angles_deg, results.temperature, "r-", linewidth=2)

        ax.set_xlabel("Crank Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Temperature (K)", fontsize=12, fontweight="bold")
        ax.set_title("Gas Temperature vs Crank Angle", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add vertical lines
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="TDC Firing")
        ax.axvline(360, color="red", linestyle="--", alpha=0.5)

        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Temperature-angle plot saved to {save_path}")

        plt.show()

    def plot_torque_curve(self, results: CycleResults, save_path: Optional[str] = None):
        """
        Plot torque vs crank angle.

        Args:
            results: Single cylinder simulation results
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(results.crank_angles_deg, results.torque, "g-", linewidth=2)
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)

        ax.set_xlabel("Crank Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (N·m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Instantaneous Torque vs Crank Angle", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Calculate and display mean torque
        mean_torque = np.mean(results.torque[results.torque > 0])
        ax.axhline(
            mean_torque,
            color="r",
            linestyle="--",
            label=f"Mean Torque = {mean_torque:.2f} N·m",
        )

        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Torque curve saved to {save_path}")

        plt.show()

    def plot_multi_cylinder_torque(
        self, results: MultiCylinderResults, save_path: Optional[str] = None
    ):
        """
        Plot multi-cylinder total torque.

        Args:
            results: Multi-cylinder simulation results
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Individual cylinder torques
        for i, cyl_result in enumerate(results.cylinder_results):
            ax1.plot(
                results.crank_angles_deg,
                cyl_result.torque,
                alpha=0.5,
                linewidth=1,
                label=f"Cylinder {i+1}",
            )

        ax1.set_xlabel("Crank Angle (degrees)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Torque (N·m)", fontsize=12, fontweight="bold")
        ax1.set_title("Individual Cylinder Torques", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, ncol=2)

        # Total torque
        ax2.plot(
            results.crank_angles_deg,
            results.total_torque,
            "b-",
            linewidth=2,
            label="Total Torque",
        )
        ax2.axhline(
            results.mean_torque,
            color="r",
            linestyle="--",
            label=f"Mean = {results.mean_torque:.2f} N·m",
        )

        ax2.set_xlabel("Crank Angle (degrees)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Total Torque (N·m)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Total Engine Torque (Variation: {results.torque_variation:.2f}%)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Multi-cylinder torque plot saved to {save_path}")

        plt.show()

    def plot_comprehensive_analysis(
        self, results: CycleResults, save_path: Optional[str] = None
    ):
        """
        Create comprehensive 4-panel analysis plot.

        Args:
            results: Single cylinder simulation results
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # P-V Diagram
        ax1 = fig.add_subplot(gs[0, 0])
        volume_cm3 = results.volume * 1e6
        pressure_bar = results.pressure / 1e5
        ax1.plot(volume_cm3, pressure_bar, "b-", linewidth=2)
        ax1.set_xlabel("Volume (cm³)", fontweight="bold")
        ax1.set_ylabel("Pressure (bar)", fontweight="bold")
        ax1.set_title("P-V Diagram", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Pressure vs Angle
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results.crank_angles_deg, pressure_bar, "b-", linewidth=2)
        ax2.set_xlabel("Crank Angle (deg)", fontweight="bold")
        ax2.set_ylabel("Pressure (bar)", fontweight="bold")
        ax2.set_title("Pressure vs Crank Angle", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Temperature vs Angle
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(results.crank_angles_deg, results.temperature, "r-", linewidth=2)
        ax3.set_xlabel("Crank Angle (deg)", fontweight="bold")
        ax3.set_ylabel("Temperature (K)", fontweight="bold")
        ax3.set_title("Temperature vs Crank Angle", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Torque vs Angle
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(results.crank_angles_deg, results.torque, "g-", linewidth=2)
        ax4.axhline(0, color="k", linestyle="-", alpha=0.3)
        ax4.set_xlabel("Crank Angle (deg)", fontweight="bold")
        ax4.set_ylabel("Torque (N·m)", fontweight="bold")
        ax4.set_title("Instantaneous Torque", fontweight="bold")
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            "Comprehensive Engine Cycle Analysis", fontsize=16, fontweight="bold"
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comprehensive analysis saved to {save_path}")

        plt.show()
