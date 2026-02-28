"""
Multi-Cylinder Simulator
Coordinates multiple cylinders with proper phasing, force balance,
and torque summation.

Author: Mohith Sai Gorla
Date:   27-02-2026
"""

import math
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .engine_config import EngineConfiguration, EngineType
from .single_cylinder_simulator import SingleCylinderSimulator, CycleResults


import numpy.typing as npt


@dataclass
class MultiCylinderResults:
    """Results from a multi-cylinder engine simulation."""

    time: npt.NDArray[np.float64]  # [s]
    crank_angles_deg: npt.NDArray[np.float64]  # [deg]

    cylinder_results: List[CycleResults]

    total_torque: npt.NDArray[np.float64]  # [N·m]  sum over all cylinders
    total_power: npt.NDArray[np.float64]  # [W]    instantaneous power

    # Reciprocating inertia force balance (X = lateral, Y = axial/vertical)
    primary_force_x: npt.NDArray[np.float64]  # [N]  Σ F1 · sin(bank_angle)
    primary_force_y: npt.NDArray[np.float64]  # [N]  Σ F1 · cos(bank_angle)
    secondary_force_x: npt.NDArray[np.float64]  # [N]  Σ F2 · sin(bank_angle)
    secondary_force_y: npt.NDArray[np.float64]  # [N]  Σ F2 · cos(bank_angle)
    total_unbalanced_force: npt.NDArray[np.float64]  # [N]  √(Fx² + Fy²)

    # Scalar summary
    mean_torque: float = 0.0
    torque_variation: float = 0.0  # %
    total_indicated_power_kw: float = 0.0
    smoothness_factor: float = 0.0  # lower = smoother


class Cylinder:
    """Represents a single cylinder in a multi-cylinder engine.

    Stores the phase offset and bank angle for one cylinder and provides
    the mapping from global to local crank angle.
    """

    def __init__(
        self,
        cylinder_number: int,
        crank_phase_deg: float,
        bank_angle_deg: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        cylinder_number : int    1-indexed cylinder identifier
        crank_phase_deg : float  Crank phase offset  [deg]
        bank_angle_deg  : float  Engine bank angle   [deg]  (0 for inline)
        """
        self.number = cylinder_number
        self.crank_phase_deg = crank_phase_deg
        self.crank_phase_rad = math.radians(crank_phase_deg)
        self.bank_angle_deg = bank_angle_deg
        self.bank_angle_rad = math.radians(bank_angle_deg)

        self.results: Optional[CycleResults] = None

    def local_angle(self, global_angle_deg: float) -> float:
        """Map global crank angle to this cylinder's local crank angle.

        local_angle = (global_angle + crank_phase) mod 720°

        Parameters
        ----------
        global_angle_deg : float  [deg]

        Returns
        -------
        float  Local crank angle  [deg]  ∈ [0°, 720°)
        """
        return (global_angle_deg + self.crank_phase_deg) % 720.0


class MultiCylinderSimulator:
    """Simulates a multi-cylinder engine with correct phasing and balance.

    Each cylinder is modelled by phase-shifting the reference cycle data
    (computed once from SingleCylinderSimulator) to its own crank angle.
    This is accurate for identical-cylinder configurations.

    Force Balance
    -------------
    The *reciprocating inertia* force on a single cylinder at angle θ is:

        F_recip(θ) = −m_recip · a_piston(θ)

    Decomposed into harmonics (first-order approximation):
        F₁(θ) = m_recip · r · ω² · cos(θ)        (primary / 1st order)
        F₂(θ) = m_recip · r · ω² · λ · cos(2θ)   (secondary / 2nd order)

    These are projected onto engine X and Y axes using the bank angle β:
        Fx += Fₙ · sin(β)
        Fy += Fₙ · cos(β)
    """

    def __init__(self, config: EngineConfiguration) -> None:
        """
        Parameters
        ----------
        config : EngineConfiguration
        """
        self.config = config
        self.num_cylinders = config.geometry.num_cylinders

        # Build cylinder objects
        self.cylinders: List[Cylinder] = []
        for i in range(self.num_cylinders):
            phase = config.multi_cylinder.crank_phases[i]

            # V-engine: alternate bank angles ± (bank_angle / 2)
            if config.multi_cylinder.engine_type == EngineType.V_ENGINE:
                bank = (
                    config.multi_cylinder.bank_angle / 2.0
                    if i % 2 == 0
                    else -config.multi_cylinder.bank_angle / 2.0
                )
            else:
                bank = 0.0

            self.cylinders.append(
                Cylinder(
                    cylinder_number=i + 1,
                    crank_phase_deg=phase,
                    bank_angle_deg=bank,
                )
            )

        # Single-cylinder simulator for reference cycle computation
        self._single_sim = SingleCylinderSimulator(config)

        self.rpm = config.operating.rpm
        self.angular_velocity = config.operating.angular_velocity  # rad/s

        # Reciprocating mass (piston + pin + fraction of con-rod)
        # Reuse value from single-cylinder simulator
        self._m_recip: float = self._single_sim.piston_mass
        self._r: float = self._single_sim.crank_radius
        self._lambda: float = self._single_sim.slider_crank.lambda_ratio

    def simulate_engine(self) -> MultiCylinderResults:
        """Simulate the complete multi-cylinder engine.

        Returns
        -------
        MultiCylinderResults
        """
        # ── Simulate reference cylinder ───────────────────────────────────
        ref = self._single_sim.simulate_cycle()
        self.cylinders[0].results = ref

        crank_angles_deg = ref.crank_angles_deg
        num_steps = len(crank_angles_deg)

        # Time array: t = θ / ω_deg  where ω_deg = 360·RPM/60  [deg/s]
        omega_deg = 360.0 * self.rpm / 60.0  # deg/s
        time = crank_angles_deg / omega_deg  # [s]

        # ── Initialise combined arrays ────────────────────────────────────
        total_torque = np.zeros(num_steps)
        primary_force_x = np.zeros(num_steps)
        primary_force_y = np.zeros(num_steps)
        secondary_force_x = np.zeros(num_steps)
        secondary_force_y = np.zeros(num_steps)

        # ── Per-cylinder contribution ─────────────────────────────────────
        for cyl in self.cylinders:
            if cyl.number == 1:
                cyl_results = ref
            else:
                # BUG-06 fix: properly phase-shift reference data
                cyl_results = self._phase_shift_results(ref, cyl.crank_phase_deg)
                cyl.results = cyl_results

            bank_sin = math.sin(cyl.bank_angle_rad)
            bank_cos = math.cos(cyl.bank_angle_rad)

            # Torque is already phase-shifted to the global grid in cyl_results
            # so we just add it directly (vectorized)
            total_torque += cyl_results.torque

            # ── BUG-07 fix: reciprocating inertia force harmonics ──
            # Local angles for this cylinder in radians [vectorized]
            local_deg = (crank_angles_deg + cyl.crank_phase_deg) % 720.0
            local_rad = np.deg2rad(local_deg)

            inertia_scale = self._m_recip * self._r * self.angular_velocity**2

            f_primary = inertia_scale * np.cos(local_rad)
            f_secondary = inertia_scale * self._lambda * np.cos(2.0 * local_rad)

            # Project to X (lateral) and Y (cylinder axis) via bank angle
            primary_force_x += f_primary * bank_sin
            primary_force_y += f_primary * bank_cos
            secondary_force_x += f_secondary * bank_sin
            secondary_force_y += f_secondary * bank_cos

        # ── Resultant unbalanced force ────────────────────────────────────
        fx_total = primary_force_x + secondary_force_x
        fy_total = primary_force_y + secondary_force_y
        total_unbalanced_force = np.hypot(fx_total, fy_total)

        # ── Power ─────────────────────────────────────────────────────────
        total_power = total_torque * self.angular_velocity

        # ── Scalar summary ────────────────────────────────────────────────
        mean_torque = float(np.mean(total_torque))
        torque_std = float(np.std(total_torque))
        torque_variation = (
            torque_std / abs(mean_torque) * 100.0 if abs(mean_torque) > 1e-9 else 0.0
        )
        smoothness_factor = torque_variation / self.num_cylinders

        cycles_per_second = self.rpm / 120.0
        total_indicated_work = sum(cyl.results.indicated_work for cyl in self.cylinders)
        total_indicated_power_kw = total_indicated_work * cycles_per_second / 1000.0

        return MultiCylinderResults(
            time=time,
            crank_angles_deg=crank_angles_deg,
            cylinder_results=[cyl.results for cyl in self.cylinders],
            total_torque=total_torque,
            total_power=total_power,
            primary_force_x=primary_force_x,
            primary_force_y=primary_force_y,
            secondary_force_x=secondary_force_x,
            secondary_force_y=secondary_force_y,
            total_unbalanced_force=total_unbalanced_force,
            mean_torque=mean_torque,
            torque_variation=torque_variation,
            total_indicated_power_kw=total_indicated_power_kw,
            smoothness_factor=smoothness_factor,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _phase_shift_results(
        self, reference: CycleResults, phase_deg: float
    ) -> CycleResults:
        """Create a phase-shifted copy of the reference cycle results.

        BUG-06 fix: previously returned `reference` unchanged.

        For each global angle θ in the reference angle array, the shifted
        cylinder's data is read from angle  (θ + phase_deg) mod 720°.
        Interpolation is used to handle non-integer angle offsets.

        Parameters
        ----------
        reference : CycleResults  Reference (cylinder 1) simulation results.
        phase_deg : float         Phase offset  [deg].

        Returns
        -------
        CycleResults  Phase-shifted cycle results.
        """

        angles = reference.crank_angles_deg
        n = len(angles)

        # Local angles for the shifted cylinder
        local_angles = (angles + phase_deg) % 720.0

        ang_res_deg = angles[1] - angles[0] if n > 1 else 1.0
        shift_steps = phase_deg / ang_res_deg

        if abs(shift_steps - round(shift_steps)) < 1e-9:
            # Exact integer shift - use ultra-fast np.roll
            # rolling left (-shift_idx) maps data(theta + phase) to theta
            shift_idx = int(round(shift_steps))

            def _interp_circular(data: np.ndarray) -> np.ndarray:
                data_base = data[:-1]
                rolled_base = np.roll(data_base, -shift_idx)
                return np.append(rolled_base, rolled_base[0])

        else:
            # Fractional shift - use numpy interpolation
            def _interp_circular(data: np.ndarray) -> np.ndarray:
                angles_ext = np.concatenate([angles - 720.0, angles, angles + 720.0])
                data_ext = np.concatenate([data, data, data])
                return np.interp(local_angles, angles_ext, data_ext)

        return CycleResults(
            crank_angles_deg=reference.crank_angles_deg,
            crank_angles_rad=reference.crank_angles_rad,
            displacement=_interp_circular(reference.displacement),
            velocity=_interp_circular(reference.velocity),
            acceleration=_interp_circular(reference.acceleration),
            volume=_interp_circular(reference.volume),
            pressure=_interp_circular(reference.pressure),
            temperature=_interp_circular(reference.temperature),
            gamma=_interp_circular(reference.gamma),
            gas_force=_interp_circular(reference.gas_force),
            inertia_force=_interp_circular(reference.inertia_force),
            total_force=_interp_circular(reference.total_force),
            torque=_interp_circular(reference.torque),
            indicated_work=reference.indicated_work,
            imep=reference.imep,
            thermal_efficiency=reference.thermal_efficiency,
            heat_added=reference.heat_added,
            heat_rejected=reference.heat_rejected,
            phases=list(reference.phases),
        )

    @staticmethod
    def _find_angle_index(angle_array: np.ndarray, target_angle: float) -> int:
        """Find index of the closest angle in a sorted angle array.

        Parameters
        ----------
        angle_array  : np.ndarray  Sorted angle array  [deg]
        target_angle : float       Target angle  [deg]

        Returns
        -------
        int  Index of closest match
        """
        target_norm = target_angle % 720.0
        return int(np.argmin(np.abs(angle_array - target_norm)))

    # ── Balance analysis ──────────────────────────────────────────────────

    def analyze_balance(self, results: MultiCylinderResults) -> Dict[str, object]:
        """Compute force and moment balance metrics.

        Parameters
        ----------
        results : MultiCylinderResults

        Returns
        -------
        Dict[str, object]  Balance quality metrics.
        """
        pf_mag = np.hypot(results.primary_force_x, results.primary_force_y)
        sf_mag = np.hypot(results.secondary_force_x, results.secondary_force_y)

        pf_max = float(np.max(pf_mag))
        pf_mean = float(np.mean(pf_mag))
        sf_max = float(np.max(sf_mag))
        sf_mean = float(np.mean(sf_mag))

        unbalanced_max = float(np.max(results.total_unbalanced_force))
        unbalanced_mean = float(np.mean(results.total_unbalanced_force))

        # Balance quality: fraction of unbalanced force relative to single-cylinder
        # peak total force (consistent reference)
        single_cyl_peak = float(np.max(np.abs(results.cylinder_results[0].total_force)))
        balance_quality = (
            (unbalanced_mean / single_cyl_peak) * 100.0
            if single_cyl_peak > 1e-9
            else 0.0
        )

        # Threshold for "balanced": < 1% of a typical peak gas force
        _balance_threshold = 100.0  # N  (informational criterion)

        return {
            "primary_force_max_n": pf_max,
            "primary_force_mean_n": pf_mean,
            "secondary_force_max_n": sf_max,
            "secondary_force_mean_n": sf_mean,
            "unbalanced_force_max_n": unbalanced_max,
            "unbalanced_force_mean_n": unbalanced_mean,
            "balance_quality_percent": balance_quality,
            "is_primary_balanced": pf_mean < _balance_threshold,
            "is_secondary_balanced": sf_mean < _balance_threshold,
        }
