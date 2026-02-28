"""
Single Cylinder Simulator
Coordinates kinematics and thermodynamics for a single cylinder.

Author: Mohith Sai Gorla
Date:   27-02-2026
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

from .engine_config import EngineConfiguration
from .kinematics import SliderCrank, ValveTiming
from .thermodynamics import (
    EngineCycle,
    WorkingFluid,
    CombustionModel,
    HeatTransferModel,
    ThermodynamicState,
)


import numpy.typing as npt


@dataclass
class CycleResults:
    """Results from a single-cylinder 4-stroke simulation.

    All arrays are aligned to `crank_angles_deg`.
    """

    crank_angles_deg: npt.NDArray[np.float64]  # [deg]
    crank_angles_rad: npt.NDArray[np.float64]  # [rad]

    # Kinematics
    displacement: npt.NDArray[np.float64]  # [m]   from TDC
    velocity: npt.NDArray[np.float64]  # [m/s]
    acceleration: npt.NDArray[np.float64]  # [m/s²]

    # Volumes
    volume: npt.NDArray[np.float64]  # [m³]

    # Thermodynamics
    pressure: npt.NDArray[np.float64]  # [Pa]
    temperature: npt.NDArray[np.float64]  # [K]
    gamma: npt.NDArray[np.float64]  # [-]

    # Forces and torque
    gas_force: npt.NDArray[np.float64]  # [N]  net gas force on piston
    inertia_force: npt.NDArray[np.float64]  # [N]  piston + pin assembly inertia
    total_force: npt.NDArray[np.float64]  # [N]  gas + inertia
    torque: npt.NDArray[np.float64]  # [N·m]

    # Performance metrics
    indicated_work: float = 0.0  # [J]   W = ∮ P dV
    imep: float = 0.0  # [Pa]  Indicated Mean Effective Pressure
    thermal_efficiency: float = 0.0  # [-]
    heat_added: float = 0.0  # [J]   Q_in
    heat_rejected: float = 0.0  # [J]   Q_out (positive)

    # Cycle phase labels
    phases: List[str] = field(default_factory=list)


class SingleCylinderSimulator:
    """Simulates a single cylinder over a complete 4-stroke cycle.

    Combines:
    - Exact slider-crank kinematics (SliderCrank)
    - Otto-cycle thermodynamics with variable γ (EngineCycle)
    - Wiebe or ideal-CV combustion model (CombustionModel)
    - Optional Woschni heat transfer (HeatTransferModel)
    - Force and torque analysis
    """

    # Default piston assembly mass used when not yet configurable
    _DEFAULT_PISTON_MASS_KG: float = 0.5

    def __init__(self, config: EngineConfiguration) -> None:
        """
        Parameters
        ----------
        config : EngineConfiguration
            Validated engine configuration object.
        """
        self.config = config

        # ── Geometry ──────────────────────────────────────────────────────
        self.bore = config.geometry.bore
        self.stroke = config.geometry.stroke
        self.crank_radius = config.geometry.crank_radius
        self.rod_length = config.geometry.connecting_rod_length
        self.compression_ratio = config.geometry.compression_ratio

        self.piston_area = config.geometry.piston_area
        self.displacement_volume = config.geometry.displacement_per_cylinder
        self.clearance_volume = config.geometry.clearance_volume

        # ── Operating conditions ──────────────────────────────────────────
        self.rpm = config.operating.rpm
        self.angular_velocity = config.operating.angular_velocity  # rad/s

        # ── Initial conditions ────────────────────────────────────────────
        self.initial_pressure = config.thermodynamics.intake_pressure
        self.initial_temperature = config.thermodynamics.intake_temperature
        self.peak_temperature = config.thermodynamics.peak_combustion_temperature

        # Trapped mass at BDC (full cylinder at intake conditions)
        total_volume = self.clearance_volume + self.displacement_volume
        self.mass = (self.initial_pressure * total_volume) / (
            config.thermodynamics.gas_constant * self.initial_temperature
        )

        # Piston + pin assembly mass [kg] — hardcoded for now; make configurable
        self.piston_mass: float = self._DEFAULT_PISTON_MASS_KG

        # ── Sub-systems ───────────────────────────────────────────────────
        self.slider_crank = SliderCrank(self.crank_radius, self.rod_length)
        self.valve_timing = ValveTiming()

        self.working_fluid = WorkingFluid(
            gas_constant=config.thermodynamics.gas_constant,
            cv_a0=config.thermodynamics.cv_a0,
            cv_a1=config.thermodynamics.cv_a1,
            cv_a2=config.thermodynamics.cv_a2,
        )

        _comb_type = config.combustion.combustion_type.value
        self.combustion_model = CombustionModel(combustion_type=_comb_type)

        if _comb_type == "wiebe":
            # CombustionModel uses start_deg relative to TDC firing.
            # Config stores combustion_start_btdc (positive = before TDC).
            self.combustion_model.set_wiebe_parameters(
                a=config.combustion.wiebe_a,
                m=config.combustion.wiebe_m,
                start_deg=-config.combustion.combustion_start_btdc,
                duration_deg=config.combustion.combustion_duration,
            )

        heat_transfer: Optional[HeatTransferModel] = None
        if config.combustion.enable_heat_transfer:
            heat_transfer = HeatTransferModel(
                bore=self.bore,
                wall_temperature=config.combustion.wall_temperature,
            )
        self.heat_transfer = heat_transfer

        self.engine_cycle = EngineCycle(
            self.working_fluid,
            self.combustion_model,
            self.heat_transfer,
        )

        # ── Precompute heat input ───────────────────────────
        # Compressed temperature at TDC (isentropic compression from BDC)
        gamma_init = self.working_fluid.gamma(self.initial_temperature)
        self._T_comp_tdc = self.initial_temperature * (
            self.compression_ratio ** (gamma_init - 1.0)
        )

        # Q_in = m · [u(T_peak) − u(T_comp_TDC)]
        # Using the exact polynomial integral.
        self._total_heat_input = self.working_fluid.delta_internal_energy(
            temperature_1=self._T_comp_tdc,
            temperature_2=self.peak_temperature,
            mass=self.mass,
        )
        if self._total_heat_input <= 0.0:
            raise ValueError(
                f"Total heat input is non-positive ({self._total_heat_input:.2f} J). "
                f"peak_temperature ({self.peak_temperature} K) must exceed "
                f"compressed TDC temperature ({self._T_comp_tdc:.1f} K)."
            )

    # ── Simulation ────────────────────────────────────────────────────────

    def simulate_cycle(self) -> CycleResults:
        """Run a complete 4-stroke cycle simulation.

        Algorithm
        ---------
        The cycle covers 0° – 720° (inclusive endpoints for plotting).
        Each iteration computes:

        Phase detection  (via ValveTiming.get_cycle_phase)
        ├── INTAKE      : constant-pressure model — P = P_intake, T = T_intake
        ├── COMPRESSION : isentropic step from previous volume to current volume
        ├── EXPANSION   : (a) add incremental combustion heat at constant volume
        │                  (b) isentropic expansion to current volume
        └── EXHAUST     : constant-pressure model — P = P_intake, T = T_intake

        State continuity
        ----------------
        At the COMPRESSION→EXPANSION boundary (TDC, ≈540→0°):
          The last compression step produces the state at clearance volume.
          The burned fraction at θ=0+ is used to compute the first heat increment.

        Returns
        -------
        CycleResults
        """
        ang_res_deg = self.config.simulation.angular_resolution
        # Include the endpoint (720°) for a closed P-V trace
        crank_angles_deg = np.arange(0.0, 720.0 + ang_res_deg, ang_res_deg)
        num_steps = len(crank_angles_deg)
        crank_angles_rad = np.deg2rad(crank_angles_deg)

        # Precompute constants for heat transfer
        time_step = ang_res_deg / (6.0 * self.rpm) if self.rpm > 0 else 0.0
        mean_piston_speed = 2.0 * self.stroke * (self.rpm / 60.0)

        # ── Allocate result arrays ────────────────────────────────────────
        displacement = np.zeros(num_steps)
        velocity = np.zeros(num_steps)
        acceleration = np.zeros(num_steps)
        volume = np.zeros(num_steps)
        pressure = np.zeros(num_steps)
        temperature = np.zeros(num_steps)
        gamma = np.zeros(num_steps)
        gas_force = np.zeros(num_steps)
        inertia_force = np.zeros(num_steps)
        total_force = np.zeros(num_steps)
        torque = np.zeros(num_steps)
        phases: List[str] = []

        # ── Initial thermodynamic state at TDC after compression ─────────
        # (before any combustion; crank angle = 0°, volume = clearance_volume)
        gamma_tdc = self.working_fluid.gamma(self._T_comp_tdc)
        P_comp_tdc = self.initial_pressure * (self.compression_ratio**gamma_tdc)

        current_state = ThermodynamicState(
            pressure=P_comp_tdc,
            temperature=self._T_comp_tdc,
            volume=self.clearance_volume,
            mass=self.mass,
            internal_energy=self.working_fluid.internal_energy(
                self._T_comp_tdc, self.mass
            ),
            gamma=gamma_tdc,
        )

        # ── Track how much of Q_in has been released ──────────────────────
        # Used to distribute incremental combustion heat correctly.
        burn_fraction_prev: float = 0.0

        # ── Main simulation loop ──────────────────────────────────────────
        for i, (theta_deg, theta_rad) in enumerate(
            zip(crank_angles_deg, crank_angles_rad)
        ):
            # 1. Kinematics
            y, v, a = self.slider_crank.kinematics_at_angle(
                theta_rad, self.angular_velocity
            )
            displacement[i] = y
            velocity[i] = v
            acceleration[i] = a

            # 2. Cylinder volume
            vol = self.clearance_volume + self.piston_area * y
            volume[i] = vol

            # 3. Cycle phase
            phase = self.valve_timing.get_cycle_phase(theta_deg)
            phases.append(phase)

            # 4. Thermodynamic state update
            if phase == "INTAKE":
                # Constant-pressure intake (simplified gas-exchange model)
                current_state = ThermodynamicState(
                    pressure=self.initial_pressure,
                    temperature=self.initial_temperature,
                    volume=vol,
                    mass=self.mass,
                    internal_energy=self.working_fluid.internal_energy(
                        self.initial_temperature, self.mass
                    ),
                    gamma=self.working_fluid.gamma(self.initial_temperature),
                )
                burn_fraction_prev = 0.0  # reset for next compression

            elif phase == "COMPRESSION":
                # Isentropic compression step-by-step
                current_state = self.engine_cycle.compress_isentropic(
                    current_state, vol
                )

                if self.heat_transfer is not None:
                    surface_area = 2.0 * self.piston_area + math.pi * self.bore * y
                    q_loss = self.heat_transfer.heat_loss(
                        pressure=current_state.pressure,
                        temperature=current_state.temperature,
                        surface_area=surface_area,
                        mean_piston_speed=mean_piston_speed,
                        time_step=time_step,
                    )
                    if q_loss > 0.0:
                        current_state = self.engine_cycle.reject_heat_constant_volume(
                            current_state, q_loss
                        )
                    elif q_loss < 0.0:
                        current_state = self.engine_cycle.add_heat_constant_volume(
                            current_state, -q_loss
                        )

                burn_fraction_prev = 0.0  # no combustion during compression

            elif phase == "EXPANSION":
                # ── (a) Combustion heat release (constant-volume increment) ──
                # The Wiebe / ideal combustion model distributes heat over the
                # angle range [combustion_start, combustion_end].
                # Only add heat if the burn fraction has increased since the
                # previous step.
                burn_fraction_curr = self.combustion_model.burn_fraction(theta_deg)
                delta_burn = burn_fraction_curr - burn_fraction_prev

                if delta_burn > 0.0:
                    heat_increment = self._total_heat_input * delta_burn
                    # Constant-volume heat addition — volume does not change here
                    current_state = self.engine_cycle.add_heat_constant_volume(
                        current_state, heat_increment
                    )
                    burn_fraction_prev = burn_fraction_curr

                # ── (b) Isentropic expansion to new volume ────────────────
                # Only expand if volume has increased (piston moving downward).
                # Near TDC the volume may be equal to the previous step or
                # even slightly smaller due to angle discretisation — guard
                # against accidentally compressing.
                if vol > current_state.volume:
                    current_state = self.engine_cycle.expand_isentropic(
                        current_state, vol
                    )

                if self.heat_transfer is not None:
                    surface_area = 2.0 * self.piston_area + math.pi * self.bore * y
                    q_loss = self.heat_transfer.heat_loss(
                        pressure=current_state.pressure,
                        temperature=current_state.temperature,
                        surface_area=surface_area,
                        mean_piston_speed=mean_piston_speed,
                        time_step=time_step,
                    )
                    if q_loss > 0.0:
                        current_state = self.engine_cycle.reject_heat_constant_volume(
                            current_state, q_loss
                        )
                    elif q_loss < 0.0:
                        current_state = self.engine_cycle.add_heat_constant_volume(
                            current_state, -q_loss
                        )

                # If vol <= current_state.volume (at exact TDC step),
                # the state remains unchanged (zero-work step).

            elif phase == "EXHAUST":
                # Constant-pressure exhaust (simplified blowdown model)
                current_state = ThermodynamicState(
                    pressure=self.initial_pressure,
                    temperature=self.initial_temperature,
                    volume=vol,
                    mass=self.mass,
                    internal_energy=self.working_fluid.internal_energy(
                        self.initial_temperature, self.mass
                    ),
                    gamma=self.working_fluid.gamma(self.initial_temperature),
                )

            # 5. Store thermodynamic state
            pressure[i] = current_state.pressure
            temperature[i] = current_state.temperature
            gamma[i] = current_state.gamma

            # 6. Forces
            # Net gas force on piston (atmospheric back-pressure not modelled)
            gas_force[i] = current_state.pressure * self.piston_area
            inertia_force[i] = -self.piston_mass * acceleration[i]
            total_force[i] = gas_force[i] + inertia_force[i]

            # 7. Torque via exact kinematic conversion factor
            torque_factor = self.slider_crank.force_to_torque_factor(theta_rad)
            torque[i] = total_force[i] * torque_factor

        # ── Performance metrics ───────────────────────────────────────────
        indicated_work = self.engine_cycle.calculate_work_pdv(pressure, volume)

        # IMEP = W / Vd  (by definition, positive for a power cycle)
        imep = indicated_work / self.displacement_volume

        # Heat balance
        heat_added = self._total_heat_input
        heat_rejected = max(0.0, heat_added - indicated_work)  # ≥ 0 by 2nd law
        thermal_efficiency = indicated_work / heat_added if heat_added > 0.0 else 0.0

        return CycleResults(
            crank_angles_deg=crank_angles_deg,
            crank_angles_rad=crank_angles_rad,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            volume=volume,
            pressure=pressure,
            temperature=temperature,
            gamma=gamma,
            gas_force=gas_force,
            inertia_force=inertia_force,
            total_force=total_force,
            torque=torque,
            indicated_work=indicated_work,
            imep=imep,
            thermal_efficiency=thermal_efficiency,
            heat_added=heat_added,
            heat_rejected=heat_rejected,
            phases=phases,
        )

    # ── Performance metrics ───────────────────────────────────────────────

    def calculate_performance_metrics(self, results: CycleResults) -> Dict[str, float]:
        """Compute comprehensive performance metrics from cycle results.

        Parameters
        ----------
        results : CycleResults

        Returns
        -------
        Dict[str, float]
            Keys include power [kW], IMEP [bar], efficiency, peak values, etc.
        """
        # Cycles per second for a 4-stroke engine: N / (60 × 2)
        cycles_per_second = self.rpm / 120.0

        indicated_power_w = results.indicated_work * cycles_per_second
        indicated_power_kw = indicated_power_w / 1000.0
        indicated_power_hp = indicated_power_kw * 1.34102  # kW → HP

        imep_bar = results.imep / 1.0e5

        mean_piston_speed = 2.0 * self.stroke * (self.rpm / 60.0)  # m/s

        displacement_liters = self.displacement_volume * 1000.0  # m³ → L
        specific_power = (
            indicated_power_kw / displacement_liters if displacement_liters > 0 else 0.0
        )  # kW/L

        # Theoretical Otto-cycle efficiency using γ at intake conditions
        gamma_intake = results.gamma[0]
        theoretical_efficiency = 1.0 - (
            1.0 / (self.compression_ratio ** (gamma_intake - 1.0))
        )

        # Mean torque over positive-torque region
        positive_torque = results.torque[results.torque > 0.0]
        mean_torque_nm = (
            float(np.mean(positive_torque)) if len(positive_torque) else 0.0
        )

        return {
            "indicated_work_j": results.indicated_work,
            "imep_pa": results.imep,
            "imep_bar": imep_bar,
            "indicated_power_kw": indicated_power_kw,
            "indicated_power_hp": indicated_power_hp,
            "thermal_efficiency": results.thermal_efficiency,
            "theoretical_efficiency": theoretical_efficiency,
            "heat_added_j": results.heat_added,
            "heat_rejected_j": results.heat_rejected,
            "mean_piston_speed_ms": mean_piston_speed,
            "displacement_liters": displacement_liters,
            "specific_power_kw_per_liter": specific_power,
            "peak_pressure_mpa": float(np.max(results.pressure)) / 1.0e6,
            "peak_temperature_k": float(np.max(results.temperature)),
            "peak_torque_nm": float(np.max(results.torque)),
            "mean_torque_nm": mean_torque_nm,
        }
