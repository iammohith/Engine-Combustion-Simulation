"""
Thermodynamics Module
Implements Otto cycle calculations with variable specific heats.

Author: Mohith Sai Gorla
Date:   27-02-2026

Mathematical Basis
------------------
Working fluid: ideal gas with polynomial cv(T) = a0 + a1·T + a2·T²  [J/(kg·K)]

Specific internal energy (integrated from 0 K):
    u(T) = a0·T + (a1/2)·T² + (a2/3)·T³                           [J/kg]

Isentropic relation (variable γ, iterative):
    T2/T1 = (V1/V2)^(γ_eff - 1)
where γ_eff is the effective ratio that satisfies energy conservation.

Woschni heat-transfer correlation (simplified form):
    h = 3.26 · D^{-0.2} · P^{0.8} · T^{-0.55} · w^{0.8}         [W/(m²·K)]
    with P in bar, T in K, w in m/s.
"""

import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid

# ── Physical constants ───────────────────────────────────────────────────────
_WOSCHNI_C1: float = 3.26  # Woschni correlation pre-factor
_WOSCHNI_MIN_VELOCITY: float = (
    0.5  # m/s — minimum characteristic velocity to avoid w^0.8 = 0
)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThermodynamicState:
    """Immutable snapshot of the thermodynamic state at one crank-angle step.

    All quantities are in SI units.

    Attributes
    ----------
    pressure         : Pa
    temperature      : K   (must be > 0)
    volume           : m³  (must be > 0)
    mass             : kg  (must be > 0)
    internal_energy  : J   = m · u(T)
    gamma            : dimensionless, ratio cp/cv  (must be > 1)
    """

    pressure: float
    temperature: float
    volume: float
    mass: float
    internal_energy: float
    gamma: float

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0 K, got {self.temperature}")
        if self.volume <= 0.0:
            raise ValueError(f"Volume must be > 0 m³, got {self.volume}")
        if self.mass <= 0.0:
            raise ValueError(f"Mass must be > 0 kg, got {self.mass}")
        if self.gamma <= 1.0:
            raise ValueError(f"Gamma must be > 1, got {self.gamma}")
        if self.pressure <= 0.0:
            raise ValueError(f"Pressure must be > 0 Pa, got {self.pressure}")


# ── Working fluid ────────────────────────────────────────────────────────────


class WorkingFluid:
    """Working fluid properties with temperature-dependent specific heats.

    Models the charge as a calorically imperfect ideal gas whose specific
    heat at constant volume follows the polynomial:

        cv(T) = a0 + a1·T + a2·T²    [J/(kg·K)]

    The specific internal energy is obtained by integrating cv from 0 K:

        u(T) = a0·T + (a1/2)·T² + (a2/3)·T³    [J/kg]

    This is the *only* thermodynamically consistent form when cv is a
    polynomial in T (Moran & Shapiro, §3.5).

    The default coefficients represent dry air at low-to-moderate temperatures:
        cv = 718 J/(kg·K)  →  γ ≈ 1.400 at 298 K
    """

    def __init__(
        self,
        gas_constant: float = 287.058,  # J/(kg·K) — CODATA 2018 value
        cv_a0: float = 718.0,
        cv_a1: float = 0.0,
        cv_a2: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        gas_constant : float
            Specific gas constant R = R_universal / M  [J/(kg·K)].
            Default 287.058 J/(kg·K) for dry air (M = 28.966 g/mol).
        cv_a0 : float
            Constant term in cv polynomial  [J/(kg·K)].
        cv_a1 : float
            Linear coefficient  [J/(kg·K²)].
        cv_a2 : float
            Quadratic coefficient  [J/(kg·K³)].

        Raises
        ------
        ValueError
            If gas_constant ≤ 0 or cv_a0 ≤ 0.
        """
        if gas_constant <= 0.0:
            raise ValueError(f"gas_constant must be > 0, got {gas_constant}")
        if cv_a0 <= 0.0:
            raise ValueError(f"cv_a0 must be > 0, got {cv_a0}")

        self.R = gas_constant
        self.cv_a0 = cv_a0
        self.cv_a1 = cv_a1
        self.cv_a2 = cv_a2

    # ── Thermodynamic properties ──────────────────────────────────────────

    def cv(self, temperature: float) -> float:
        """Specific heat at constant volume  cv(T)  [J/(kg·K)].

        Precondition: temperature > 0 K.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0 K, got {temperature}")
        return self.cv_a0 + self.cv_a1 * temperature + self.cv_a2 * temperature**2

    def cp(self, temperature: float) -> float:
        """Specific heat at constant pressure  cp(T) = cv(T) + R  [J/(kg·K)]."""
        return self.cv(temperature) + self.R

    def gamma(self, temperature: float) -> float:
        """Specific heat ratio  γ(T) = cp/cv  [dimensionless].

        Postcondition: result > 1.
        """
        cv_val = self.cv(temperature)
        return (cv_val + self.R) / cv_val

    def specific_internal_energy(self, temperature: float) -> float:
        """Specific internal energy  u(T) = ∫₀ᵀ cv(T') dT'  [J/kg].

        Exact analytical integral of the polynomial cv:
            u(T) = a0·T + (a1/2)·T² + (a2/3)·T³

        This is the thermodynamically consistent form (BUG-01 fix).
        Using u = cv(T)·T is only valid for a constant cv.

        Precondition: temperature > 0 K.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0 K, got {temperature}")
        T = temperature
        return self.cv_a0 * T + (self.cv_a1 / 2.0) * T**2 + (self.cv_a2 / 3.0) * T**3

    def internal_energy(self, temperature: float, mass: float) -> float:
        """Total internal energy  U = m · u(T)  [J].

        Parameters
        ----------
        temperature : float  [K]
        mass        : float  [kg]  must be > 0
        """
        if mass <= 0.0:
            raise ValueError(f"Mass must be > 0 kg, got {mass}")
        return mass * self.specific_internal_energy(temperature)

    def delta_internal_energy(
        self, temperature_1: float, temperature_2: float, mass: float
    ) -> float:
        """Change in total internal energy  ΔU = m·[u(T2) - u(T1)]  [J].

        This is the correct way to compute heat added/removed in a
        constant-volume process, avoiding the cv(T̄)·ΔT approximation.

        Parameters
        ----------
        temperature_1 : float  T₁  [K]
        temperature_2 : float  T₂  [K]
        mass          : float  [kg]
        """
        return mass * (
            self.specific_internal_energy(temperature_2)
            - self.specific_internal_energy(temperature_1)
        )

    def temperature_from_internal_energy(
        self,
        total_internal_energy: float,
        mass: float,
        T_guess: float = 1000.0,
        tolerance: float = 1e-6,
        max_iter: int = 50,
    ) -> float:
        """Invert  U(T) = m · u(T)  to find  T.

        Solves f(T) = m · u(T) - U_target = 0  using the Newton–Raphson
        method.  The analytical derivative f'(T) = m · cv(T) ensures
        quadratic convergence for well-conditioned cases.

        Parameters
        ----------
        total_internal_energy : float  [J]  (must be > 0)
        mass                  : float  [kg]
        T_guess               : float  Initial temperature estimate  [K]
        tolerance             : float  Absolute temperature tolerance  [K]
        max_iter              : int    Maximum Newton iterations

        Returns
        -------
        float  Temperature T such that m·u(T) ≈ U_target  [K]

        Raises
        ------
        RuntimeError
            If Newton iteration does not converge within max_iter steps.
        ValueError
            If total_internal_energy ≤ 0.
        """
        if total_internal_energy <= 0.0:
            raise ValueError(
                f"total_internal_energy must be > 0, got {total_internal_energy}"
            )
        if mass <= 0.0:
            raise ValueError(f"Mass must be > 0 kg, got {mass}")

        U_target = total_internal_energy
        T = max(T_guess, 1.0)  # Protect against T_guess ≤ 0

        for iteration in range(max_iter):
            U_current = mass * self.specific_internal_energy(T)
            f = U_current - U_target
            fp = mass * self.cv(T)  # dU/dT = m·cv(T)

            # Guard against degenerate derivative (cv very small — unphysical)
            if abs(fp) < 1e-30:
                raise RuntimeError(
                    f"Newton iteration: derivative ~0 at T={T:.2f} K (iter={iteration})"
                )

            dT = -f / fp
            T += dT

            # Keep T in physical range
            T = max(T, 1.0)

            if abs(dT) < tolerance:
                return T

        raise RuntimeError(
            f"temperature_from_internal_energy did not converge after {max_iter} iterations. "
            f"Last T={T:.4f} K, ΔT={dT:.4e} K"
        )


# ── Combustion models ────────────────────────────────────────────────────────


class CombustionModel:
    """Heat release models for the combustion process.

    Supports
    --------
    ``'ideal_cv'``
        Ideal constant-volume combustion (instantaneous at TDC, θ = 0°).
    ``'wiebe'``
        Wiebe function model for realistic finite-duration burn rate.

    Wiebe Function
    --------------
    The mass fraction burned follows:

        x_b(θ) = 1 − exp[−a · ((θ − θ_s) / Δθ)^(m+1)]

    where
        θ_s  = combustion start angle  [deg]
        Δθ   = combustion duration     [deg]
        a    = efficiency parameter    (default 5.0 → ~99.3% burn completion)
        m    = shape factor            (default 2.0 → Gaussian-like profile)

    The heat release rate is the analytical derivative:

        dx_b/dθ = [a(m+1)/Δθ] · [(θ−θ_s)/Δθ]^m · exp[−a·((θ−θ_s)/Δθ)^(m+1)]

    Reference: Heywood, Internal Combustion Engine Fundamentals, §9.2.
    """

    def __init__(self, combustion_type: str = "ideal_cv") -> None:
        """
        Parameters
        ----------
        combustion_type : str
            ``'ideal_cv'`` or ``'wiebe'``.

        Raises
        ------
        ValueError
            If combustion_type is not a recognised string.
        """
        _valid = {"ideal_cv", "wiebe"}
        if combustion_type not in _valid:
            raise ValueError(
                f"combustion_type must be one of {_valid}, got '{combustion_type}'"
            )
        self.combustion_type = combustion_type

        # Wiebe parameters
        self.wiebe_a = 5.0  # → ~99.3% burnout
        self.wiebe_m = 2.0
        self.combustion_start_deg = -20.0  # BTDC
        self.combustion_duration_deg = 50.0

    def set_wiebe_parameters(
        self,
        a: float,
        m: float,
        start_deg: float,
        duration_deg: float,
    ) -> None:
        """Set Wiebe function parameters.

        Parameters
        ----------
        a            : Efficiency parameter  (a > 0; a = 5 → 99.3% burnout)
        m            : Shape factor          (m ≥ 0)
        start_deg    : Combustion start angle relative to TDC firing  [deg]
                       Negative = BTDC.
        duration_deg : Combustion duration  [deg]  (must be > 0)

        Raises
        ------
        ValueError
            If parameters are outside physical bounds.
        """
        if a <= 0.0:
            raise ValueError(f"Wiebe 'a' must be > 0, got {a}")
        if m < 0.0:
            raise ValueError(f"Wiebe 'm' must be ≥ 0, got {m}")
        if duration_deg <= 0.0:
            raise ValueError(f"Combustion duration must be > 0°, got {duration_deg}")
        self.wiebe_a = a
        self.wiebe_m = m
        self.combustion_start_deg = start_deg
        self.combustion_duration_deg = duration_deg

    def burn_fraction(self, crank_angle_deg: float) -> float:
        """Mass fraction burned at the given crank angle  x_b ∈ [0, 1].

        Parameters
        ----------
        crank_angle_deg : float
            Crank angle in degrees measured from TDC firing (0°).

        Returns
        -------
        float
            Burned mass fraction in [0, 1].
        """
        if self.combustion_type == "ideal_cv":
            return 1.0 if crank_angle_deg >= 0.0 else 0.0

        # ── Wiebe ────────────────────────────────────────────────────────
        theta = crank_angle_deg - self.combustion_start_deg

        if theta < 0.0:
            return 0.0
        if theta >= self.combustion_duration_deg:
            return 1.0

        normalized = theta / self.combustion_duration_deg
        exponent = -self.wiebe_a * (normalized ** (self.wiebe_m + 1.0))
        # Replace 1.0 - math.exp(exponent) with -math.expm1(exponent) for numerical stability
        return -math.expm1(exponent)

    def heat_release_rate(self, crank_angle_deg: float, total_heat: float) -> float:
        """Instantaneous heat release rate  dQ/dθ  [J/degree].

        Parameters
        ----------
        crank_angle_deg : float  [degrees from TDC firing]
        total_heat      : float  Total heat to be released in cycle  [J]

        Returns
        -------
        float  Heat release rate  [J/deg]
        """
        if self.combustion_type == "ideal_cv":
            # Dirac delta at TDC — implementation hands total heat at θ = 0
            return total_heat if crank_angle_deg == 0.0 else 0.0

        # ── Wiebe ────────────────────────────────────────────────────────
        theta = crank_angle_deg - self.combustion_start_deg

        if theta < 0.0 or theta > self.combustion_duration_deg:
            return 0.0

        normalized = theta / self.combustion_duration_deg

        # dx_b/dθ = [a(m+1)/Δθ] · (θ/Δθ)^m · exp[−a·(θ/Δθ)^(m+1)]
        exponent = -self.wiebe_a * (normalized ** (self.wiebe_m + 1.0))
        derivative = (
            (self.wiebe_a * (self.wiebe_m + 1.0) / self.combustion_duration_deg)
            * (normalized**self.wiebe_m)
            * math.exp(exponent)
        )
        return total_heat * derivative


# ── Heat transfer ────────────────────────────────────────────────────────────


class HeatTransferModel:
    """Convective heat transfer to cylinder walls using the Woschni correlation.

    Simplified Woschni formulation (combustion phase, without pressure-rise term):

        h = C₁ · D^{-0.2} · P^{0.8} · T^{-0.55} · w^{0.8}

    where
        C₁ = 3.26  [Woschni, 1967]
        D  = bore  [m]
        P  = cylinder pressure  [bar]
        T  = gas temperature    [K]
        w  = characteristic velocity = max(mean_piston_speed, w_min)  [m/s]

    Reference: Woschni, G. (1967). SAE Technical Paper 670931.

    Note: The full Woschni correlation includes a pressure-rise term
    (C₂·Vd·T_IVC·ΔP/(P_IVC·V_IVC)) during combustion. This simplified form
    omits that term, which can underestimate h by 20–40% at peak pressure.
    Extending to the full form requires IVC reference conditions.
    """

    def __init__(
        self,
        bore: float,
        wall_temperature: float = 450.0,
        min_piston_speed: float = _WOSCHNI_MIN_VELOCITY,
    ) -> None:
        """
        Parameters
        ----------
        bore             : float  Cylinder bore  [m]  (> 0)
        wall_temperature : float  Cylinder wall temperature  [K]  (> 0)
        min_piston_speed : float  Minimum characteristic velocity  [m/s]
                           Prevents h = 0 at TDC/BDC where mean speed → 0.
                           Default: 0.5 m/s  (BUG-12 fix)

        Raises
        ------
        ValueError
            If bore ≤ 0 or wall_temperature ≤ 0.
        """
        if bore <= 0.0:
            raise ValueError(f"bore must be > 0 m, got {bore}")
        if wall_temperature <= 0.0:
            raise ValueError(f"wall_temperature must be > 0 K, got {wall_temperature}")
        self.bore = bore
        self.wall_temperature = wall_temperature
        self.min_piston_speed = max(min_piston_speed, 0.0)

    def woschni_coefficient(
        self,
        pressure_bar: float,
        temperature_k: float,
        mean_piston_speed: float,
    ) -> float:
        """Woschni convective heat transfer coefficient  h  [W/(m²·K)].

        Parameters
        ----------
        pressure_bar      : float  Cylinder pressure  [bar]  (> 0)
        temperature_k     : float  Gas temperature     [K]   (> 0)
        mean_piston_speed : float  Mean piston speed   [m/s]

        Returns
        -------
        float  Heat transfer coefficient  h  [W/(m²·K)]

        Raises
        ------
        ValueError
            If pressure_bar ≤ 0 or temperature_k ≤ 0.
        """
        if pressure_bar <= 0.0:
            raise ValueError(f"pressure_bar must be > 0, got {pressure_bar}")
        if temperature_k <= 0.0:
            raise ValueError(f"temperature_k must be > 0, got {temperature_k}")

        # Enforce minimum characteristic velocity (BUG-12 fix)
        w = max(abs(mean_piston_speed), self.min_piston_speed)

        h = (
            _WOSCHNI_C1
            * (self.bore**-0.2)
            * (pressure_bar**0.8)
            * (temperature_k**-0.55)
            * (w**0.8)
        )
        return h

    def heat_loss(
        self,
        pressure: float,
        temperature: float,
        surface_area: float,
        mean_piston_speed: float,
        time_step: float,
    ) -> float:
        """Heat loss from gas to walls during one time step  Q_loss  [J].

        Sign convention: returned value is *positive* when heat flows from gas
        to wall (i.e., gas loses energy).

        Parameters
        ----------
        pressure          : float  Cylinder pressure  [Pa]
        temperature       : float  Gas temperature     [K]
        surface_area      : float  Effective heat transfer area  [m²]
        mean_piston_speed : float  Mean piston speed  [m/s]
        time_step         : float  Duration of time step  [s]  (> 0)

        Returns
        -------
        float  Heat lost from gas to wall  [J]  (positive = gas cools)

        Raises
        ------
        ValueError
            If time_step ≤ 0.
        """
        if time_step <= 0.0:
            raise ValueError(f"time_step must be > 0 s, got {time_step}")

        pressure_bar = pressure / 1.0e5  # Pa → bar

        h = self.woschni_coefficient(pressure_bar, temperature, mean_piston_speed)

        # Q = h · A · (T_gas − T_wall) · Δt
        q_dot = h * surface_area * (temperature - self.wall_temperature)
        return q_dot * time_step


# ── Engine cycle ─────────────────────────────────────────────────────────────


class EngineCycle:
    """Single-cylinder thermodynamic cycle calculator.

    Implements the four processes of the Otto cycle:

    1. Isentropic compression   (process 1→2)
    2. Constant-volume heat addition  (combustion, process 2→3)
    3. Isentropic expansion         (power stroke, process 3→4)
    4. Constant-volume heat rejection (exhaust, process 4→1)

    All isentropic processes handle variable γ(T) through an internally
    consistent Newton–Raphson iteration on the *temperature* (not γ).

    Reference: Pulkrabek, Engineering Fundamentals of the ICE, Ch. 3.
    """

    def __init__(
        self,
        working_fluid: WorkingFluid,
        combustion_model: CombustionModel,
        heat_transfer: Optional[HeatTransferModel] = None,
    ) -> None:
        """
        Parameters
        ----------
        working_fluid   : WorkingFluid
        combustion_model: CombustionModel
        heat_transfer   : HeatTransferModel, optional
        """
        self.fluid = working_fluid
        self.combustion = combustion_model
        self.heat_transfer = heat_transfer

    # ── Isentropic processes ──────────────────────────────────────────────

    def compress_isentropic(
        self,
        state_initial: ThermodynamicState,
        volume_final: float,
        tolerance: float = 1e-7,
        max_iter: int = 60,
    ) -> ThermodynamicState:
        """Isentropic compression from state_initial to volume_final.

        Algorithm (BUG-03 fix)
        ----------------------
        For a calorically imperfect ideal gas undergoing an isentropic process,
        the first law (no heat transfer, reversible) gives:

            ΔU = −W  →  m·[u(T2) − u(T1)] = −∫P dV

        For the closed-form integration, we use the isentropic relation
        combined with the ideal gas law.  We iterate directly on T₂ using:

            f(T₂) = u(T₂) − u(T₁) + (R·T₁/γ_eff)·[(V₁/V₂)^(γ_eff−1) − 1]·...

        In practice, a simpler and robust Newton scheme uses:

            T₂_next = T₁ · (V₁/V₂)^(γ_eff(T_mid) − 1)

        where γ_eff is evaluated at T_mid = (T₁ + T₂)/2 and iterated.
        This approach:
          - Converges in ~5–10 iterations for typical engine conditions
          - Satisfies both the energy equation and the ps relation
          - Degenerates exactly to the constant-γ formula when cv is constant

        Parameters
        ----------
        state_initial : ThermodynamicState
        volume_final  : float   [m³] — must be > 0
        tolerance     : float   Temperature convergence criterion  [K]
        max_iter      : int     Maximum iterations

        Returns
        -------
        ThermodynamicState at volume_final

        Raises
        ------
        ValueError
            If volume_final ≤ 0.
        RuntimeError
            If self-consistent iteration does not converge.
        """
        if volume_final <= 0.0:
            raise ValueError(f"volume_final must be > 0 m³, got {volume_final}")

        T1 = state_initial.temperature
        V1 = state_initial.volume
        V2 = volume_final
        V_ratio = V1 / V2  # > 1 for compression, < 1 for expansion

        # ── Self-consistent iteration on T₂ ──────────────────────────────
        # Start with γ at T₁
        T2 = T1 * (V_ratio ** (self.fluid.gamma(T1) - 1.0))
        T2 = max(T2, 1.0)  # physical guard

        for iteration in range(max_iter):
            T_mid = 0.5 * (T1 + T2)
            gamma_eff = self.fluid.gamma(T_mid)
            T2_new = T1 * (V_ratio ** (gamma_eff - 1.0))
            T2_new = max(T2_new, 1.0)

            if abs(T2_new - T2) < tolerance:
                T2 = T2_new
                break
            T2 = T2_new
        else:
            raise RuntimeError(
                f"compress_isentropic did not converge after {max_iter} iterations. "
                f"T1={T1:.2f} K, V_ratio={V_ratio:.4f}, last T2={T2:.4f} K"
            )

        gamma_final = self.fluid.gamma(T2)

        # Pressure from isentropic relation P·V^γ = const, using γ at T_mid
        T_mid_final = 0.5 * (T1 + T2)
        gamma_p = self.fluid.gamma(T_mid_final)
        P2 = state_initial.pressure * (V_ratio**gamma_p)

        # Alternatively enforce ideal gas law for consistency:
        # P2 = state_initial.pressure * (T2/T1) * (V1/V2)  # ideal gas
        # We use the isentropic P relation as it is more physically meaningful.

        U2 = self.fluid.internal_energy(T2, state_initial.mass)

        return ThermodynamicState(
            pressure=P2,
            temperature=T2,
            volume=V2,
            mass=state_initial.mass,
            internal_energy=U2,
            gamma=gamma_final,
        )

    def expand_isentropic(
        self,
        state_initial: ThermodynamicState,
        volume_final: float,
        tolerance: float = 1e-7,
        max_iter: int = 60,
    ) -> ThermodynamicState:
        """Isentropic expansion from state_initial to volume_final.

        Uses the same self-consistent iteration as compress_isentropic.
        The direction (expansion vs. compression) is encoded in V_ratio < 1.
        """
        return self.compress_isentropic(
            state_initial, volume_final, tolerance, max_iter
        )

    # ── Constant-volume heat addition / rejection ─────────────────────────

    def add_heat_constant_volume(
        self,
        state_initial: ThermodynamicState,
        heat_added: float,
        tolerance: float = 1e-7,
        max_iter: int = 60,
    ) -> ThermodynamicState:
        """Add heat at constant volume (combustion process).

        First Law at constant volume:
            ΔU = Q_added
            U₂ = U₁ + Q_added

        The temperature T₂ is found by inverting U₂ = m·u(T₂)
        using Newton–Raphson (via WorkingFluid.temperature_from_internal_energy).
        This is exact for the polynomial u(T) model.  (BUG-02 fix)

        Sign convention: heat_added > 0 → gas gains energy (combustion).

        Parameters
        ----------
        state_initial : ThermodynamicState
        heat_added    : float  Heat added to gas  [J].  May be negative (heat rejection).

        Returns
        -------
        ThermodynamicState after heat addition

        Raises
        ------
        RuntimeError
            If Newton–Raphson inversion does not converge.
        """
        U_new = state_initial.internal_energy + heat_added

        if U_new <= 0.0:
            raise ValueError(
                f"Heat rejection would drive internal energy to ≤ 0 "
                f"(U_initial={state_initial.internal_energy:.2f} J, "
                f"heat_added={heat_added:.2f} J)"
            )

        T_new = self.fluid.temperature_from_internal_energy(
            total_internal_energy=U_new,
            mass=state_initial.mass,
            T_guess=state_initial.temperature,
            tolerance=tolerance,
            max_iter=max_iter,
        )

        # Pressure from ideal gas law at constant volume:
        # P₂/P₁ = T₂/T₁   (constant V, constant m, ideal gas)
        P_new = state_initial.pressure * (T_new / state_initial.temperature)
        gamma_new = self.fluid.gamma(T_new)

        return ThermodynamicState(
            pressure=P_new,
            temperature=T_new,
            volume=state_initial.volume,
            mass=state_initial.mass,
            internal_energy=U_new,
            gamma=gamma_new,
        )

    def reject_heat_constant_volume(
        self,
        state_initial: ThermodynamicState,
        heat_rejected: float,
    ) -> ThermodynamicState:
        """Reject heat at constant volume (exhaust blowdown model).

        Parameters
        ----------
        state_initial  : ThermodynamicState
        heat_rejected  : float  Heat removed from gas  [J]  (positive value).

        Returns
        -------
        ThermodynamicState after heat rejection.
        """
        if heat_rejected < 0.0:
            raise ValueError(
                f"heat_rejected must be ≥ 0 (positive convention), got {heat_rejected}"
            )
        return self.add_heat_constant_volume(state_initial, -heat_rejected)

    # ── Work calculation ──────────────────────────────────────────────────

    def calculate_work_pdv(
        self,
        pressure_array: np.ndarray,
        volume_array: np.ndarray,
    ) -> float:
        """Net indicated work from the P-V diagram  W = ∮ P dV  [J].

        Uses the scipy trapezoidal rule.  Sign convention follows thermodynamics:
            W > 0  →  net work output (power cycle, expansion > compression)
            W < 0  →  net work input  (compressor cycle)

        The cycle integral ∮ P dV is evaluated by integrating the *ordered*
        P-V trace.  For a well-formed power cycle the result is positive.

        Parameters
        ----------
        pressure_array : np.ndarray  [Pa]
        volume_array   : np.ndarray  [m³]

        Returns
        -------
        float  Net work  [J]

        Raises
        ------
        ValueError
            If arrays have different lengths or fewer than 2 elements.
        """
        if len(pressure_array) != len(volume_array):
            raise ValueError(
                f"pressure_array and volume_array must have the same length, "
                f"got {len(pressure_array)} and {len(volume_array)}"
            )
        if len(pressure_array) < 2:
            raise ValueError("Arrays must have at least 2 elements for integration")

        return float(trapezoid(pressure_array, volume_array))
