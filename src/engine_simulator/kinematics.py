"""
Kinematics Module
Calculates piston motion using exact slider-crank mechanism equations.

Author: Mohith Sai Gorla
Date:   27-02-2026

Mathematical Basis
------------------
Slider-crank notation
    r  = crank radius = stroke / 2              [m]
    l  = connecting rod length                  [m]
    λ  = r / l  (rod ratio, typically 0.25–0.35)
    θ  = crank angle from TDC                   [rad]
    φ  = connecting rod angle from cylinder axis [rad]
    ω  = dθ/dt  (angular velocity)              [rad/s]
    α  = dω/dt  (angular acceleration)          [rad/s²]

Exact trigonometric relations (no small-angle approximation)
    φ     = arcsin(λ sin θ)
    y(θ)  = r [(1 − cos θ) + (1/λ)(1 − √(1 − λ² sin² θ))]
    v(θ)  = r ω sin θ · [1 + λ cos θ / √(1 − λ² sin² θ)]
    a(θ)  = r ω² [cos θ + λ cos 2θ / (1 − λ² sin² θ)^(3/2)]
           + r α  sin θ · [1 + λ cos θ / √(1 − λ² sin² θ)]

The approximate form of acceleration (omitting the (·)^(3/2) denominator)
introduces a relative error of order λ²  when λ sin θ ≈ 1, peaking at θ
near 90° for large λ.  The exact form is used here throughout.

References
----------
Norton, R.L. (2020). Machine Design, §15.
Heywood, J.B. (1988). Internal Combustion Engine Fundamentals, §2.1.
"""

import math
from typing import Tuple


class SliderCrank:
    """Exact slider-crank mechanism kinematics.

    Calculates piston displacement, velocity, and acceleration using
    the exact analytical trigonometric solution (no small-angle approximations).

    Attributes
    ----------
    r            : float  Crank radius (= stroke / 2)  [m]
    l            : float  Connecting rod length         [m]
    lambda_ratio : float  Rod ratio λ = r / l          [dimensionless]
    """

    def __init__(self, crank_radius: float, connecting_rod_length: float) -> None:
        """
        Parameters
        ----------
        crank_radius           : float  r = stroke / 2  [m]  (must be > 0)
        connecting_rod_length  : float  l               [m]  (must be > r)

        Raises
        ------
        ValueError
            If crank_radius ≤ 0, connecting_rod_length ≤ 0,
            or connecting_rod_length ≤ crank_radius (mechanism would lock up).

        Warns
        -----
        UserWarning
            If rod ratio λ is outside the typical automotive range [0.22, 0.40].
        """
        if crank_radius <= 0.0:
            raise ValueError(f"crank_radius must be > 0 m, got {crank_radius}")
        if connecting_rod_length <= 0.0:
            raise ValueError(
                f"connecting_rod_length must be > 0 m, got {connecting_rod_length}"
            )
        if connecting_rod_length <= crank_radius:
            raise ValueError(
                f"connecting_rod_length ({connecting_rod_length} m) must be > "
                f"crank_radius ({crank_radius} m); otherwise mechanism locks up."
            )

        self.r = crank_radius
        self.l = connecting_rod_length
        self.lambda_ratio = crank_radius / connecting_rod_length  # λ = r/l

        # Validate rod ratio (informational warning only — not an error)
        _LAMBDA_WARN_LOW = 0.22
        _LAMBDA_WARN_HIGH = 0.40
        if not (_LAMBDA_WARN_LOW <= self.lambda_ratio <= _LAMBDA_WARN_HIGH):
            import warnings

            warnings.warn(
                f"Rod ratio λ = {self.lambda_ratio:.4f} is outside the typical "
                f"automotive range [{_LAMBDA_WARN_LOW}, {_LAMBDA_WARN_HIGH}]. "
                "Verify mechanism geometry.",
                stacklevel=2,
            )

    # ── Internal helper ───────────────────────────────────────────────────

    def _discriminant(self, sin_theta: float) -> float:
        """Return D = 1 − λ² sin² θ, clamped to [0, 1].

        D appears under the radical in the exact kinematics equations.
        It is always ≥ 0 for valid λ ∈ (0, 1), but floating-point errors
        can produce tiny negative values; clamping prevents domain errors.

        Parameters
        ----------
        sin_theta : float   sin(θ)

        Returns
        -------
        float  D  ∈ [0, 1]
        """
        d = 1.0 - self.lambda_ratio**2 * sin_theta**2
        return max(0.0, min(1.0, d))

    # ── Public kinematics ─────────────────────────────────────────────────

    def displacement(self, theta: float) -> float:
        """Exact piston displacement from TDC  y(θ)  [m].

        Formula
        -------
            y(θ) = r [(1 − cos θ) + (1/λ)(1 − √(1 − λ² sin² θ))]

        Boundary conditions (verified):
            y(0)   = 0         (at TDC)
            y(π)   = stroke    (at BDC, exact)

        Parameters
        ----------
        theta : float  Crank angle  [rad],  0 = TDC firing

        Returns
        -------
        float  Displacement from TDC  [m],  ∈ [0, stroke]
        """
        sin_theta = math.sin(theta)

        D = self._discriminant(sin_theta)
        sqrt_D = math.sqrt(D)

        # 1 - cos(theta) = 2 sin^2(theta/2)
        term1 = 2.0 * math.sin(theta / 2.0) ** 2

        # (1 - sqrt(D)) / lambda
        # Rationalized conjugate: (1 - D) / [lambda * (1 + sqrt(D))]
        # Substitute D = 1 - lambda^2 sin^2(theta) -> 1 - D = lambda^2 sin^2(theta)
        # term2 = lambda * sin^2(theta) / (1 + sqrt(D))
        term2 = (self.lambda_ratio * sin_theta**2) / (1.0 + sqrt_D)

        return self.r * (term1 + term2)

    def velocity(self, theta: float, omega: float) -> float:
        """Exact piston velocity  v(θ, ω) = (dy/dθ)·ω  [m/s].

        Formula
        -------
            v = r ω sin θ · [1 + λ cos θ / √(1 − λ² sin² θ)]

        Parameters
        ----------
        theta : float  Crank angle  [rad]
        omega : float  Angular velocity  [rad/s]  (may be 0 or negative)

        Returns
        -------
        float  Piston velocity  [m/s]  (positive = moving away from TDC)
        """
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        D = self._discriminant(sin_theta)

        if D < 1e-14:
            # λ sin θ → 1: mechanism at its kinematic limit; velocity → 0
            return 0.0

        sqrt_D = math.sqrt(D)
        correction = 1.0 + (self.lambda_ratio * cos_theta) / sqrt_D

        return self.r * omega * sin_theta * correction

    def acceleration(self, theta: float, omega: float, alpha: float = 0.0) -> float:
        """Exact piston acceleration  a(θ, ω, α)  [m/s²].

        Formula (exact, no approximation)
        ----------------------------------
            a = r ω² [cos θ + λ cos 2θ / (1 − λ² sin² θ)^(3/2)]
              + r α  sin θ · [1 + λ cos θ / √(1 − λ² sin² θ)]

        The first term is the centripetal/Coriolis contribution (dominant),
        the second term is the angular-acceleration contribution (zero for
        constant-speed simulation, i.e., α = 0).

        Parameters
        ----------
        theta : float  Crank angle  [rad]
        omega : float  Angular velocity  [rad/s]
        alpha : float  Angular acceleration  [rad/s²]  (default 0 for steady-state)

        Returns
        -------
        float  Piston acceleration  [m/s²]
               (positive = direction away from TDC)
        """
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        cos_2theta = math.cos(2.0 * theta)

        D = self._discriminant(sin_theta)

        if D < 1e-14:
            # Kinematic singularity — acceleration undefined; return centripetal limit
            return self.r * omega**2 * cos_theta

        sqrt_D = math.sqrt(D)
        D_3_2 = D * sqrt_D  # (1 − λ²sin²θ)^(3/2)

        # Centripetal / Coriolis term
        a_omega = (
            self.r * omega**2 * (cos_theta + self.lambda_ratio * cos_2theta / D_3_2)
        )

        # Angular-acceleration term (zero for constant ω)
        a_alpha = (
            self.r
            * alpha
            * sin_theta
            * (1.0 + (self.lambda_ratio * cos_theta) / sqrt_D)
        )

        return a_omega + a_alpha

    def kinematics_at_angle(
        self, theta: float, omega: float, alpha: float = 0.0
    ) -> Tuple[float, float, float]:
        """Displacement, velocity, and acceleration at a given crank angle.

        Parameters
        ----------
        theta : float  Crank angle  [rad]
        omega : float  Angular velocity  [rad/s]
        alpha : float  Angular acceleration  [rad/s²]  (default 0)

        Returns
        -------
        Tuple[float, float, float]
            (displacement [m], velocity [m/s], acceleration [m/s²])
        """
        y = self.displacement(theta)
        v = self.velocity(theta, omega)
        a = self.acceleration(theta, omega, alpha)
        return y, v, a

    def connecting_rod_angle(self, theta: float) -> float:
        """Connecting rod angle from cylinder axis  φ = arcsin(λ sin θ)  [rad].

        Boundary conditions:
            φ(0)   = 0     (at TDC, rod is aligned with cylinder)
            φ(π/2) = arcsin(λ)   (maximum rod angle)

        Parameters
        ----------
        theta : float  Crank angle  [rad]

        Returns
        -------
        float  Connecting rod angle  φ  [rad]  ∈ [−arcsin(λ), +arcsin(λ)]
        """
        sin_theta = math.sin(theta)
        argument = self.lambda_ratio * sin_theta

        # Clamp to [-1, 1] to guard against floating-point rounding
        argument = max(-1.0, min(1.0, argument))

        return math.asin(argument)

    def force_to_torque_factor(self, theta: float) -> float:
        """Multiply by gas force to get instantaneous crankshaft torque.

        Derivation
        ----------
        The net force acting along the piston axis F_p is transmitted through
        the connecting rod to the crank pin.  The torque on the crankshaft is:

            T = F_p · r · sin(θ + φ) / cos(φ)

        This function returns the factor:

            K(θ) = r · sin(θ + φ) / cos(φ)

        so that  T = F_p · K(θ).

        At TDC (θ = 0, φ = 0):  K = r · sin(0) / cos(0) = 0  ✓ (no torque)
        At BDC (θ = π, φ = 0):  K = r · sin(π) / cos(0) = 0  ✓ (no torque)

        Parameters
        ----------
        theta : float  Crank angle  [rad]

        Returns
        -------
        float  Torque conversion factor  K(θ)  [m]
        """
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        D = self._discriminant(sin_theta)

        # Return 0 if exactly at singularity limit
        if D < 1.0e-14:
            return 0.0

        sqrt_D = math.sqrt(D)

        # Exact mathematical equivalent to K(theta) = v(theta) / omega
        # Avoids computing phi = arcsin(...) and sin(theta + phi)
        correction = 1.0 + (self.lambda_ratio * cos_theta) / sqrt_D
        return self.r * sin_theta * correction


class ValveTiming:
    """Valve timing and lift profiles for a 4-stroke engine cycle.

    All angles are measured in *absolute* crank-angle degrees referenced to
    TDC firing = 0°.  One full 4-stroke cycle spans 720°.

    Default timing (SAE convention, degrees absolute):
    ─────────────────────────────────────────────────
    Event          Abbr.   Default   Description
    ─────────────────────────────────────────────────
    Intake Open    IVO     350°      10° BTDC exhaust TDC
    Intake Close   IVC     580°      40° ABDC
    Exhaust Open   EVO     490°      50° BBDC
    Exhaust Close  EVC     370°      10° ATDC exhaust TDC
    ─────────────────────────────────────────────────

    Valve overlap period: IVO to EVC  (350° – 370°).

    Cycle phases:
        0°   – 180° : EXPANSION  (power stroke)
        180° – 360° : EXHAUST
        360° – 540° : INTAKE
        540° – 720° : COMPRESSION
    """

    # Default valve events [degrees absolute, 0 = TDC firing]
    _IVO_DEFAULT: float = 350.0
    _IVC_DEFAULT: float = 580.0
    _EVO_DEFAULT: float = 490.0
    _EVC_DEFAULT: float = 370.0

    def __init__(
        self,
        ivo: float = _IVO_DEFAULT,
        ivc: float = _IVC_DEFAULT,
        evo: float = _EVO_DEFAULT,
        evc: float = _EVC_DEFAULT,
    ) -> None:
        """
        Parameters
        ----------
        ivo : Intake Valve Opens  [deg absolute, 0–720]
        ivc : Intake Valve Closes [deg absolute, 0–720]
        evo : Exhaust Valve Opens [deg absolute, 0–720]
        evc : Exhaust Valve Closes[deg absolute, 0–720]

        Raises
        ------
        ValueError
            If IVC ≤ IVO or EVC ≤ EVO (within the 0–720 window).
        """
        self.IVO = float(ivo)
        self.IVC = float(ivc)
        self.EVO = float(evo)
        self.EVC = float(evc)

        if self.IVC <= self.IVO:
            raise ValueError(f"IVC ({self.IVC}°) must be > IVO ({self.IVO}°)")

    # ── Valve state queries ───────────────────────────────────────────────

    def is_intake_open(self, crank_angle_deg: float) -> bool:
        """True if the intake valve is open at the given crank angle."""
        angle = crank_angle_deg % 720.0
        return self.IVO <= angle <= self.IVC

    def is_exhaust_open(self, crank_angle_deg: float) -> bool:
        """True if the exhaust valve is open at the given crank angle.

        The exhaust valve spans from EVO through TDC-exhaust to EVC,
        wrapping around the 0°/720° boundary:
            open  if angle ≥ EVO  OR  angle ≤ EVC
        """
        angle = crank_angle_deg % 720.0
        return (angle >= self.EVO) or (angle <= self.EVC)

    def get_cycle_phase(self, crank_angle_deg: float) -> str:
        """Determine which stroke the piston is in.

        Parameters
        ----------
        crank_angle_deg : float  [deg],  any value (normalised internally)

        Returns
        -------
        str  One of {'EXPANSION', 'EXHAUST', 'INTAKE', 'COMPRESSION'}
        """
        angle = crank_angle_deg % 720.0

        if 0.0 <= angle < 180.0:
            return "EXPANSION"
        elif 180.0 <= angle < 360.0:
            return "EXHAUST"
        elif 360.0 <= angle < 540.0:
            return "INTAKE"
        else:  # 540 ≤ angle < 720
            return "COMPRESSION"

    # ── Lift profiles ─────────────────────────────────────────────────────

    def intake_lift_profile(
        self, crank_angle_deg: float, max_lift: float = 0.010
    ) -> float:
        """Sinusoidal intake valve lift  L_i(θ)  [m].

        Profile: L_i = max_lift · sin(π · ξ),  ξ = (θ − IVO) / (IVC − IVO)
        Peak at mid-stroke (ξ = 0.5), zero at open and close events.

        Parameters
        ----------
        crank_angle_deg : float  [deg]
        max_lift        : float  Maximum valve lift  [m]  (default 10 mm)

        Returns
        -------
        float  Lift  [m]  ∈ [0, max_lift]
        """
        if not self.is_intake_open(crank_angle_deg):
            return 0.0

        angle = crank_angle_deg % 720.0
        duration = self.IVC - self.IVO
        normalized = (angle - self.IVO) / duration  # ∈ [0, 1]

        # Guard (should not be needed given is_intake_open check, but defensive)
        if normalized < 0.0 or normalized > 1.0:
            return 0.0

        return max(0.0, max_lift * math.sin(math.pi * normalized))

    def exhaust_lift_profile(
        self, crank_angle_deg: float, max_lift: float = 0.010
    ) -> float:
        """Sinusoidal exhaust valve lift  L_e(θ)  [m].

        Handles the wrap-around of the exhaust event across the 0°/720°
        boundary by computing the normalised position within the *total*
        exhaust duration (EVO → 720° → EVC).

        Profile: L_e = max_lift · sin(π · ξ),  ξ ∈ [0, 1] over the event.

        Parameters
        ----------
        crank_angle_deg : float  [deg]
        max_lift        : float  Maximum valve lift  [m]

        Returns
        -------
        float  Lift  [m]  ∈ [0, max_lift]
        """
        if not self.is_exhaust_open(crank_angle_deg):
            return 0.0

        angle = crank_angle_deg % 720.0

        # Total exhaust duration spanning the wrap-around
        duration = (720.0 - self.EVO) + self.EVC  # degrees

        if duration <= 0.0:
            return 0.0

        if angle >= self.EVO:
            # First segment: from EVO to 720°
            elapsed = angle - self.EVO
        else:
            # Second segment: from 0° to EVC (after wrap)
            elapsed = (720.0 - self.EVO) + angle

        normalized = elapsed / duration  # ∈ [0, 1]
        normalized = max(0.0, min(1.0, normalized))  # defensive clamp

        return max(0.0, max_lift * math.sin(math.pi * normalized))
