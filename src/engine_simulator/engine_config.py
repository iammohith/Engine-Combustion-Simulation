"""
Engine Configuration Module
Defines engine specifications and simulation parameters.

Author: Mohith Sai Gorla
Date:   27-02-2026
"""

import math
import json
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

# ── Enumerations ──────────────────────────────────────────────────────────────


class EngineType(Enum):
    """Supported engine configurations."""

    INLINE = "inline"
    V_ENGINE = "v_engine"
    FLAT = "flat"


class CombustionType(Enum):
    """Combustion model types."""

    IDEAL_CONSTANT_VOLUME = "ideal_cv"
    WIEBE = "wiebe"


class UnitSystem(Enum):
    """Unit system for calculations."""

    SI = "si"
    IMPERIAL = "imperial"


# ── Geometry ──────────────────────────────────────────────────────────────────


@dataclass
class GeometryParameters:
    """Engine geometry specifications (all in SI units).

    Attributes
    ----------
    bore                  : m
    stroke                : m
    connecting_rod_length : m
    compression_ratio     : dimensionless  (> 1)
    num_cylinders         : int            (≥ 1)
    """

    bore: float
    stroke: float
    connecting_rod_length: float
    compression_ratio: float
    num_cylinders: int = 1

    def __post_init__(self) -> None:
        if self.bore <= 0.0:
            raise ValueError(f"bore must be > 0 m, got {self.bore}")
        if self.stroke <= 0.0:
            raise ValueError(f"stroke must be > 0 m, got {self.stroke}")
        if self.connecting_rod_length <= 0.0:
            raise ValueError(
                f"connecting_rod_length must be > 0 m, got {self.connecting_rod_length}"
            )
        _r = self.stroke / 2.0
        if self.connecting_rod_length <= _r:
            raise ValueError(
                f"connecting_rod_length ({self.connecting_rod_length} m) must be > "
                f"crank_radius ({_r} m); otherwise slider-crank mechanism locks up."
            )
        if self.compression_ratio <= 1.0:
            raise ValueError(
                f"compression_ratio must be > 1, got {self.compression_ratio}"
            )
        if self.num_cylinders < 1:
            raise ValueError(f"num_cylinders must be ≥ 1, got {self.num_cylinders}")

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def crank_radius(self) -> float:
        """Crank radius  r = stroke / 2  [m]."""
        return self.stroke / 2.0

    @property
    def rod_ratio(self) -> float:
        """Rod ratio  L/r = connecting_rod_length / crank_radius  [dimensionless].

        Typical automotive range: 3.0 – 4.5.
        """
        return self.connecting_rod_length / self.crank_radius

    @property
    def lambda_ratio(self) -> float:
        """Crank-to-rod ratio  λ = r / L = 1 / rod_ratio  [dimensionless].

        Used directly in slider-crank equations.
        """
        return self.crank_radius / self.connecting_rod_length

    @property
    def piston_area(self) -> float:
        """Piston cross-sectional area  A = π·D²/4  [m²]."""
        return math.pi * self.bore**2 / 4.0

    @property
    def displacement_per_cylinder(self) -> float:
        """Swept (displacement) volume per cylinder  Vd = A·stroke  [m³]."""
        return self.piston_area * self.stroke

    @property
    def total_displacement(self) -> float:
        """Total engine displacement  [m³]."""
        return self.displacement_per_cylinder * self.num_cylinders

    @property
    def clearance_volume(self) -> float:
        """Clearance (TDC) volume  Vc = Vd / (CR − 1)  [m³].

        Derivation:
            CR = (Vc + Vd) / Vc  →  Vc = Vd / (CR − 1)
        """
        # compression_ratio > 1 is enforced in __post_init__
        return self.displacement_per_cylinder / (self.compression_ratio - 1.0)

    @property
    def total_volume_at_bdc(self) -> float:
        """Total cylinder volume at BDC  Vt = Vd + Vc  [m³]."""
        return self.displacement_per_cylinder + self.clearance_volume


# ── Thermodynamics ────────────────────────────────────────────────────────────


@dataclass
class ThermodynamicParameters:
    """Thermodynamic specifications for the working fluid and initial conditions.

    Note (BUG-11 fix)
    -----------------
    The cv/cp/gamma methods that previously lived here have been removed.
    Use ``thermodynamics.WorkingFluid`` — constructed from the coefficients
    below — as the single authoritative source for fluid properties.
    """

    intake_pressure: float = 101_325.0  # Pa   (standard atmosphere)
    intake_temperature: float = 298.15  # K    (25 °C)
    peak_combustion_temperature: float = 2500.0  # K    (typical SI target)
    gas_constant: float = 287.058  # J/(kg·K)  dry air

    # Polynomial cv coefficients:  cv(T) = cv_a0 + cv_a1·T + cv_a2·T²  [J/(kg·K)]
    cv_a0: float = 718.0  # constant term
    cv_a1: float = 0.0  # linear coefficient
    cv_a2: float = 0.0  # quadratic coefficient

    def __post_init__(self) -> None:
        if self.intake_temperature <= 0.0:
            raise ValueError(
                f"intake_temperature must be > 0 K, got {self.intake_temperature}"
            )
        if self.intake_pressure <= 0.0:
            raise ValueError(
                f"intake_pressure must be > 0 Pa, got {self.intake_pressure}"
            )
        if self.gas_constant <= 0.0:
            raise ValueError(
                f"gas_constant must be > 0 J/(kg·K), got {self.gas_constant}"
            )
        if self.cv_a0 <= 0.0:
            raise ValueError(f"cv_a0 must be > 0 J/(kg·K), got {self.cv_a0}")


# ── Combustion ────────────────────────────────────────────────────────────────


@dataclass
class CombustionParameters:
    """Combustion model and heat-transfer parameters."""

    combustion_type: CombustionType = CombustionType.IDEAL_CONSTANT_VOLUME

    # Wiebe function parameters (used when combustion_type == WIEBE)
    wiebe_a: float = 5.0  # Efficiency parameter (a = 5 → 99.3% burnout)
    wiebe_m: float = 2.0  # Shape factor
    combustion_start_btdc: float = 20.0  # degrees BTDC
    combustion_duration: float = 50.0  # degrees

    # Heat transfer
    enable_heat_transfer: bool = False
    wall_temperature: float = 450.0  # K

    def __post_init__(self) -> None:
        if self.wiebe_a <= 0.0:
            raise ValueError(f"wiebe_a must be > 0, got {self.wiebe_a}")
        if self.wiebe_m < 0.0:
            raise ValueError(f"wiebe_m must be ≥ 0, got {self.wiebe_m}")
        if self.combustion_duration <= 0.0:
            raise ValueError(
                f"combustion_duration must be > 0°, got {self.combustion_duration}"
            )
        if self.wall_temperature <= 0.0:
            raise ValueError(
                f"wall_temperature must be > 0 K, got {self.wall_temperature}"
            )


# ── Multi-cylinder ────────────────────────────────────────────────────────────


@dataclass
class MultiCylinderParameters:
    """Multi-cylinder engine configuration."""

    engine_type: EngineType = EngineType.INLINE
    bank_angle: float = 0.0  # degrees (for V-engines)
    firing_order: List[int] = field(default_factory=lambda: [1])
    crank_phases: List[float] = field(default_factory=lambda: [0.0])  # degrees

    @staticmethod
    def get_default_firing_order(
        num_cylinders: int, engine_type: EngineType
    ) -> List[int]:
        """Standard firing orders for common configurations."""
        defaults: Dict = {
            (4, EngineType.INLINE): [1, 3, 4, 2],
            (6, EngineType.INLINE): [1, 5, 3, 6, 2, 4],
            (6, EngineType.V_ENGINE): [1, 2, 3, 4, 5, 6],
            (8, EngineType.V_ENGINE): [1, 8, 4, 3, 6, 5, 7, 2],
        }
        return defaults.get(
            (num_cylinders, engine_type),
            list(range(1, num_cylinders + 1)),
        )

    @staticmethod
    def get_default_crank_phases(
        num_cylinders: int, engine_type: EngineType
    ) -> List[float]:
        """Standard crank phase offsets [degrees] for common configurations."""
        known: Dict = {
            (4, EngineType.INLINE): [0.0, 180.0, 180.0, 0.0],
            (6, EngineType.INLINE): [0.0, 120.0, 240.0, 240.0, 120.0, 0.0],
            (6, EngineType.V_ENGINE): [0.0, 120.0, 240.0, 0.0, 120.0, 240.0],
            (8, EngineType.V_ENGINE): [
                0.0,
                90.0,
                270.0,
                180.0,
                270.0,
                180.0,
                90.0,
                0.0,
            ],
        }
        if (num_cylinders, engine_type) in known:
            return known[(num_cylinders, engine_type)]
        # Default: evenly spaced over 720° (4-stroke cycle)
        interval = 720.0 / num_cylinders
        return [i * interval for i in range(num_cylinders)]


# ── Operating conditions ──────────────────────────────────────────────────────


@dataclass
class OperatingConditions:
    """Engine operating conditions."""

    rpm: float = 3000.0

    def __post_init__(self) -> None:
        if self.rpm <= 0.0:
            raise ValueError(f"rpm must be > 0, got {self.rpm}")

    @property
    def angular_velocity(self) -> float:
        """Angular velocity  ω = 2π·N/60  [rad/s]."""
        return self.rpm * 2.0 * math.pi / 60.0

    @property
    def cycle_time(self) -> float:
        """Duration of one complete 4-stroke cycle  t = 2 rev / (N/60)  [s]."""
        return 120.0 / self.rpm


# ── Simulation parameters ─────────────────────────────────────────────────────


@dataclass
class SimulationParameters:
    """Simulation runtime parameters."""

    angular_resolution: float = 1.0  # degrees per step
    num_cycles: int = 1
    enable_animation: bool = False
    enable_heat_transfer: bool = False

    def __post_init__(self) -> None:
        if self.angular_resolution <= 0.0:
            raise ValueError(
                f"angular_resolution must be > 0°, got {self.angular_resolution}"
            )
        if self.num_cycles < 1:
            raise ValueError(f"num_cycles must be ≥ 1, got {self.num_cycles}")

    @property
    def total_degrees(self) -> float:
        """Total crank-angle span  [degrees]."""
        return 720.0 * self.num_cycles

    @property
    def num_steps(self) -> int:
        """Number of simulation steps (exclusive end)."""
        return int(round(self.total_degrees / self.angular_resolution))


# ── Top-level configuration ───────────────────────────────────────────────────


@dataclass
class EngineConfiguration:
    """Complete engine configuration.

    All sub-configurations are validated individually upon construction.
    Cross-parameter consistency is checked in __post_init__.
    """

    geometry: GeometryParameters
    thermodynamics: ThermodynamicParameters
    combustion: CombustionParameters
    multi_cylinder: MultiCylinderParameters
    operating: OperatingConditions
    simulation: SimulationParameters
    unit_system: UnitSystem = UnitSystem.SI

    def __post_init__(self) -> None:
        self._validate_cross_parameters()

    def _validate_cross_parameters(self) -> None:
        """Enforce cross-dataclass consistency constraints."""
        errors: List[str] = []
        notices: List[str] = []

        # Compression ratio typical range
        cr = self.geometry.compression_ratio
        if not (6.0 <= cr <= 25.0):
            notices.append(f"Compression ratio {cr:.1f} outside typical range [6, 25]")

        # Rod ratio
        rr = self.geometry.rod_ratio
        if not (2.0 <= rr <= 10.0):
            notices.append(f"Rod ratio {rr:.2f} outside typical range [2, 10]")

        # RPM
        rpm = self.operating.rpm
        if not (100 <= rpm <= 20_000):
            notices.append(f"RPM {rpm:.0f} outside typical range [100, 20000]")

        # Multi-cylinder phases length
        n_cyl = self.geometry.num_cylinders
        n_phase = len(self.multi_cylinder.crank_phases)
        if n_phase != n_cyl:
            errors.append(
                f"crank_phases length {n_phase} must equal num_cylinders {n_cyl}"
            )

        n_fo = len(self.multi_cylinder.firing_order)
        if n_fo != n_cyl:
            errors.append(
                f"firing_order length {n_fo} must equal num_cylinders {n_cyl}"
            )

        if errors:
            raise ValueError("EngineConfiguration errors: " + "; ".join(errors))

        if notices:
            import warnings

            for msg in notices:
                warnings.warn(msg, stacklevel=3)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialise configuration to a plain dictionary."""
        return {
            "geometry": {
                "bore": self.geometry.bore,
                "stroke": self.geometry.stroke,
                "connecting_rod_length": self.geometry.connecting_rod_length,
                "compression_ratio": self.geometry.compression_ratio,
                "num_cylinders": self.geometry.num_cylinders,
            },
            "thermodynamics": {
                "intake_pressure": self.thermodynamics.intake_pressure,
                "intake_temperature": self.thermodynamics.intake_temperature,
                "peak_combustion_temperature": self.thermodynamics.peak_combustion_temperature,
                "gas_constant": self.thermodynamics.gas_constant,
                "cv_a0": self.thermodynamics.cv_a0,
                "cv_a1": self.thermodynamics.cv_a1,
                "cv_a2": self.thermodynamics.cv_a2,
            },
            "combustion": {
                "combustion_type": self.combustion.combustion_type.value,
                "wiebe_a": self.combustion.wiebe_a,
                "wiebe_m": self.combustion.wiebe_m,
                "combustion_start_btdc": self.combustion.combustion_start_btdc,
                "combustion_duration": self.combustion.combustion_duration,
                "enable_heat_transfer": self.combustion.enable_heat_transfer,
                "wall_temperature": self.combustion.wall_temperature,
            },
            "multi_cylinder": {
                "engine_type": self.multi_cylinder.engine_type.value,
                "bank_angle": self.multi_cylinder.bank_angle,
                "firing_order": self.multi_cylinder.firing_order,
                "crank_phases": self.multi_cylinder.crank_phases,
            },
            "operating": {"rpm": self.operating.rpm},
            "simulation": {
                "angular_resolution": self.simulation.angular_resolution,
                "num_cycles": self.simulation.num_cycles,
                "enable_animation": self.simulation.enable_animation,
                "enable_heat_transfer": self.simulation.enable_heat_transfer,
            },
            "unit_system": self.unit_system.value,
        }

    def to_json(self, filepath: str) -> None:
        """Persist configuration to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "EngineConfiguration":
        """Load configuration from a JSON file.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.
        KeyError
            If a required field is missing from the JSON.
        ValueError
            If a field has an invalid value.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        try:
            geo_data = data["geometry"]
            thm_data = data["thermodynamics"]
            comb_data = data["combustion"]
            mc_data = data["multi_cylinder"]
            op_data = data["operating"]
            sim_data = data["simulation"]
        except KeyError as exc:
            raise KeyError(f"Missing section in configuration file: {exc}") from exc

        # JSON serialises int/bool as number — cast explicitly to correct types.
        geo_data["num_cylinders"] = int(geo_data["num_cylinders"])

        comb_type_str = comb_data.pop("combustion_type")
        comb_data["enable_heat_transfer"] = bool(
            comb_data.get("enable_heat_transfer", False)
        )

        mc_type_str = mc_data.pop("engine_type")
        # firing_order and crank_phases are lists — JSON preserves them correctly.

        sim_data["num_cycles"] = int(sim_data.get("num_cycles", 1))
        sim_data["enable_animation"] = bool(sim_data.get("enable_animation", False))
        sim_data["enable_heat_transfer"] = bool(
            sim_data.get("enable_heat_transfer", False)
        )

        return cls(
            geometry=GeometryParameters(**geo_data),
            thermodynamics=ThermodynamicParameters(**thm_data),
            combustion=CombustionParameters(
                combustion_type=CombustionType(comb_type_str),
                **comb_data,
            ),
            multi_cylinder=MultiCylinderParameters(
                engine_type=EngineType(mc_type_str),
                **mc_data,
            ),
            operating=OperatingConditions(**op_data),
            simulation=SimulationParameters(**sim_data),
            unit_system=UnitSystem(data.get("unit_system", "si")),
        )


# ── Factory functions ─────────────────────────────────────────────────────────


def create_default_inline_4() -> EngineConfiguration:
    """Create a default naturally-aspirated inline-4 engine configuration.

    Based on a representative 1.6 L DOHC engine:
        Bore × Stroke : 82 mm × 90 mm
        CR            : 10.5 : 1
        Rod length    : 150 mm
        Firing order  : 1-3-4-2
    """
    return EngineConfiguration(
        geometry=GeometryParameters(
            bore=0.082,  # 82 mm
            stroke=0.090,  # 90 mm
            connecting_rod_length=0.150,  # 150 mm (rod ratio = 3.33)
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
        operating=OperatingConditions(rpm=3000.0),
        simulation=SimulationParameters(angular_resolution=1.0),
    )


def create_default_v8() -> EngineConfiguration:
    """Create a default naturally-aspirated V8 engine configuration.

    Based on a representative 5.5 L V8:
        Bore × Stroke : 103 mm × 92 mm
        CR            : 11.0 : 1
        Rod length    : 165 mm
        Firing order  : 1-8-4-3-6-5-7-2  (cross-plane)
    """
    return EngineConfiguration(
        geometry=GeometryParameters(
            bore=0.103,
            stroke=0.092,
            connecting_rod_length=0.165,
            compression_ratio=11.0,
            num_cylinders=8,
        ),
        thermodynamics=ThermodynamicParameters(
            peak_combustion_temperature=2800.0,
        ),
        combustion=CombustionParameters(),
        multi_cylinder=MultiCylinderParameters(
            engine_type=EngineType.V_ENGINE,
            bank_angle=90.0,
            firing_order=[1, 8, 4, 3, 6, 5, 7, 2],
            crank_phases=[0.0, 90.0, 270.0, 180.0, 270.0, 180.0, 90.0, 0.0],
        ),
        operating=OperatingConditions(rpm=5000.0),
        simulation=SimulationParameters(angular_resolution=1.0),
    )
