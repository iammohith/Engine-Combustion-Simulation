"""
Comprehensive Thermodynamics Test Suite
Unit, boundary, adversarial, and property-based tests for the
thermodynamics module.

Author: Mohith Sai Gorla
Date:   27-02-2026

Test categories
---------------
TestWorkingFluid          : cv, cp, gamma, u(T), ΔU, T(U) inversion
TestCombustionModel       : burn_fraction, heat_release_rate
TestHeatTransferModel     : Woschni coefficient, heat_loss
TestEngineCycle           : compress, expand, add_heat, work
TestNumericalStability    : adversarial inputs, edge cases
"""

import math
import pytest
import numpy as np

from engine_simulator.thermodynamics import (
    WorkingFluid,
    CombustionModel,
    HeatTransferModel,
    EngineCycle,
    ThermodynamicState,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_state(
    P: float = 101_325.0,
    T: float = 300.0,
    V: float = 1.0e-3,
    mass: float = 1.0e-3,
    fluid: WorkingFluid = None,
) -> ThermodynamicState:
    """Convenience factory for ThermodynamicState."""
    if fluid is None:
        fluid = WorkingFluid()
    return ThermodynamicState(
        pressure=P,
        temperature=T,
        volume=V,
        mass=mass,
        internal_energy=fluid.internal_energy(T, mass),
        gamma=fluid.gamma(T),
    )


# ── WorkingFluid tests ────────────────────────────────────────────────────────


class TestWorkingFluid:

    def setup_method(self):
        """Constant-cv fluid (default air)."""
        self.fluid = WorkingFluid(cv_a0=718.0, gas_constant=287.058)
        """Variable-cv fluid: cv(T) = 700 + 0.1·T  [J/(kg·K)]"""
        self.fluid_var = WorkingFluid(cv_a0=700.0, cv_a1=0.1, gas_constant=287.058)

    # ── Construction guards ───────────────────────────────────────────────

    def test_invalid_gas_constant(self):
        with pytest.raises(ValueError):
            WorkingFluid(gas_constant=-1.0)

    def test_invalid_cv_a0(self):
        with pytest.raises(ValueError):
            WorkingFluid(cv_a0=0.0)

    # ── cv, cp, gamma at reference temperature ────────────────────────────

    def test_cv_constant(self):
        """Constant fluid: cv should be exactly cv_a0."""
        assert self.fluid.cv(300.0) == pytest.approx(718.0, rel=1e-10)
        assert self.fluid.cv(1000.0) == pytest.approx(718.0, rel=1e-10)

    def test_cv_variable(self):
        """Variable cv(T) = 700 + 0.1·T."""
        assert self.fluid_var.cv(300.0) == pytest.approx(730.0, rel=1e-9)
        assert self.fluid_var.cv(1000.0) == pytest.approx(800.0, rel=1e-9)

    def test_cp_equals_cv_plus_R(self):
        """cp(T) = cv(T) + R by Mayer's relation."""
        for T in [300.0, 800.0, 2000.0]:
            expected = self.fluid.cv(T) + self.fluid.R
            assert self.fluid.cp(T) == pytest.approx(expected, rel=1e-12)

    def test_gamma_gt_1(self):
        """γ must always be > 1 (Second Law constraint)."""
        for T in [100.0, 500.0, 3000.0]:
            assert self.fluid.gamma(T) > 1.0

    def test_gamma_air_constant(self):
        """For constant cv=718, R=287: γ = 1005/718 ≈ 1.39972."""
        expected = (718.0 + 287.058) / 718.0
        assert self.fluid.gamma(300.0) == pytest.approx(expected, rel=1e-8)

    def test_cv_at_zero_raises(self):
        with pytest.raises(ValueError):
            self.fluid.cv(0.0)

    def test_cv_negative_raises(self):
        with pytest.raises(ValueError):
            self.fluid.cv(-100.0)

    # ── Internal energy integral (BUG-01 fix) ─────────────────────────────

    def test_specific_internal_energy_constant_cv(self):
        """For constant cv, u(T) = cv·T exactly."""
        T = 500.0
        expected = 718.0 * T
        assert self.fluid.specific_internal_energy(T) == pytest.approx(
            expected, rel=1e-10
        )

    def test_specific_internal_energy_variable_cv(self):
        """u(T) = 700·T + 0.1/2·T² for cv_a0=700, cv_a1=0.1."""
        T = 400.0
        expected = 700.0 * T + 0.1 / 2.0 * T**2
        assert self.fluid_var.specific_internal_energy(T) == pytest.approx(
            expected, rel=1e-10
        )

    def test_internal_energy_total(self):
        """U = m · u(T)."""
        mass = 0.002
        T = 300.0
        expected = mass * self.fluid.specific_internal_energy(T)
        assert self.fluid.internal_energy(T, mass) == pytest.approx(expected, rel=1e-12)

    def test_internal_energy_zero_mass_raises(self):
        with pytest.raises(ValueError):
            self.fluid.internal_energy(300.0, 0.0)

    def test_delta_internal_energy_sign(self):
        """ΔU > 0 for T2 > T1."""
        assert self.fluid.delta_internal_energy(300.0, 400.0, 1.0) > 0.0

    def test_delta_internal_energy_variable_cv(self):
        """ΔU = m·[u(T2)−u(T1)] — cross-check numerical integration."""
        T1, T2, mass = 300.0, 800.0, 0.5

        # Numerical integration reference
        T_vals = np.linspace(T1, T2, 10_000)
        cv_vals = self.fluid_var.cv_a0 + self.fluid_var.cv_a1 * T_vals
        dU_ref = mass * float(np.trapezoid(cv_vals, T_vals))

        dU_calc = self.fluid_var.delta_internal_energy(T1, T2, mass)
        assert dU_calc == pytest.approx(dU_ref, rel=1e-4)

    # ── Temperature inversion (BUG-02 fix) ───────────────────────────────

    def test_temperature_from_internal_energy_roundtrip(self):
        """T(U(T)) == T for constant cv."""
        mass, T_orig = 0.001, 750.0
        U = self.fluid.internal_energy(T_orig, mass)
        T_recovered = self.fluid.temperature_from_internal_energy(
            U, mass, T_guess=T_orig
        )
        assert T_recovered == pytest.approx(T_orig, rel=1e-6)

    def test_temperature_from_internal_energy_variable_cv(self):
        """Roundtrip for variable cv."""
        mass, T_orig = 0.002, 1500.0
        U = self.fluid_var.internal_energy(T_orig, mass)
        T_recovered = self.fluid_var.temperature_from_internal_energy(
            U, mass, T_guess=500.0
        )
        assert T_recovered == pytest.approx(T_orig, rel=1e-5)

    def test_temperature_from_internal_energy_nonpositive_raises(self):
        with pytest.raises(ValueError):
            self.fluid.temperature_from_internal_energy(0.0, 1.0)

    def test_temperature_from_internal_energy_zero_mass_raises(self):
        with pytest.raises(ValueError):
            self.fluid.temperature_from_internal_energy(1000.0, 0.0)


# ── CombustionModel tests ─────────────────────────────────────────────────────


class TestCombustionModel:

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            CombustionModel(combustion_type="unknown")

    # ── Ideal CV ──────────────────────────────────────────────────────────

    def test_ideal_cv_before_tdc(self):
        model = CombustionModel("ideal_cv")
        assert model.burn_fraction(-1.0) == 0.0

    def test_ideal_cv_at_and_after_tdc(self):
        model = CombustionModel("ideal_cv")
        assert model.burn_fraction(0.0) == 1.0
        assert model.burn_fraction(90.0) == 1.0

    # ── Wiebe ─────────────────────────────────────────────────────────────

    def setup_method(self):
        self.wiebe = CombustionModel("wiebe")
        self.wiebe.set_wiebe_parameters(a=5.0, m=2.0, start_deg=0.0, duration_deg=60.0)

    def test_wiebe_before_start(self):
        assert self.wiebe.burn_fraction(-1.0) == pytest.approx(0.0, abs=1e-12)

    def test_wiebe_at_start(self):
        assert self.wiebe.burn_fraction(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_wiebe_after_end_is_one(self):
        assert self.wiebe.burn_fraction(60.0) == pytest.approx(1.0, abs=1e-9)
        assert self.wiebe.burn_fraction(90.0) == pytest.approx(1.0, abs=1e-9)

    def test_wiebe_monotone(self):
        """Burn fraction must be strictly non-decreasing."""
        angles = np.arange(-5.0, 70.0, 0.5)
        xb = [self.wiebe.burn_fraction(a) for a in angles]
        for i in range(1, len(xb)):
            assert xb[i] >= xb[i - 1], f"Non-monotone at θ={angles[i]:.1f}°"

    def test_wiebe_normalization(self):
        """Integral over the burn window must equal total heat (≈ 1 for unit Q)."""
        model = CombustionModel("wiebe")
        model.set_wiebe_parameters(a=5.0, m=2.0, start_deg=0.0, duration_deg=60.0)
        angles = np.linspace(0.0, 60.0, 10_000)
        rates = [model.heat_release_rate(a, 1.0) for a in angles]
        integral = float(np.trapezoid(rates, angles))
        # Integral of dxb/dtheta over [0, Dtheta] = x_b(Dtheta) - x_b(0) ≈ 1
        assert integral == pytest.approx(model.burn_fraction(60.0), abs=0.02)

    def test_wiebe_invalid_params(self):
        model = CombustionModel("wiebe")
        with pytest.raises(ValueError):
            model.set_wiebe_parameters(a=-1.0, m=2.0, start_deg=0.0, duration_deg=60.0)
        with pytest.raises(ValueError):
            model.set_wiebe_parameters(a=5.0, m=-1.0, start_deg=0.0, duration_deg=60.0)
        with pytest.raises(ValueError):
            model.set_wiebe_parameters(a=5.0, m=2.0, start_deg=0.0, duration_deg=0.0)


# ── HeatTransferModel tests ───────────────────────────────────────────────────


class TestHeatTransferModel:

    def setup_method(self):
        self.ht = HeatTransferModel(bore=0.082, wall_temperature=450.0)

    def test_invalid_bore(self):
        with pytest.raises(ValueError):
            HeatTransferModel(bore=0.0, wall_temperature=450.0)

    def test_invalid_wall_temperature(self):
        with pytest.raises(ValueError):
            HeatTransferModel(bore=0.082, wall_temperature=0.0)

    def test_woschni_positive(self):
        h = self.ht.woschni_coefficient(50.0, 1500.0, 10.0)
        assert h > 0.0

    def test_woschni_zero_velocity_uses_minimum(self):
        """w = 0 → uses min_piston_speed, not zero (BUG-12 fix)."""
        h_zero = self.ht.woschni_coefficient(50.0, 1500.0, 0.0)
        h_min = self.ht.woschni_coefficient(50.0, 1500.0, self.ht.min_piston_speed)
        assert h_zero == pytest.approx(h_min, rel=1e-10)

    def test_woschni_increases_with_pressure(self):
        h_low = self.ht.woschni_coefficient(10.0, 1000.0, 5.0)
        h_high = self.ht.woschni_coefficient(50.0, 1000.0, 5.0)
        assert h_high > h_low

    def test_woschni_invalid_pressure(self):
        with pytest.raises(ValueError):
            self.ht.woschni_coefficient(0.0, 1000.0, 5.0)

    def test_woschni_invalid_temperature(self):
        with pytest.raises(ValueError):
            self.ht.woschni_coefficient(10.0, 0.0, 5.0)

    def test_heat_loss_positive_when_gas_hotter_than_wall(self):
        loss = self.ht.heat_loss(
            pressure=2_000_000.0,
            temperature=1500.0,
            surface_area=1e-3,
            mean_piston_speed=10.0,
            time_step=1e-4,
        )
        assert loss > 0.0, "Heat should flow from hot gas to cooler wall"

    def test_heat_loss_negative_time_step_raises(self):
        with pytest.raises(ValueError):
            self.ht.heat_loss(2e6, 1500.0, 1e-3, 10.0, time_step=-1e-4)


# ── EngineCycle tests ─────────────────────────────────────────────────────────


class TestEngineCycle:

    def setup_method(self):
        self.fluid = WorkingFluid()
        self.combustion = CombustionModel("ideal_cv")
        self.cycle = EngineCycle(self.fluid, self.combustion)

    def _state(
        self, T: float, V: float, P: float = None, mass: float = 1.0e-3
    ) -> ThermodynamicState:
        if P is None:
            # Ideal gas: P = mRT/V
            P = mass * self.fluid.R * T / V
        return ThermodynamicState(
            pressure=P,
            temperature=T,
            volume=V,
            mass=mass,
            internal_energy=self.fluid.internal_energy(T, mass),
            gamma=self.fluid.gamma(T),
        )

    # ── Isentropic compression (BUG-03 fix) ──────────────────────────────

    def test_compress_isentropic_tdc_bdc(self):
        """Compression to clearance volume should raise temperature."""
        V1 = 500.0e-6  # 500 cc
        V2 = 50.0e-6  # 50 cc (CR = 10)
        s1 = self._state(T=298.15, V=V1)
        s2 = self.cycle.compress_isentropic(s1, V2)
        assert s2.temperature > s1.temperature

    def test_compress_isentropic_pressure_increases(self):
        V1, V2 = 500.0e-6, 50.0e-6
        s1 = self._state(T=298.15, V=V1)
        s2 = self.cycle.compress_isentropic(s1, V2)
        assert s2.pressure > s1.pressure

    def test_compress_expand_roundtrip(self):
        """Isentropic compress then expand must return to original state."""
        V1, V2 = 500.0e-6, 50.0e-6
        s1 = self._state(T=298.15, V=V1)
        s2 = self.cycle.compress_isentropic(s1, V2)
        s3 = self.cycle.expand_isentropic(s2, V1)
        assert s3.temperature == pytest.approx(s1.temperature, rel=1e-4)
        assert s3.pressure == pytest.approx(s1.pressure, rel=1e-4)

    def test_compress_isentropic_invalid_volume(self):
        s = self._state(300.0, 5e-4)
        with pytest.raises(ValueError):
            self.cycle.compress_isentropic(s, 0.0)

    def test_compress_isentropic_matches_constant_gamma_formula(self):
        """For constant cv, must match the analytic P1·V1^γ = P2·V2^γ."""
        V1, V2 = 1.0e-3, 1.0e-4
        T1, P1 = 300.0, 101_325.0
        s1 = self._state(T=T1, V=V1, P=P1)

        gamma = self.fluid.gamma(T1)
        T2_theory = T1 * (V1 / V2) ** (gamma - 1.0)
        P2_theory = P1 * (V1 / V2) ** gamma

        s2 = self.cycle.compress_isentropic(s1, V2)
        # Constant cv → iteration converges in 1–2 steps; within 0.5%
        assert s2.temperature == pytest.approx(T2_theory, rel=5e-3)
        assert s2.pressure == pytest.approx(P2_theory, rel=5e-3)

    # ── Constant-volume heat addition (BUG-02 fix) ────────────────────────

    def test_add_heat_first_law(self):
        """ΔU = Q at constant volume (First Law)."""
        Q = 500.0  # J
        V = 5.0e-5
        s1 = self._state(T=600.0, V=V)
        s2 = self.cycle.add_heat_constant_volume(s1, Q)
        delta_U = s2.internal_energy - s1.internal_energy
        assert delta_U == pytest.approx(Q, rel=1e-5)

    def test_add_heat_volume_constant(self):
        """Volume must not change in constant-volume heat addition."""
        V = 5.0e-5
        s1 = self._state(T=600.0, V=V)
        s2 = self.cycle.add_heat_constant_volume(s1, 200.0)
        assert s2.volume == pytest.approx(V, rel=1e-12)

    def test_add_heat_temperature_increases(self):
        s1 = self._state(T=600.0, V=5e-5)
        s2 = self.cycle.add_heat_constant_volume(s1, 100.0)
        assert s2.temperature > s1.temperature

    def test_add_heat_pressure_from_ideal_gas(self):
        """P2/P1 = T2/T1 at constant volume (ideal gas)."""
        V = 5.0e-5
        s1 = self._state(T=600.0, V=V)
        s2 = self.cycle.add_heat_constant_volume(s1, 300.0)
        P_ratio_expected = s2.temperature / s1.temperature
        P_ratio_computed = s2.pressure / s1.pressure
        assert P_ratio_computed == pytest.approx(P_ratio_expected, rel=1e-4)

    def test_add_excessive_heat_raises(self):
        """Removing more energy than available should raise."""
        s1 = self._state(T=300.0, V=5e-5, mass=1e-3)
        # U_s1 ≈ 718 × 300 × 0.001 = 215.4 J; removing 300 J is impossible
        with pytest.raises(ValueError):
            self.cycle.add_heat_constant_volume(s1, -300.0)

    def test_reject_heat_is_consistent_with_add_heat(self):
        """reject_heat(Q) == add_heat(-Q)."""
        V = 5.0e-5
        s1 = self._state(T=1500.0, V=V)
        Q = 100.0
        s_add = self.cycle.add_heat_constant_volume(s1, -Q)
        s_reject = self.cycle.reject_heat_constant_volume(s1, Q)
        assert s_add.temperature == pytest.approx(s_reject.temperature, rel=1e-10)

    # ── PdV work ──────────────────────────────────────────────────────────

    def test_calculate_work_pdv_positive_for_expansion(self):
        """For a power cycle, ∮ P dV should be positive."""
        V = np.linspace(1e-4, 5e-4, 100)
        P = 1.0e6 / V  # Isothermal P~1/V (positive net work)
        W = self.cycle.calculate_work_pdv(P, V)
        assert W > 0.0

    def test_calculate_work_pdv_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.cycle.calculate_work_pdv(np.ones(5), np.ones(6))

    def test_calculate_work_pdv_too_short_raises(self):
        with pytest.raises(ValueError):
            self.cycle.calculate_work_pdv(np.array([1.0]), np.array([1.0]))


# ── ThermodynamicState validation ─────────────────────────────────────────────


class TestThermodynamicState:

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            ThermodynamicState(
                pressure=1e5,
                temperature=-1.0,
                volume=1e-3,
                mass=1e-3,
                internal_energy=1.0,
                gamma=1.4,
            )

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError):
            ThermodynamicState(
                pressure=1e5,
                temperature=300.0,
                volume=0.0,
                mass=1e-3,
                internal_energy=1.0,
                gamma=1.4,
            )

    def test_gamma_le_1_raises(self):
        with pytest.raises(ValueError):
            ThermodynamicState(
                pressure=1e5,
                temperature=300.0,
                volume=1e-3,
                mass=1e-3,
                internal_energy=1.0,
                gamma=0.9,
            )

    def test_negative_pressure_raises(self):
        with pytest.raises(ValueError):
            ThermodynamicState(
                pressure=-1.0,
                temperature=300.0,
                volume=1e-3,
                mass=1e-3,
                internal_energy=1.0,
                gamma=1.4,
            )


# ── Numerical stability adversarial tests ─────────────────────────────────────


class TestNumericalStability:

    def test_isentropic_very_high_cr(self):
        """Compression ratio of 22 (diesel) must not overflow."""
        fluid = WorkingFluid()
        model = CombustionModel("ideal_cv")
        cycle = EngineCycle(fluid, model)
        V1 = 1.0e-3
        V2 = V1 / 22.0
        s1 = ThermodynamicState(
            pressure=101_325.0,
            temperature=323.15,
            volume=V1,
            mass=1.0e-3,
            internal_energy=fluid.internal_energy(323.15, 1.0e-3),
            gamma=fluid.gamma(323.15),
        )
        s2 = cycle.compress_isentropic(s1, V2)
        assert s2.temperature > 800.0  # diesel ignition range
        assert math.isfinite(s2.temperature)
        assert math.isfinite(s2.pressure)

    def test_lambda_close_to_zero(self):
        """Very small λ (very long rod) must not crash."""
        from engine_simulator.kinematics import SliderCrank

        r = 0.04
        L = 1.0  # λ = 0.04 — very long rod
        sc = SliderCrank(r, L)
        for deg in range(0, 361, 10):
            theta = math.radians(deg)
            y = sc.displacement(theta)
            assert math.isfinite(y)

    def test_wiebe_large_a(self):
        """Large a → sharp combustion; burn fraction at end ≈ 1."""
        model = CombustionModel("wiebe")
        model.set_wiebe_parameters(a=50.0, m=2.0, start_deg=0.0, duration_deg=10.0)
        assert model.burn_fraction(10.0) == pytest.approx(1.0, abs=1e-6)

    def test_internal_energy_inversion_far_initial_guess(self):
        """Newton should converge even when T_guess is far from the root."""
        fluid = WorkingFluid()
        mass, T_true = 0.001, 2000.0
        U_target = fluid.internal_energy(T_true, mass)
        T_recovered = fluid.temperature_from_internal_energy(
            U_target, mass, T_guess=100.0
        )
        assert T_recovered == pytest.approx(T_true, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
