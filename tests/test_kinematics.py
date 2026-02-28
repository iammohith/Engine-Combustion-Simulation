"""
Unit Tests for Kinematics Module
Tests slider-crank mechanism calculations.

Author: Mohith Sai Gorla
Date:   27-02-2026
"""

import pytest
import math
import numpy as np
from engine_simulator.kinematics import SliderCrank, ValveTiming


class TestSliderCrank:
    """Test cases for SliderCrank class."""

    def setup_method(self):
        """Set up typical automotive engine dimensions (86 mm stroke)."""
        self.stroke = 0.086
        self.crank_radius = self.stroke / 2.0
        self.rod_length = 0.143
        self.sc = SliderCrank(self.crank_radius, self.rod_length)

    # ── Construction guards ───────────────────────────────────────────────

    def test_initialization(self):
        assert self.sc.r == self.crank_radius
        assert self.sc.l == self.rod_length
        assert self.sc.lambda_ratio == pytest.approx(
            self.crank_radius / self.rod_length, rel=1e-12
        )

    def test_invalid_zero_crank_radius(self):
        with pytest.raises(ValueError):
            SliderCrank(0.0, 0.143)

    def test_invalid_rod_shorter_than_crank(self):
        """Connecting rod shorter than crank radius causes lockup."""
        with pytest.raises(ValueError):
            SliderCrank(0.1, 0.05)

    # ── Displacement ──────────────────────────────────────────────────────

    def test_displacement_at_tdc(self):
        assert abs(self.sc.displacement(0.0)) < 1e-12

    def test_displacement_at_bdc(self):
        y_bdc = self.sc.displacement(math.pi)
        assert abs(y_bdc - self.stroke) < 1e-10

    def test_displacement_range(self):
        for deg in range(0, 361):
            theta = math.radians(deg)
            y = self.sc.displacement(theta)
            assert (
                0.0 <= y <= self.stroke * 1.001
            ), f"Displacement {y*1e3:.4f} mm out of range at theta={deg} deg"

    def test_displacement_symmetry(self):
        """y(theta) = y(-theta): displacement is an even function."""
        for deg in [30, 60, 90, 120, 150]:
            y1 = self.sc.displacement(math.radians(deg))
            y2 = self.sc.displacement(math.radians(-deg))
            assert abs(y1 - y2) < 1e-10

    # ── Velocity ──────────────────────────────────────────────────────────

    def test_velocity_at_tdc_is_zero(self):
        assert abs(self.sc.velocity(0.0, 100.0)) < 1e-12

    def test_velocity_at_bdc_is_zero(self):
        assert abs(self.sc.velocity(math.pi, 100.0)) < 1e-8

    def test_velocity_at_zero_omega_is_zero(self):
        for deg in [0, 45, 90, 180, 270]:
            v = self.sc.velocity(math.radians(deg), omega=0.0)
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_velocity_near_max_at_90_degrees(self):
        omega = 100.0
        v_90 = self.sc.velocity(math.pi / 2.0, omega)
        assert abs(v_90) > self.crank_radius * omega * 0.9

    # ── Acceleration ──────────────────────────────────────────────────────

    def test_acceleration_at_tdc(self):
        """a(TDC) = r·omega^2·(1+lambda) — maximum and positive."""
        omega = 314.16
        a_tdc = self.sc.acceleration(0.0, omega)
        a_theory = self.crank_radius * omega**2 * (1.0 + self.sc.lambda_ratio)
        assert a_tdc > 0.0
        assert abs(a_tdc - a_theory) < a_theory * 0.01

    def test_acceleration_zero_omega_zero_alpha(self):
        assert self.sc.acceleration(math.pi / 4, omega=0.0, alpha=0.0) == 0.0

    # ── Connecting rod angle ──────────────────────────────────────────────

    def test_connecting_rod_angle_at_tdc(self):
        assert abs(self.sc.connecting_rod_angle(0.0)) < 1e-12

    def test_connecting_rod_angle_at_90_degrees(self):
        phi = self.sc.connecting_rod_angle(math.pi / 2.0)
        phi_exp = math.asin(self.sc.lambda_ratio)
        assert abs(phi - phi_exp) < 1e-10

    # ── Torque factor ─────────────────────────────────────────────────────

    def test_torque_factor_at_tdc_approx_zero(self):
        assert abs(self.sc.force_to_torque_factor(0.0)) < 1e-3

    def test_torque_factor_positive_at_90_degrees(self):
        assert self.sc.force_to_torque_factor(math.pi / 2.0) > 0.0


class TestValveTiming:
    """Test cases for ValveTiming class."""

    def setup_method(self):
        self.vt = ValveTiming()

    def test_default_angles_valid(self):
        assert self.vt.IVC > self.vt.IVO

    def test_intake_open_during_intake_stroke(self):
        assert self.vt.is_intake_open(400.0)
        assert self.vt.is_intake_open(500.0)

    def test_intake_closed_during_compression(self):
        assert not self.vt.is_intake_open(600.0)
        assert not self.vt.is_intake_open(700.0)

    def test_exhaust_open_during_exhaust_stroke(self):
        assert self.vt.is_exhaust_open(200.0)
        assert self.vt.is_exhaust_open(300.0)

    def test_exhaust_closed_during_intake(self):
        assert not self.vt.is_exhaust_open(450.0)

    def test_cycle_phases_correct(self):
        assert self.vt.get_cycle_phase(90.0) == "EXPANSION"
        assert self.vt.get_cycle_phase(270.0) == "EXHAUST"
        assert self.vt.get_cycle_phase(450.0) == "INTAKE"
        assert self.vt.get_cycle_phase(630.0) == "COMPRESSION"

    def test_valve_overlap_period(self):
        """Both valves open during overlap window around TDC exhaust (360 deg)."""
        overlap_angle = 360.0
        assert self.vt.is_intake_open(overlap_angle)
        assert self.vt.is_exhaust_open(overlap_angle)

    def test_intake_lift_zero_when_closed(self):
        assert self.vt.intake_lift_profile(300.0) == 0.0

    def test_intake_lift_peak_at_mid_stroke(self):
        max_lift = 0.010
        mid = (self.vt.IVO + self.vt.IVC) / 2.0
        lift_mid = self.vt.intake_lift_profile(mid, max_lift)
        assert lift_mid > max_lift * 0.99

    def test_exhaust_lift_zero_when_closed(self):
        assert self.vt.exhaust_lift_profile(450.0) == 0.0

    def test_exhaust_lift_nonnegative(self):
        for deg in np.arange(0, 720, 2):
            lift = self.vt.exhaust_lift_profile(float(deg))
            assert lift >= 0.0, f"Negative exhaust lift at theta={deg} deg"


class TestKinematicsIntegration:
    """Integration tests for kinematics module."""

    def test_velocity_periodicity(self):
        r, L, omega = 0.043, 0.143, 100.0
        sc = SliderCrank(r, L)
        for deg in [0, 45, 90, 180, 270, 360]:
            theta = math.radians(deg)
            v1 = sc.velocity(theta, omega)
            v2 = sc.velocity(theta + 2 * math.pi, omega)
            assert abs(v1 - v2) < 1e-10

    def test_numerical_derivative_consistency(self):
        """Analytical velocity matches (dy/dtheta)*omega from finite differences.

        TEST-01 fix: correct formula is (y2 - y1) / dtheta * omega
        (differentiate in *angle* domain then multiply by omega).
        The original code used a time step dt and divided (y2-y1)/dt which
        mixed the angle and time domains, producing wrong units.
        """
        r, L, omega = 0.043, 0.143, 100.0
        sc = SliderCrank(r, L)
        dtheta = 1.0e-6  # rad
        theta = math.pi / 4.0

        y1 = sc.displacement(theta)
        y2 = sc.displacement(theta + dtheta)

        v_numerical = (y2 - y1) / dtheta * omega  # dy/dtheta × omega
        v_analytical = sc.velocity(theta, omega)

        assert abs(v_numerical - v_analytical) < abs(v_analytical) * 0.001

    def test_acceleration_numerical_derivative(self):
        """Analytical acceleration matches numerical derivative of velocity."""
        r, L, omega = 0.043, 0.143, 314.16
        sc = SliderCrank(r, L)
        dtheta = 1.0e-6
        theta = math.pi / 6.0

        v1 = sc.velocity(theta, omega)
        v2 = sc.velocity(theta + dtheta, omega)

        a_numerical = (v2 - v1) / dtheta * omega
        a_analytical = sc.acceleration(theta, omega)

        # First-order FD introduces ~O(dtheta) error; at dtheta=1e-6, ~0.17% < 0.5%
        assert abs(a_numerical - a_analytical) < abs(a_analytical) * 0.005

    def test_displacement_matches_manual_formula(self):
        """Spot-check exact formula at several angles."""
        r, L = 0.043, 0.143
        lam = r / L
        sc = SliderCrank(r, L)
        for deg in [0, 45, 90, 135, 180, 270]:
            theta = math.radians(deg)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            D = max(0.0, 1.0 - lam**2 * sin_t**2)
            y_manual = r * ((1.0 - cos_t) + (1.0 / lam) * (1.0 - math.sqrt(D)))
            y_code = sc.displacement(theta)
            assert abs(y_code - y_manual) < 1e-12, f"Mismatch at theta={deg} deg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
