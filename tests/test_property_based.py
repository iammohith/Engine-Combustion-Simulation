import math
from hypothesis import given, settings
from hypothesis.strategies import floats

from engine_simulator.kinematics import SliderCrank
from engine_simulator.thermodynamics import CombustionModel

# Property tests for numerical stability


@given(
    theta=floats(min_value=-2 * math.pi, max_value=2 * math.pi),
    r=floats(min_value=0.01, max_value=0.1),
    L=floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=1000)
def test_displacement_bounds(theta, r, L):
    # Protect against invalid rod ratios
    if L <= r:
        return

    sc = SliderCrank(r, L)

    disp = sc.displacement(theta)

    # Must be between [0, 2*r (stroke)] due to trigonometry bounds
    # Add a small epsilon for floating point inaccuracy
    assert disp >= -1e-12
    assert disp <= (2 * r + 1e-12)


@given(
    theta=floats(min_value=-2 * math.pi, max_value=2 * math.pi),
    r=floats(min_value=0.01, max_value=0.1),
    L=floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=1000)
def test_torque_factor_bounds(theta, r, L):
    if L <= r:
        return

    sc = SliderCrank(r, L)
    K = sc.force_to_torque_factor(theta)

    # K(theta) should be finite and bounded approximately by r * (1 + r/l)
    max_K = r * (1.0 + r / L)
    assert abs(K) <= max_K + 1e-10


@given(
    pct_duration=floats(min_value=0.0, max_value=1.5),
    a=floats(min_value=0.1, max_value=10.0),
    m=floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=1000)
def test_wiebe_mass_fraction_bounds(pct_duration, a, m):
    """
    Test the stability of the Wiebe function substitution.
    """
    cm = CombustionModel(combustion_type="wiebe")

    # Using start=-10, duration=40 means theta range is -10 to 30.
    start = -10.0
    duration = 40.0
    cm.set_wiebe_parameters(a=a, m=m, start_deg=start, duration_deg=duration)

    theta = start + pct_duration * duration

    bf = cm.burn_fraction(theta)

    # Burn fraction should always be rigorously bounded [0, 1] for all valid inputs
    # even when math.exp() would underflow/lose precision.
    assert bf >= -1e-12
    assert bf <= 1.0 + 1e-12
