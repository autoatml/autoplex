from autoplex.data.md.utils import generate_temperature_profile
import numpy as np
import pytest

def test_generate_temperature_profile_values():
    temperature_list = [100.0, 300.0]
    eqm_step_list = [2, 3]
    rate_list = [2.0]

    T_array, n_steps = generate_temperature_profile(
        temperature_list=temperature_list,
        eqm_step_list=eqm_step_list,
        rate_list=rate_list,
    )

    assert isinstance(T_array, np.ndarray)
    assert isinstance(n_steps, int)

    assert T_array[0] == 100.0
    assert T_array[-1] == 300.0

    assert n_steps == len(T_array) - 1
    
    assert n_steps == 405 #10**13 K/s -> 200 ps per 100 K

def test_single_temperature_returns_array():
    T_array, n_steps = generate_temperature_profile([300.0], [5], [0])
    np.testing.assert_array_equal(T_array, [300.0] * 6)
    assert n_steps == 5

@pytest.mark.parametrize("rate_list", [None, [], [0], [0, 0]])
def test_only_zero_rates_mean_no_ramps(rate_list):
    T_array, _ = generate_temperature_profile([300.0, 500.0, 400.0], [2, 2, 2], rate_list)
    np.testing.assert_array_equal(T_array, [300, 300, 500, 500, 400, 400, 400]) #last 400 is end point

def test_ramp_plateau_ramp():
    T_array, n_steps = generate_temperature_profile(
        [300, 450, 450, 600], eqm_step_list=[0, 500, 0, 100], rate_list=[1, 0, 1]
    )
    assert n_steps == 900 #150(=ramp)+500+0(=ramp)+150(=ramp)+100
    np.testing.assert_array_equal(T_array[149:650], 450.0)
    np.testing.assert_array_equal(T_array[799:], 600.0)

def test_extra_rate_raises():
    with pytest.raises(ValueError, match="rate_list"):
        # len(rate_list)=len(temperature_list)-1
        generate_temperature_profile([300.0, 500.0, 400.0], [5, 5, 5], [0, 2, 1])

@pytest.mark.parametrize(
    ("temperature_list", "eqm_step_list", "rate_list", "ramp_step_list", "n_steps"),
    [
        # 300 -> 500 K jumps; 500 -> 400 K: 100 K x 2 steps/K = 200 steps + 2+10+3 eq steps
        pytest.param([300.0, 500.0, 400.0], [2, 10, 3], [0, 2], [0, 200], 215, id="jump-then-cooling-ramp"),
        # then 150 K x 1 step/K = 150 steps, a plateau at 450 K, then again 150 steps
        pytest.param(
            [300.0, 450.0, 450.0, 600.0], [0, 500, 0, 100], [1, 0, 1], [150, 0, 150], 900, id="ramp-plateau-ramp"
        ),
        # 1500 K x 0.1 steps/K = 150 steps; 1200 K x 0.05 steps/K = 60 steps
        pytest.param([3000.0, 1500.0, 300.0], [5, 5, 5], [0.1, 0.05], [150, 60], 225, id="two-stage-quench"),
        # 700 K x 0.5 steps/K = 350 steps, up and down
        pytest.param([300.0, 1000.0, 300.0], [3, 3, 3], [0.5, 0.5], [350, 350], 709, id="heating-cooling-cycle"),
    ],
)
def test_rate_list_and_ramp_step_list_give_the_same_profile(
    temperature_list, eqm_step_list, rate_list, ramp_step_list, n_steps
):
    """A ramp over dT kelvin at `rate` MD steps per kelvin takes dT * rate MD steps."""
    # rate (MD steps per K) x temperature change (K) = MD steps of each ramp
    np.testing.assert_allclose(
        np.abs(np.diff(temperature_list)) * np.asarray(rate_list), ramp_step_list
    )

    by_rate, n_by_rate = generate_temperature_profile(
        temperature_list, eqm_step_list, rate_list=rate_list
    )
    by_steps, n_by_steps = generate_temperature_profile(
        temperature_list, eqm_step_list, ramp_step_list=ramp_step_list
    )

    # the same temperature at every MD step, and the same number of steps
    np.testing.assert_array_equal(by_steps, by_rate)
    assert n_by_steps == n_by_rate == n_steps == sum(eqm_step_list) + sum(ramp_step_list)


def test_ramp_step_list_is_exact():
    # with rate = 61 / 7 the rate-based length would round down to 60 steps
    T_array, n_steps = generate_temperature_profile([300.0, 307.0], [0, 0], ramp_step_list=[61])
    assert n_steps == 61
    assert T_array[-1] == 307.0


def test_ramp_step_list_is_used_instead_of_the_rate():
    _, n_steps = generate_temperature_profile([300.0, 500.0], [2, 2], [500.0], ramp_step_list=[10])
    assert n_steps == 2 + 10 + 2


@pytest.mark.parametrize("ramp_step_list", [[1, 2], [-5]])
def test_invalid_ramp_step_list_raises(ramp_step_list):
    with pytest.raises(ValueError, match="ramp_step_list"):
        generate_temperature_profile([300.0, 500.0], [2, 2], ramp_step_list=ramp_step_list)
