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
    
    assert n_steps == 405

def test_single_temperature_returns_array():
    T_array, n_steps = generate_temperature_profile([300.0], [5], [0])
    np.testing.assert_array_equal(T_array, [300.0] * 6)
    assert n_steps == 5

@pytest.mark.parametrize("rate_list", [None, [], [0], [0, 0]])
def test_only_zero_rates_mean_no_ramps(rate_list):
    T_array, _ = generate_temperature_profile([300.0, 500.0, 400.0], [2, 2, 2], rate_list)
    np.testing.assert_array_equal(T_array, [300, 300, 500, 500, 400, 400, 400])

def test_ramp_plateau_ramp():
    T_array, n_steps = generate_temperature_profile(
        [300, 450, 450, 600], eqm_step_list=[0, 500, 0, 100], rate_list=[1, 0, 1]
    )
    assert n_steps == 900
    np.testing.assert_array_equal(T_array[149:650], 450.0)
    np.testing.assert_array_equal(T_array[799:], 600.0)

def test_extra_rate_raises():
    with pytest.raises(ValueError, match="rate_list"):
        generate_temperature_profile([300.0, 500.0, 400.0], [5, 5, 5], [0, 2, 1])

