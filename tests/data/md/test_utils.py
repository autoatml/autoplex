from autoplex.data.md.utils import generate_temperature_profile
import numpy as np

def test_generate_temperature_profile_values():
    temperature_list = [100.0, 300.0]
    eqm_step_list = [2, 3]
    rate_list = [2.0]

    T_array, n_steps = generate_temperature_profile(
        temperature_list=temperature_list,
        eqm_step_list=eqm_step_list,
        rate_list=rate_list,
        time_step=1.0,
    )

    assert isinstance(T_array, np.ndarray)
    assert isinstance(n_steps, int)

    assert T_array[0] == 100.0
    assert T_array[-1] == 300.0

    assert n_steps == len(T_array) - 1
    
    assert n_steps == 405
