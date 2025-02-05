"""Testing configurations to test flows with mock VASP runs.

The following code has been taken and modified from
https://github.com/materialsproject/atomate2/blob/main/tests/conftest.py
https://github.com/materialsproject/atomate2/tree/main/tests/vasp
The code has been released under BSD 3-Clause License
and the following copyright applies:
atomate2 Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, Union, Optional

import pytest
from pytest import MonkeyPatch
from atomate2.utils.testing.vasp import monkeypatch_vasp

from jobflow import Response, job
from autoplex.data.rss.jobs import do_rss_single_node, do_rss_multi_node
from autoplex.data.common.jobs import sample_data, collect_dft_data, preprocess_data
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.fitting.common.flows import MLIPFitMaker
from jobflow import Flow

logger = logging.getLogger("autoplex")

_VFILES: Final = ("incar", "kpoints", "potcar", "poscar")
_REF_PATHS: Dict[str, Union[str, Path]] = {}
_FAKE_RUN_VASP_KWARGS: Dict[str, dict] = {}


@pytest.fixture(scope="session")
def test_dir():
    from pathlib import Path

    module_dir = Path(__file__).parent.resolve()
    test_dir = module_dir.joinpath("test_data")
    return test_dir.resolve()


@pytest.fixture(scope="session")
def vasp_test_dir(test_dir):
    return test_dir / "vasp"


@pytest.fixture()
def mock_vasp(
        monkeypatch: MonkeyPatch, vasp_test_dir: Path
) -> Generator[Callable[[Any, Any], Any], None, None]:
    """
    This fixture allows one to mock (fake) running VASP.

    It works by monkeypatching (replacing) calls to run_vasp and
    VaspInputSet.write_inputs with versions that will work when the vasp executables or
    POTCAR files are not present.

    The primary idea is that instead of running VASP to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether VASP is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs. Another potential issue is that the POTCAR files
    distributed with VASP are not present on the testing server due to licensing
    constraints. Accordingly, VaspInputSet.write_inputs will fail unless the
    "potcar_spec" option is set to True, in which case a POTCAR.spec file will be
    written instead. This fixture solves both of these issues.

    To use the fixture successfully, the following steps must be followed:
    1. "mock_vasp" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by write_vasp_input_set) and "outputs" (containing the expected
       output files to be produced by run_vasp). These files should reside in a
       subdirectory of "tests/test_data/vasp".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/vasp"
       folder. For example, if your calculation has one job named "static" and the
       reference files are present in "tests/test_data/vasp/Si_static", the dictionary
       would look like: ``{"static": "Si_static"}``.
    4. Optional: create a dictionary mapping each job name to custom keyword arguments
       that will be supplied to fake_run_vasp. This way you can configure which incar
       settings are expected for each job. For example, if your calculation has one job
       named "static" and you wish to validate that "NSW" is set correctly in the INCAR,
       your dictionary would look like ``{"static": {"incar_settings": {"NSW": 0}}``.
    5. Inside the test function, call `mock_vasp(ref_paths, fake_vasp_kwargs)`, where
       ref_paths is the dictionary created in step 3 and fake_vasp_kwargs is the
       dictionary created in step 4.
    6. Run your vasp job after calling `mock_vasp`.

    For examples, see the tests in tests/vasp/makers/core.py.
    """
    yield from monkeypatch_vasp(monkeypatch, vasp_test_dir)


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    new_path = tempfile.mkdtemp()
    os.chdir(new_path)
    yield
    if debug_mode:
        print(f"Tests ran in {new_path}")
    else:
        os.chdir(old_cwd)
        shutil.rmtree(new_path)


@pytest.fixture(scope="session")
def debug_mode():
    return True


@pytest.fixture()
def memory_jobstore():
    from jobflow import JobStore
    from maggma.stores import MemoryStore

    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    store.connect()

    return store

@job
def mock_rss(input_dir: str = None,
             selection_method: str = 'cur',
             num_of_selection: int = 3,
             bcur_params: Optional[str] = None,
             random_seed: int = None,
             e0_spin: bool = False,
             isolated_atom: bool = True,
             dimer: bool = True,
             dimer_range: list = None,
             dimer_num: int = None,
             custom_incar: Optional[str] = None,
             vasp_ref_file: str = 'vasp_ref.extxyz',
             rss_group: str = 'initial',
             test_ratio: float = 0.1,
             regularization: bool = True,
             distillation: bool = True,
             f_max: float = 200,
             pre_database_dir: Optional[str] = None,
             mlip_type: str = 'GAP',
             ref_energy_name: str = "REF_energy",
             ref_force_name: str = "REF_forces",
             ref_virial_name: str = "REF_virial",
             num_processes_fit: int = None,
             kt: float = None,
             **fit_kwargs, ):
    job2 = sample_data(selection_method=selection_method,
                       num_of_selection=num_of_selection,
                       bcur_params=bcur_params,
                       dir=input_dir,
                       random_seed=random_seed)
    job3 = DFTStaticLabelling(e0_spin=e0_spin,
                              isolated_atom=isolated_atom,
                              dimer=dimer,
                              dimer_range=dimer_range,
                              dimer_num=dimer_num,
                              custom_incar=custom_incar,
                              ).make(structures=job2.output)
    job4 = collect_dft_data(vasp_ref_file=vasp_ref_file,
                            rss_group=rss_group,
                            vasp_dirs=job3.output)
    job5 = preprocess_data(test_ratio=test_ratio,
                           regularization=regularization,
                           distillation=distillation,
                           force_max=f_max,
                           vasp_ref_dir=job4.output['vasp_ref_dir'], pre_database_dir=pre_database_dir)
    job6 = MLIPFitMaker(mlip_type=mlip_type,
                        ref_energy_name=ref_energy_name,
                        ref_force_name=ref_force_name,
                        ref_virial_name=ref_virial_name,
                        num_processes_fit=num_processes_fit,
                        apply_data_preprocessing=False,
                        ).make(isolated_atom_energies=job4.output['isolated_atom_energies'],
                               database_dir=job5.output, **fit_kwargs)
    job_list = [job2, job3, job4, job5, job6]

    return Response(
        replace=Flow(job_list),
        output={
            'test_error': job6.output['test_error'],
            'pre_database_dir': job5.output,
            'mlip_path': job6.output['mlip_path'][0],
            'isolated_atom_energies': job4.output['isolated_atom_energies'],
            'current_iter': 0,
            'kt': kt
        },
    )


@job
def mock_do_rss_iterations(input=None,
                           input_dir: str = None,
                           selection_method1: str = 'cur',
                           selection_method2: str = 'bcur1s',
                           num_of_selection1: int = 3,
                           num_of_selection2: int = 5,
                           bcur_params: Optional[str] = None,
                           random_seed: int = None,
                           mlip_type: str = 'GAP',
                           scalar_pressure_method: str = 'exp',
                           scalar_exp_pressure: float = 100,
                           scalar_pressure_exponential_width: float = 0.2,
                           scalar_pressure_low: float = 0,
                           scalar_pressure_high: float = 50,
                           max_steps: int = 10,
                           force_tol: float = 0.1,
                           stress_tol: float = 0.1,
                           Hookean_repul: bool = False,
                           write_traj: bool = True,
                           num_processes_rss: int = 4,
                           device: str = "cpu",
                           stop_criterion: float = 0.01,
                           max_iteration_number: int = 9,
                           **fit_kwargs, ):
    if input is None:
        input = {'test_error': None,
                 'pre_database_dir': None,
                 'mlip_path': None,
                 'isolated_atom_energies': None,
                 'current_iter': None,
                 'kt': 0.6}
    if input['test_error'] is not None and input['test_error'] > stop_criterion and input[
        'current_iter'] < max_iteration_number:
        if input['kt'] > 0.15:
            kt = input['kt'] - 0.1
        else:
            kt = 0.1
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print(f'The error of {current_iter}th iteration:', input['test_error'])

        bcur_params['kt'] = kt

        job2 = sample_data(selection_method=selection_method1,
                           num_of_selection=num_of_selection1,
                           bcur_params=bcur_params,
                           dir=input_dir,
                           random_seed=random_seed)
        job3 = do_rss_single_node(mlip_type=mlip_type,
                                  iteration_index=f'{current_iter}th',
                                  mlip_path=input['mlip_path'],
                                  structures=job2.output,
                                  scalar_pressure_method=scalar_pressure_method,
                                  scalar_exp_pressure=scalar_exp_pressure,
                                  scalar_pressure_exponential_width=scalar_pressure_exponential_width,
                                  scalar_pressure_low=scalar_pressure_low,
                                  scalar_pressure_high=scalar_pressure_high,
                                  max_steps=max_steps,
                                  force_tol=force_tol,
                                  stress_tol=stress_tol,
                                  hookean_repul=Hookean_repul,
                                  write_traj=write_traj,
                                  num_processes_rss=num_processes_rss,
                                  device=device)
        job4 = sample_data(selection_method=selection_method2,
                           num_of_selection=num_of_selection2,
                           bcur_params=bcur_params,
                           traj_path=job3.output,
                           random_seed=random_seed,
                           isolated_atom_energies=input["isolated_atom_energies"])

        job_list = [job2, job3, job4]

        return Response(detour=job_list, output=job4.output)


@job
def mock_do_rss_iterations_multi_jobs(input=None,
                                      input_dir: str = None,
                                      selection_method1: str = 'cur',
                                      selection_method2: str = 'bcur1s',
                                      num_of_selection1: int = 3,
                                      num_of_selection2: int = 5,
                                      bcur_params: Optional[str] = None,
                                      random_seed: int = None,
                                      mlip_type: str = 'GAP',
                                      scalar_pressure_method: str = 'exp',
                                      scalar_exp_pressure: float = 100,
                                      scalar_pressure_exponential_width: float = 0.2,
                                      scalar_pressure_low: float = 0,
                                      scalar_pressure_high: float = 50,
                                      max_steps: int = 10,
                                      force_tol: float = 0.1,
                                      stress_tol: float = 0.1,
                                      Hookean_repul: bool = False,
                                      write_traj: bool = True,
                                      num_processes_rss: int = 4,
                                      device: str = "cpu",
                                      stop_criterion: float = 0.01,
                                      max_iteration_number: int = 9,
                                      num_groups: int = 2,
                                      remove_traj_files: bool = True,
                                      **fit_kwargs, ):
    if input is None:
        input = {'test_error': None,
                 'pre_database_dir': None,
                 'mlip_path': None,
                 'isolated_atom_energies': None,
                 'current_iter': None,
                 'kt': 0.6}
    if input['test_error'] is not None and input['test_error'] > stop_criterion and input[
        'current_iter'] < max_iteration_number:
        if input['kt'] > 0.15:
            kt = input['kt'] - 0.1
        else:
            kt = 0.1
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print(f'The error of {current_iter}th iteration:', input['test_error'])

        bcur_params['kT'] = kt

        job2 = sample_data(selection_method=selection_method1,
                           num_of_selection=num_of_selection1,
                           bcur_params=bcur_params,
                           dir=input_dir,
                           random_seed=random_seed)
        job3 = do_rss_multi_node(mlip_type=mlip_type,
                                 iteration_index=f'{current_iter}th',
                                 mlip_path=input['mlip_path'],
                                 structure=job2.output,
                                 scalar_pressure_method=scalar_pressure_method,
                                 scalar_exp_pressure=scalar_exp_pressure,
                                 scalar_pressure_exponential_width=scalar_pressure_exponential_width,
                                 scalar_pressure_low=scalar_pressure_low,
                                 scalar_pressure_high=scalar_pressure_high,
                                 max_steps=max_steps,
                                 force_tol=force_tol,
                                 stress_tol=stress_tol,
                                 hookean_repul=Hookean_repul,
                                 write_traj=write_traj,
                                 num_processes_rss=num_processes_rss,
                                 device=device,
                                 num_groups=num_groups, )
        job4 = sample_data(selection_method=selection_method2,
                           num_of_selection=num_of_selection2,
                           bcur_params=bcur_params,
                           traj_path=job3.output,
                           random_seed=random_seed,
                           isolated_atom_energies=input["isolated_atom_energies"],
                           remove_traj_files=remove_traj_files)

        job_list = [job2, job3, job4]

        return Response(detour=job_list, output=job4.output)