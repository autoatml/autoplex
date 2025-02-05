import os
from pathlib import Path
from jobflow import run_locally, Flow

from tests.conftest import mock_rss, mock_do_rss_iterations, mock_do_rss_iterations_multi_jobs
from autoplex.settings import RssConfig
from autoplex.auto.rss.flows import RssMaker

os.environ["OMP_NUM_THREADS"] = "1"


def test_mock_workflow(test_dir, mock_vasp, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    # atoms = read(test_files_dir, index=':')
    # structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    ref_paths = {
        **{f"static_bulk_{i}": f"rss/Si_bulk_{i + 1}/" for i in range(18)},
        "static_isolated_0": "rss/Si_isolated_1/",
        "static_dimer_0": "rss/Si_dimer_1/",
        "static_dimer_1": "rss/Si_dimer_2/",
        "static_dimer_2": "rss/Si_dimer_3/",
    }

    fake_run_vasp_kwargs = {
        "static_isolated_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_1": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_2": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job1 = mock_rss(input_dir=test_files_dir,
                    selection_method='cur',
                    num_of_selection=18,
                    bcur_params={'soap_paras': {'l_max': 3,
                                                'n_max': 3,
                                                'atom_sigma': 0.5,
                                                'cutoff': 4.0,
                                                'cutoff_transition_width': 1.0,
                                                'zeta': 4.0,
                                                'average': True,
                                                'species': True,
                                                },
                                 },
                    random_seed=42,
                    e0_spin=True,
                    isolated_atom=True,
                    dimer=False,
                    dimer_range=None,
                    dimer_num=None,
                    custom_incar={
                        "ADDGRID": None,
                        "ENCUT": 200,
                        "EDIFF": 1E-04,
                        "ISMEAR": 0,
                        "SIGMA": 0.05,
                        "PREC": "Normal",
                        "ISYM": None,
                        "KSPACING": 0.3,
                        "NPAR": 8,
                        "LWAVE": "False",
                        "LCHARG": "False",
                        "ENAUG": None,
                        "GGA": None,
                        "ISPIN": None,
                        "LAECHG": None,
                        "LELF": None,
                        "LORBIT": None,
                        "LVTOT": None,
                        "NSW": None,
                        "SYMPREC": None,
                        "NELM": 50,
                        "LMAXMIX": None,
                        "LASPH": None,
                        "AMIN": None,
                    },
                    vasp_ref_file='vasp_ref.extxyz',
                    gap_rss_group='initial',
                    test_ratio=0.1,
                    regularization=True,
                    distillation=True,
                    f_max=0.7,
                    pre_database_dir=None,
                    mlip_type='GAP',
                    ref_energy_name="REF_energy",
                    ref_force_name="REF_forces",
                    ref_virial_name="REF_virial",
                    num_processes_fit=4,
                    kt=0.6
                    )

    job2 = mock_do_rss_iterations(input=job1.output,
                                  input_dir=test_files_dir,
                                  selection_method1='cur',
                                  selection_method2='bcur1s',
                                  num_of_selection1=5,
                                  num_of_selection2=3,
                                  bcur_params={'soap_paras': {'l_max': 3,
                                                              'n_max': 3,
                                                              'atom_sigma': 0.5,
                                                              'cutoff': 4.0,
                                                              'cutoff_transition_width': 1.0,
                                                              'zeta': 4.0,
                                                              'average': True,
                                                              'species': True,
                                                              },
                                               'frac_of_bcur': 0.8,
                                               'bolt_max_num': 3000,
                                               'kernel_exp': 4.0,
                                               'energy_label': 'energy'},
                                  random_seed=None,
                                  e0_spin=False,
                                  isolated_atom=False,
                                  dimer=False,
                                  dimer_range=None,
                                  dimer_num=None,
                                  custom_incar=None,
                                  vasp_ref_file='vasp_ref.extxyz',
                                  rss_group='initial',
                                  test_ratio=0.1,
                                  regularization=True,
                                  distillation=True,
                                  f_max=200,
                                  pre_database_dir=None,
                                  mlip_type='GAP',
                                  ref_energy_name="REF_energy",
                                  ref_force_name="REF_forces",
                                  ref_virial_name="REF_virial",
                                  num_processes_fit=None,
                                  scalar_pressure_method='exp',
                                  scalar_exp_pressure=100,
                                  scalar_pressure_exponential_width=0.2,
                                  scalar_pressure_low=0,
                                  scalar_pressure_high=50,
                                  max_steps=100,
                                  force_tol=0.6,
                                  stress_tol=0.6,
                                  Hookean_repul=False,
                                  write_traj=True,
                                  num_processes_rss=4,
                                  device="cpu",
                                  stop_criterion=0.01,
                                  max_iteration_number=9
                                  )

    response = run_locally(
        Flow([job1, job2]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    assert Path(job1.output["mlip_path"][0].resolve(memory_jobstore)).exists()

    selected_atoms = job2.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 3


def test_mock_workflow_multi_node(test_dir, mock_vasp, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    # atoms = read(test_files_dir, index=':')
    # structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    ref_paths = {
        **{f"static_bulk_{i}": f"rss/Si_bulk_{i + 1}/" for i in range(18)},
        "static_isolated_0": "rss/Si_isolated_1/",
        "static_dimer_0": "rss/Si_dimer_1/",
        "static_dimer_1": "rss/Si_dimer_2/",
        "static_dimer_2": "rss/Si_dimer_3/",
    }

    fake_run_vasp_kwargs = {
        "static_isolated_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_1": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_2": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job1 = mock_rss(input_dir=test_files_dir,
                    selection_method='cur',
                    num_of_selection=18,
                    bcur_params={'soap_paras': {'l_max': 3,
                                                'n_max': 3,
                                                'atom_sigma': 0.5,
                                                'cutoff': 4.0,
                                                'cutoff_transition_width': 1.0,
                                                'zeta': 4.0,
                                                'average': True,
                                                'species': True,
                                                },
                                 },
                    random_seed=42,
                    e0_spin=True,
                    isolated_atom=True,
                    dimer=False,
                    dimer_range=None,
                    dimer_num=None,
                    custom_incar={
                        "ADDGRID": None,
                        "ENCUT": 200,
                        "EDIFF": 1E-04,
                        "ISMEAR": 0,
                        "SIGMA": 0.05,
                        "PREC": "Normal",
                        "ISYM": None,
                        "KSPACING": 0.3,
                        "NPAR": 8,
                        "LWAVE": "False",
                        "LCHARG": "False",
                        "ENAUG": None,
                        "GGA": None,
                        "ISPIN": None,
                        "LAECHG": None,
                        "LELF": None,
                        "LORBIT": None,
                        "LVTOT": None,
                        "NSW": None,
                        "SYMPREC": None,
                        "NELM": 50,
                        "LMAXMIX": None,
                        "LASPH": None,
                        "AMIN": None,
                    },
                    vasp_ref_file='vasp_ref.extxyz',
                    gap_rss_group='initial',
                    test_ratio=0.1,
                    regularization=True,
                    distillation=True,
                    f_max=0.7,
                    pre_database_dir=None,
                    mlip_type='GAP',
                    ref_energy_name="REF_energy",
                    ref_force_name="REF_forces",
                    ref_virial_name="REF_virial",
                    num_processes_fit=4,
                    kt=0.6
                    )

    job2 = mock_do_rss_iterations_multi_jobs(input=job1.output,
                                             input_dir=test_files_dir,
                                             selection_method1='cur',
                                             selection_method2='bcur1s',
                                             num_of_selection1=5,
                                             num_of_selection2=3,
                                             bcur_params={'soap_paras': {'l_max': 3,
                                                                         'n_max': 3,
                                                                         'atom_sigma': 0.5,
                                                                         'cutoff': 4.0,
                                                                         'cutoff_transition_width': 1.0,
                                                                         'zeta': 4.0,
                                                                         'average': True,
                                                                         'species': True,
                                                                         },
                                                          'frac_of_bcur': 0.8,
                                                          'bolt_max_num': 3000,
                                                          'kernel_exp': 4.0,
                                                          'energy_label': 'energy'},
                                             random_seed=None,
                                             e0_spin=False,
                                             isolated_atom=True,
                                             dimer=False,
                                             dimer_range=None,
                                             dimer_num=None,
                                             custom_incar=None,
                                             vasp_ref_file='vasp_ref.extxyz',
                                             rss_group='initial',
                                             test_ratio=0.1,
                                             regularization=True,
                                             distillation=True,
                                             f_max=200,
                                             pre_database_dir=None,
                                             mlip_type='GAP',
                                             ref_energy_name="REF_energy",
                                             ref_force_name="REF_forces",
                                             ref_virial_name="REF_virial",
                                             num_processes_fit=None,
                                             scalar_pressure_method='exp',
                                             scalar_exp_pressure=100,
                                             scalar_pressure_exponential_width=0.2,
                                             scalar_pressure_low=0,
                                             scalar_pressure_high=50,
                                             max_steps=100,
                                             force_tol=0.6,
                                             stress_tol=0.6,
                                             Hookean_repul=False,
                                             write_traj=True,
                                             num_processes_rss=4,
                                             device="cpu",
                                             stop_criterion=0.01,
                                             max_iteration_number=9,
                                             num_groups=2,
                                             remove_traj_files=True,
                                             )

    response = run_locally(
        Flow([job1, job2]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    assert Path(job1.output["mlip_path"][0].resolve(memory_jobstore)).exists()

    selected_atoms = job2.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 3

def test_rssmaker_custom_config_file(test_dir):

    config_model = RssConfig.from_file(test_dir / "rss" / "rss_config.yaml")

    # Test if config is updated as expected
    rss = RssMaker(rss_config=config_model)

    assert rss.rss_config.tag == "test"
    assert rss.rss_config.generated_struct_numbers == [9000, 1000]
    assert rss.rss_config.num_processes_buildcell == 64
    assert rss.rss_config.num_processes_fit == 64
    assert rss.rss_config.device_for_rss == "cuda"
    assert rss.rss_config.isolatedatom_box == [10, 10, 10]
    assert rss.rss_config.dimer_box == [10, 10, 10]

