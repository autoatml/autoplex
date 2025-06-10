import os
from pathlib import Path
from jobflow import run_locally, Flow
from tests.conftest import mock_rss, mock_do_rss_iterations, mock_do_rss_iterations_multi_jobs
from autoplex.settings import RssConfig

from autoplex.auto.rss.flows import RssMaker

os.environ["OMP_NUM_THREADS"] = "1"

def test_rss_workflow(test_dir, mock_vasp, memory_jobstore, clean_dir):
    from autoplex.settings import RssConfig
    from autoplex.auto.rss.flows import RssMaker
    from jobflow import Flow

    from jobflow import run_locally


    # We need this to run the tutorial directly in the jupyter notebook
    ref_paths = {
        "static_bulk_0": "rss_Si_small/static_bulk_0",
        "static_bulk_1": "rss_Si_small/static_bulk_1",
        "static_bulk_2": "rss_Si_small/static_bulk_2",
        "static_bulk_3": "rss_Si_small/static_bulk_3",
        "static_bulk_4": "rss_Si_small/static_bulk_4",
        "static_bulk_5": "rss_Si_small/static_bulk_5",
        "static_bulk_6": "rss_Si_small/static_bulk_6",
        "static_bulk_7": "rss_Si_small/static_bulk_7",
        "static_bulk_8": "rss_Si_small/static_bulk_8",
        "static_bulk_9": "rss_Si_small/static_bulk_9",
        "static_bulk_10": "rss_Si_small/static_bulk_10",
        "static_bulk_11": "rss_Si_small/static_bulk_11",
        "static_bulk_12": "rss_Si_small/static_bulk_12",
        "static_bulk_13": "rss_Si_small/static_bulk_13",
        "static_bulk_14": "rss_Si_small/static_bulk_14",
        "static_bulk_15": "rss_Si_small/static_bulk_15",
        "static_bulk_16": "rss_Si_small/static_bulk_16",
        "static_bulk_17": "rss_Si_small/static_bulk_17",
        "static_bulk_18": "rss_Si_small/static_bulk_18",
        "static_bulk_19": "rss_Si_small/static_bulk_19",
        "static_isolated_0": "rss_Si_small/static_isolated_0",
    }

    fake_run_vasp_kwargs = {
        **{f"static_bulk_{i}": {"incar_settings": ["NSW", "ISMEAR"], "check_inputs": ["incar", "potcar"]} for i in
           range(20)},
        "static_isolated_0": {"incar_settings": ["NSW", "ISMEAR"], "check_inputs": ["incar", "potcar"]},
    }

    rss_config = RssConfig.from_file(test_dir/"rss/rss_si_config.yaml")

    rss_job = RssMaker(name="rss", rss_config=rss_config).make()
    from atomate2.vasp.powerups import update_user_incar_settings
    rss_job=update_user_incar_settings(rss_job, {"NPAR":8})
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    responses=run_locally(
        Flow(jobs=[rss_job], output=rss_job.output),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert rss_job.name == "rss"

def test_rss_workflow_custom_makers(test_dir, mock_vasp, memory_jobstore, clean_dir):
    from autoplex.settings import RssConfig
    from autoplex.auto.rss.flows import RssMaker
    from jobflow import Flow

    from jobflow import run_locally


    # We need this to run the tutorial directly in the jupyter notebook
    ref_paths = {
        "static_bulk_0": "rss_Si_small/static_bulk_0",
        "static_bulk_1": "rss_Si_small/static_bulk_1",
        "static_bulk_2": "rss_Si_small/static_bulk_2",
        "static_bulk_3": "rss_Si_small/static_bulk_3",
        "static_bulk_4": "rss_Si_small/static_bulk_4",
        "static_bulk_5": "rss_Si_small/static_bulk_5",
        "static_bulk_6": "rss_Si_small/static_bulk_6",
        "static_bulk_7": "rss_Si_small/static_bulk_7",
        "static_bulk_8": "rss_Si_small/static_bulk_8",
        "static_bulk_9": "rss_Si_small/static_bulk_9",
        "static_bulk_10": "rss_Si_small/static_bulk_10",
        "static_bulk_11": "rss_Si_small/static_bulk_11",
        "static_bulk_12": "rss_Si_small/static_bulk_12",
        "static_bulk_13": "rss_Si_small/static_bulk_13",
        "static_bulk_14": "rss_Si_small/static_bulk_14",
        "static_bulk_15": "rss_Si_small/static_bulk_15",
        "static_bulk_16": "rss_Si_small/static_bulk_16",
        "static_bulk_17": "rss_Si_small/static_bulk_17",
        "static_bulk_18": "rss_Si_small/static_bulk_18",
        "static_bulk_19": "rss_Si_small/static_bulk_19",
        "static_isolated_0": "rss_Si_small/static_isolated_0",
    }

    fake_run_vasp_kwargs = {
        **{f"static_bulk_{i}": {"incar_settings": ["NSW", "ISMEAR"], "check_inputs": ["incar", "potcar"]} for i in
           range(20)},
        "static_isolated_0": {"incar_settings": ["NSW", "ISMEAR"], "check_inputs": ["incar", "potcar"]},
    }

    rss_config = RssConfig.from_file(test_dir/"rss/rss_si_config.yaml")
    # test the default VASPmakers instead
    rss_config=rss_config.as_dict()
    rss_config["custom_incar"]=None
    rss_config["custom_potcar"]=None

    rss_job = RssMaker(name="rss", rss_config=rss_config).make()
    from atomate2.vasp.powerups import update_user_incar_settings
    rss_job=update_user_incar_settings(rss_job, {"NPAR":8})
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    responses=run_locally(
        Flow(jobs=[rss_job], output=rss_job.output),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert rss_job.name == "rss"


def test_rss_workflow_ml_potentials(test_dir, memory_jobstore, clean_dir):
    from autoplex.settings import RssConfig
    from autoplex.auto.rss.flows import RssMaker
    from jobflow import Flow

    from jobflow import run_locally

    rss_config = {'tag': 'Si', 'train_from_scratch': True,
                  'resume_from_previous_state': {'test_error': None, 'pre_database_dir': None, 'mlip_path': None,
                                                 'isolated_atom_energies': None}, 'generated_struct_numbers': [200],
                  'buildcell_options': [
                      {'ABFIX': False, 'NFORM': '1', 'SYMMOPS': '1-4', 'SYSTEM': None, 'SLACK': 0.25, 'OCTET': False,
                       'OVERLAP': 0.1, 'MINSEP': None, 'NATOM': '{6,8,10,12,14,16,18,20,22,24}'}],
                  'fragment_file': None, 'fragment_numbers': None, 'num_processes_buildcell': 16,
                  'num_of_initial_selected_structs': [20], 'num_of_rss_selected_structs': 1,
                  'initial_selection_enabled': True, 'rss_selection_method': 'bcur2i', 'bcur_params': {
            'soap_paras': {'l_max': 12, 'n_max': 12, 'atom_sigma': 0.0875, 'cutoff': 10.5,
                           'cutoff_transition_width': 1.0, 'zeta': 4.0, 'average': True, 'species': True},
            'frac_of_bcur': 0.8, 'bolt_max_num': 3000}, 'random_seed': None, 'include_isolated_atom': True,
                  'isolatedatom_box': [20.0, 20.0, 20.0], 'e0_spin': False, 'include_dimer': False,
                  'dimer_box': [20.0, 20.0, 20.0], 'dimer_range': [1.0, 5.0], 'dimer_num': 21,
                  'static_energy_maker': None, 'static_energy_maker_isolated_species': None,
                  'static_energy_maker_isolated_species_spin_polarization': None, 'vasp_ref_file': 'vasp_ref.extxyz',
                  'config_types': ['initial', 'traj_early', 'traj'], 'rss_group': ['traj'], 'test_ratio': 0.0,
                  'regularization': True, 'retain_existing_sigma': False, 'scheme': 'linear-hull',
                  'reg_minmax': [[0.1, 1.0], [0.001, 0.1], [0.0316, 0.316], [0.0632, 0.632]], 'distillation': False,
                  'force_max': None, 'force_label': None, 'pre_database_dir': None, 'mlip_type': 'GAP',
                  'ref_energy_name': 'REF_energy', 'ref_force_name': 'REF_forces', 'ref_virial_name': 'REF_virial',
                  'auto_delta': True, 'num_processes_fit': 32, 'device_for_fitting': 'cpu',
                  'scalar_pressure_method': 'uniform', 'scalar_exp_pressure': 1,
                  'scalar_pressure_exponential_width': 0.2, 'scalar_pressure_low': 0, 'scalar_pressure_high': 25,
                  'max_steps': 200, 'force_tol': 0.01, 'stress_tol': 0.01, 'stop_criterion': 1e-14,
                  'max_iteration_number': 2, 'num_groups': 6, 'initial_kt': 0.3, 'current_iter_index': 1,
                  'hookean_repul': False, 'hookean_paras': None, 'keep_symmetry': False, 'write_traj': True,
                  'num_processes_rss': 128, 'device_for_rss': 'cpu', 'mlip_hypers': {'GAP': {
            'general': {'at_file': 'train.extxyz', 'default_sigma': '{0.0001 0.05 0.05 0}',
                        'energy_parameter_name': 'REF_energy', 'force_parameter_name': 'REF_forces',
                        'virial_parameter_name': 'REF_virial', 'sparse_jitter': 1e-08, 'do_copy_at_file': 'F',
                        'openmp_chunk_size': 10000, 'gp_file': 'gap_file.xml', 'e0_offset': 0.0, 'two_body': False,
                        'three_body': False, 'soap': True},
            'twob': {'distance_Nb_order': 2, 'f0': 0.0, 'add_species': 'T', 'cutoff': 5.0, 'n_sparse': 15,
                     'covariance_type': 'ard_se', 'delta': 2.0, 'theta_uniform': 0.5, 'sparse_method': 'uniform',
                     'compact_clusters': 'T'},
            'threeb': {'distance_Nb_order': 3, 'f0': 0.0, 'add_species': 'T', 'cutoff': 3.25, 'n_sparse': 100,
                       'covariance_type': 'ard_se', 'delta': 2.0, 'theta_uniform': 1.0, 'sparse_method': 'uniform',
                       'compact_clusters': 'T'},
            'soap': {'add_species': 'T', 'l_max': 6, 'n_max': 8, 'atom_sigma': 0.5, 'zeta': 4, 'cutoff': 5.0,
                     'cutoff_transition_width': 1.0, 'central_weight': 1.0, 'n_sparse': 1000, 'delta': 1.0, 'f0': 0.0,
                     'covariance_type': 'dot_product', 'sparse_method': 'cur_points'}}, 'J_ACE': {'order': 3,
                                                                                                  'totaldegree': 6,
                                                                                                  'cutoff': 2.0,
                                                                                                  'solver': 'BLR'},
                                                                                     'NEQUIP': {'root': 'results',
                                                                                                'run_name': 'autoplex',
                                                                                                'seed': 123,
                                                                                                'dataset_seed': 123,
                                                                                                'append': False,
                                                                                                'default_dtype': 'float64',
                                                                                                'model_dtype': 'float64',
                                                                                                'allow_tf32': True,
                                                                                                'r_max': 4.0,
                                                                                                'num_layers': 4,
                                                                                                'l_max': 2,
                                                                                                'parity': True,
                                                                                                'num_features': 32,
                                                                                                'nonlinearity_type': 'gate',
                                                                                                'nonlinearity_scalars': {
                                                                                                    'e': 'silu',
                                                                                                    'o': 'tanh'},
                                                                                                'nonlinearity_gates': {
                                                                                                    'e': 'silu',
                                                                                                    'o': 'tanh'},
                                                                                                'num_basis': 8,
                                                                                                'besselbasis_trainable': True,
                                                                                                'polynomialcutoff_p': 5,
                                                                                                'invariant_layers': 2,
                                                                                                'invariant_neurons': 64,
                                                                                                'avg_num_neighbors': 'auto',
                                                                                                'use_sc': True,
                                                                                                'dataset': 'ase',
                                                                                                'validation_dataset': 'ase',
                                                                                                'dataset_file_name': './train_nequip.extxyz',
                                                                                                'validation_dataset_file_name': './test.extxyz',
                                                                                                'ase_args': {
                                                                                                    'format': 'extxyz'},
                                                                                                'dataset_key_mapping': {
                                                                                                    'forces': 'forces',
                                                                                                    'energy': 'total_energy'},
                                                                                                'validation_dataset_key_mapping': {
                                                                                                    'forces': 'forces',
                                                                                                    'energy': 'total_energy'},
                                                                                                'chemical_symbols': [],
                                                                                                'wandb': False,
                                                                                                'verbose': 'info',
                                                                                                'log_batch_freq': 10,
                                                                                                'log_epoch_freq': 1,
                                                                                                'save_checkpoint_freq': -1,
                                                                                                'save_ema_checkpoint_freq': -1,
                                                                                                'n_train': 1000,
                                                                                                'n_val': 1000,
                                                                                                'learning_rate': 0.005,
                                                                                                'batch_size': 5,
                                                                                                'validation_batch_size': 10,
                                                                                                'max_epochs': 10000,
                                                                                                'shuffle': True,
                                                                                                'metrics_key': 'validation_loss',
                                                                                                'use_ema': True,
                                                                                                'ema_decay': 0.99,
                                                                                                'ema_use_num_updates': True,
                                                                                                'report_init_validation': True,
                                                                                                'early_stopping_patiences': {
                                                                                                    'validation_loss': 50},
                                                                                                'early_stopping_lower_bounds': {
                                                                                                    'LR': 1e-05},
                                                                                                'loss_coeffs': {
                                                                                                    'forces': 1,
                                                                                                    'total_energy': [1,
                                                                                                                     'PerAtomMSELoss']},
                                                                                                'metrics_components': [
                                                                                                    ['forces', 'mae'],
                                                                                                    ['forces', 'rmse'],
                                                                                                    ['forces', 'mae', {
                                                                                                        'PerSpecies': True,
                                                                                                        'report_per_component': False}],
                                                                                                    ['forces', 'rmse', {
                                                                                                        'PerSpecies': True,
                                                                                                        'report_per_component': False}],
                                                                                                    ['total_energy',
                                                                                                     'mae'],
                                                                                                    ['total_energy',
                                                                                                     'mae', {
                                                                                                         'PerAtom': True}]],
                                                                                                'optimizer_name': 'Adam',
                                                                                                'optimizer_amsgrad': True,
                                                                                                'lr_scheduler_name': 'ReduceLROnPlateau',
                                                                                                'lr_scheduler_patience': 100,
                                                                                                'lr_scheduler_factor': 0.5,
                                                                                                'per_species_rescale_shifts_trainable': False,
                                                                                                'per_species_rescale_scales_trainable': False,
                                                                                                'per_species_rescale_shifts': 'dataset_per_atom_total_energy_mean',
                                                                                                'per_species_rescale_scales': 'dataset_per_species_forces_rms'},
                                                                                     'M3GNET': {'exp_name': 'training',
                                                                                                'results_dir': 'm3gnet_results',
                                                                                                'foundation_model': None,
                                                                                                'use_foundation_model_element_refs': False,
                                                                                                'allow_missing_labels': False,
                                                                                                'cutoff': 5.0,
                                                                                                'threebody_cutoff': 4.0,
                                                                                                'batch_size': 10,
                                                                                                'max_epochs': 1000,
                                                                                                'include_stresses': True,
                                                                                                'data_mean': 0.0,
                                                                                                'data_std': 1.0,
                                                                                                'decay_steps': 1000,
                                                                                                'decay_alpha': 0.96,
                                                                                                'dim_node_embedding': 128,
                                                                                                'dim_edge_embedding': 128,
                                                                                                'dim_state_embedding': 0,
                                                                                                'energy_weight': 1.0,
                                                                                                'element_refs': None,
                                                                                                'force_weight': 1.0,
                                                                                                'include_line_graph': True,
                                                                                                'loss': 'mse_loss',
                                                                                                'loss_params': None,
                                                                                                'lr': 0.001,
                                                                                                'magmom_target': 'absolute',
                                                                                                'magmom_weight': 0.0,
                                                                                                'max_l': 4, 'max_n': 4,
                                                                                                'nblocks': 3,
                                                                                                'optimizer': None,
                                                                                                'rbf_type': 'Gaussian',
                                                                                                'scheduler': None,
                                                                                                'stress_weight': 0.0,
                                                                                                'sync_dist': False,
                                                                                                'is_intensive': False,
                                                                                                'units': 128},
                                                                                     'MACE': {'model': 'MACE',
                                                                                              'name': 'MACE_model',
                                                                                              'amsgrad': True,
                                                                                              'batch_size': 10,
                                                                                              'compute_avg_num_neighbors': True,
                                                                                              'compute_forces': True,
                                                                                              'config_type_weights': "{'Default':1.0}",
                                                                                              'compute_stress': False,
                                                                                              'compute_statistics': False,
                                                                                              'correlation': 3,
                                                                                              'default_dtype': 'float32',
                                                                                              'device': 'cpu',
                                                                                              'distributed': False,
                                                                                              'energy_weight': 1.0,
                                                                                              'ema': True,
                                                                                              'ema_decay': 0.99,
                                                                                              'E0s': None,
                                                                                              'forces_weight': 100.0,
                                                                                              'foundation_filter_elements': True,
                                                                                              'foundation_model': None,
                                                                                              'foundation_model_readout': True,
                                                                                              'keep_checkpoint': False,
                                                                                              'keep_isolated_atoms': False,
                                                                                              'hidden_irreps': '128x0e + 128x1o',
                                                                                              'loss': 'huber',
                                                                                              'lr': 0.001,
                                                                                              'multiheads_finetuning': False,
                                                                                              'max_num_epochs': 1500,
                                                                                              'pair_repulsion': False,
                                                                                              'patience': 2048,
                                                                                              'r_max': 5.0,
                                                                                              'restart_latest': False,
                                                                                              'seed': 123,
                                                                                              'save_cpu': True,
                                                                                              'save_all_checkpoints': False,
                                                                                              'scaling': 'rms_forces_scaling',
                                                                                              'stress_weight': 1.0,
                                                                                              'start_swa': 1200,
                                                                                              'swa': True,
                                                                                              'valid_batch_size': 10,
                                                                                              'virials_weight': 1.0,
                                                                                              'wandb': False},
                                                                                     'NEP': {'version': 4,
                                                                                             'type': [1, 'X'],
                                                                                             'type_weight': 1.0,
                                                                                             'model_type': 0,
                                                                                             'prediction': 0,
                                                                                             'cutoff': [6, 5],
                                                                                             'n_max': [4, 4],
                                                                                             'basis_size': [8, 8],
                                                                                             'l_max': [4, 2, 1],
                                                                                             'neuron': 80,
                                                                                             'lambda_1': 0.0,
                                                                                             'lambda_e': 1.0,
                                                                                             'lambda_f': 1.0,
                                                                                             'lambda_v': 0.1,
                                                                                             'force_delta': 0,
                                                                                             'batch': 1000,
                                                                                             'population': 60,
                                                                                             'generation': 100000,
                                                                                             'zbl': 2}},
                  '@module': 'autoplex.settings', '@class': 'RssConfig', '@version': '0.1.4.dev16+g1029f622'}
    # test the default VASPmakers instead
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker
    rss_config["static_energy_maker"]=ForceFieldRelaxMaker(force_field_name="MACE-MP-0b3")
    rss_config["static_energy_maker_isolated_species"]=ForceFieldRelaxMaker(force_field_name="MACE-MP-0b3")

    rss_config = RssConfig.from_dict(rss_config)





    rss_job = RssMaker(name="rss", rss_config=rss_config).make()
    from atomate2.vasp.powerups import update_user_incar_settings


    responses=run_locally(
        Flow(jobs=[rss_job], output=rss_job.output),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert rss_job.name == "rss"


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

