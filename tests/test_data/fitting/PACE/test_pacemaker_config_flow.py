"""
Unit tests to verify that Pacemaker (P-ACE) hyperparameters are correctly
passed from the autoplex configuration YAML file through the RSS workflow
to the final Pacemaker input.yaml file.
"""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import json

from monty.serialization import loadfn

from autoplex.settings import RssConfig, PacemakerSettings, MLIPHypers
from autoplex.fitting.common.utils import pace_fitting


# Sample autoplex configuration with detailed P-ACE parameters
SAMPLE_AUTOPLEX_CONFIG = """
tag: 'Si'
train_from_scratch: true
resume_from_previous_state:
  test_error:
  pre_database_dir:
  mlip_path:
  isolated_atom_energies:
generated_struct_numbers:
  - 1000
buildcell_options:
  - NFORM: '1'
    SYMMOPS: '1-4'
    SLACK: 0.25
    OVERLAP: 0.1
    NATOM: '{6,8,10,12,14,16,18,20,22,24}'
num_processes_buildcell: 128
num_of_initial_selected_structs:
  - 10
num_of_rss_selected_structs: 20
initial_selection_enabled: true
rss_selection_method: bcur2i
bcur_params:
  soap_paras:
    l_max: 12
    n_max: 12
    atom_sigma: 0.0875
    cutoff: 10.5
    cutoff_transition_width: 1.0
    zeta: 4.0
    average: true
    species: true
  frac_of_bcur: 0.8
  bolt_max_num: 3000
include_isolated_atom: true
isolatedatom_box:
  - 20.0
  - 20.0
  - 20.0
e0_spin: false
include_dimer: false
dimer_box:
  - 20.0
  - 20.0
  - 20.0
dimer_range:
  - 1.0
  - 5.0
dimer_num: 21
custom_incar:
  ISMEAR: 0
  SIGMA: 0.05
  PREC: Normal
  EDIFF: 1e-05
  NELM: 250
  LWAVE: .FALSE.
  LCHARG: .FALSE.
  ALGO: Normal
  LREAL: .FALSE.
  ISYM: 0
  ENCUT: 300.0
  KSPACING: 0.2
  KPAR: 8
  NCORE: 16
  LSCALAPACK: .FALSE.
  LPLANE: .FALSE.
vasp_ref_file: vasp_ref.extxyz
config_types:
  - initial
  - traj_early
  - traj
rss_group:
  - traj
test_ratio: 0
regularization: true
retain_existing_sigma: false
scheme: linear-hull
reg_minmax:
  - - 0.1
    - 1
  - - 0.001
    - 0.1
  - - 0.0316
    - 0.316
  - - 0.0632
    - 0.632
distillation: false
mlip_type: P-ACE
ref_energy_name: REF_energy
ref_force_name: REF_forces
ref_virial_name: REF_virial
auto_delta: true
num_processes_fit: 32
device_for_fitting: cpu
scalar_pressure_method: uniform
scalar_exp_pressure: 1
scalar_pressure_exponential_width: 0.2
scalar_pressure_low: 0
scalar_pressure_high: 25
max_steps: 200
force_tol: 0.01
stress_tol: 0.01
stop_criterion: 0.00000000000001
max_iteration_number: 4
num_groups: 6
initial_kt: 0.3
current_iter_index: 1
hookean_repul: false
keep_symmetry: false
remove_traj_files: true
num_processes_rss: 128
device_for_rss: cpu
mlip_hypers:
  P-ACE:
    cutoff: 8.0
    seed: 42
    backend:
      batch_size: 8
      batch_size_reduction: true
      batch_size_reduction_factor: 2
      display_step: 10
      evaluator: tensorpot
    fit:
      optimizer: BFGS
      maxiter: 800
      loss:
        kappa: 0.75
        L1_coeffs: 1.0e-08
        L2_coeffs: 1.0e-08
        w_energy: 1.0
        w_forces: 1.0
      repulsion: auto
      trainable_parameters: ALL
    potential:
      deltaSplineBins: 0.001
      elements:
      - Si
      embeddings:
        ALL:
          npot: FinnisSinclairShiftedScaled
          fs_parameters: [1, 1, 7.5577494194990145, 0.101]
          ndensity: 2
          rho_core_cut: 3000
          drho_core_cut: 500
      bonds:
        ALL:
          radbase: SBessel
          radparameters: [3.3135018034502046]
          rcut: 6.311052419794955
          dcut: 0.018905380302626437
          NameofCutoffFunction: cos
      functions:
        number_of_functions_per_element: 500
        ALL:
          nradmax_by_orders: [20, 12, 8, 6, 3, 2, 1]
          lmax_by_orders: [0, 7, 6, 4, 2, 1, 0]
"""

# Expected values from the YAML config for validation
EXPECTED_PACE_VALUES = {
    # Top-level
    "cutoff": 8.0,
    "seed": 42,
    # Backend
    "backend": {
        "batch_size": 8,
        "batch_size_reduction": True,
        "batch_size_reduction_factor": 2,
        "display_step": 10,
        "evaluator": "tensorpot",
    },
    # Fit
    "fit": {
        "optimizer": "BFGS",
        "maxiter": 800,
        "loss": {
            "kappa": 0.75,
            "L1_coeffs": 1.0e-08,
            "L2_coeffs": 1.0e-08,
            "w_energy": 1.0,
            "w_forces": 1.0,
        },
        "repulsion": "auto",
        "trainable_parameters": "ALL",
    },
    # Potential
    "potential": {
        "deltaSplineBins": 0.001,
        "elements": ["Si"],
        "embeddings": {
            "ALL": {
                "npot": "FinnisSinclairShiftedScaled",
                "fs_parameters": [1, 1, 7.5577494194990145, 0.101],
                "ndensity": 2,
                "rho_core_cut": 3000,
                "drho_core_cut": 500,
            }
        },
        "bonds": {
            "ALL": {
                "radbase": "SBessel",
                "radparameters": [3.3135018034502046],
                "rcut": 6.311052419794955,
                "dcut": 0.018905380302626437,
                "NameofCutoffFunction": "cos",
            }
        },
        "functions": {
            "number_of_functions_per_element": 500,
            "ALL": {
                "nradmax_by_orders": [20, 12, 8, 6, 3, 2, 1],
                "lmax_by_orders": [0, 7, 6, 4, 2, 1, 0],
            }
        },
    },
}


def print_dict_comparison(title, expected, actual, indent=0):
    """Helper function to print detailed comparison of dictionaries."""
    prefix = "  " * indent
    print(f"{prefix}{title}:")
    if isinstance(expected, dict) and isinstance(actual, dict):
        all_keys = set(expected.keys()) | set(actual.keys())
        for key in sorted(all_keys):
            exp_val = expected.get(key, "<MISSING>")
            act_val = actual.get(key, "<MISSING>")
            if isinstance(exp_val, dict) and isinstance(act_val, dict):
                print_dict_comparison(f"[{key}]", exp_val, act_val, indent + 1)
            else:
                match = "✓" if exp_val == act_val else "✗"
                print(f"{prefix}  {key}: expected={exp_val}, actual={act_val} {match}")
    else:
        match = "✓" if expected == actual else "✗"
        print(f"{prefix}  expected={expected}, actual={actual} {match}")


@pytest.fixture
def clean_dir(tmp_path, monkeypatch):
    """Change to a clean temporary directory for each test."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


class TestRssMakerConfigExtraction:
    """
    Test the exact parameter extraction logic used in RssMaker.make().
    This simulates what happens in flows.py lines 167-175.
    """
    
    @pytest.fixture
    def autoplex_config_file(self, tmp_path):
        """Create a temporary autoplex config file."""
        config_file = tmp_path / "autoplex_config.yaml"
        config_file.write_text(SAMPLE_AUTOPLEX_CONFIG)
        return config_file

    def test_rss_maker_style_config_extraction(self, autoplex_config_file):
        """
        Test the EXACT logic used in RssMaker.make() for extracting MLIP hypers.
        
        This replicates flows.py lines 167-175:
            config_params = default_config.model_dump(by_alias=True, exclude_none=True)
            mlip_hypers = config_params["mlip_hypers"][config_params["mlip_type"]]
            del config_params["mlip_hypers"]
            config_params.update(mlip_hypers)
        """
        print("\n" + "="*80)
        print("TEST: RssMaker-style config extraction")
        print("="*80)
        
        # Step 1: Load config (simulates RssMaker.__init__ and make())
        config = RssConfig.from_file(str(autoplex_config_file))
        
        print("\n[Step 1] Loaded RssConfig from YAML file")
        print(f"  mlip_type: {config.mlip_type}")
        print(f"  P_ACE.cutoff: {config.mlip_hypers.P_ACE.cutoff}")
        print(f"  P_ACE.seed: {config.mlip_hypers.P_ACE.seed}")
        print(f"  P_ACE.fit['maxiter']: {config.mlip_hypers.P_ACE.fit.get('maxiter', 'NOT SET')}")
        print(f"  P_ACE.potential['embeddings']['ALL']['fs_parameters']: {config.mlip_hypers.P_ACE.potential.get('embeddings', {}).get('ALL', {}).get('fs_parameters', 'NOT SET')}")
        
        # Step 2: model_dump (simulates flows.py line 167)
        config_params = config.model_dump(by_alias=True, exclude_none=True)
        
        print("\n[Step 2] After model_dump(by_alias=True, exclude_none=True)")
        print(f"  Keys in config_params: {list(config_params.keys())[:10]}...")  # First 10 keys
        print(f"  mlip_type in config_params: {config_params.get('mlip_type')}")
        print(f"  'mlip_hypers' in config_params: {'mlip_hypers' in config_params}")
        
        if "mlip_hypers" in config_params:
            print(f"  Keys in mlip_hypers: {list(config_params['mlip_hypers'].keys())}")
            pace_key = config_params["mlip_type"]  # Should be "P-ACE"
            print(f"  Looking for key: {pace_key}")
            if pace_key in config_params["mlip_hypers"]:
                pace_dict = config_params["mlip_hypers"][pace_key]
                print(f"  P-ACE cutoff: {pace_dict.get('cutoff', 'NOT SET')}")
                print(f"  P-ACE seed: {pace_dict.get('seed', 'NOT SET')}")
                print(f"  P-ACE fit.maxiter: {pace_dict.get('fit', {}).get('maxiter', 'NOT SET')}")
        
        # Step 3: Extract MLIP hyperparameters (simulates flows.py lines 170-171)
        mlip_hypers = config_params["mlip_hypers"][config_params["mlip_type"]]
        
        print("\n[Step 3] Extracted mlip_hypers dict")
        print(f"  Keys: {list(mlip_hypers.keys())}")
        print(f"  cutoff: {mlip_hypers.get('cutoff')}")
        print(f"  seed: {mlip_hypers.get('seed')}")
        print(f"  fit.maxiter: {mlip_hypers.get('fit', {}).get('maxiter')}")
        print(f"  fit.loss.kappa: {mlip_hypers.get('fit', {}).get('loss', {}).get('kappa')}")
        print(f"  potential.embeddings.ALL.fs_parameters: {mlip_hypers.get('potential', {}).get('embeddings', {}).get('ALL', {}).get('fs_parameters')}")
        print(f"  potential.bonds.ALL.radbase: {mlip_hypers.get('potential', {}).get('bonds', {}).get('ALL', {}).get('radbase')}")
        print(f"  backend.evaluator: {mlip_hypers.get('backend', {}).get('evaluator')}")
        
        # Step 4: Delete mlip_hypers and update config_params (flows.py lines 172-173)
        del config_params["mlip_hypers"]
        config_params.update(mlip_hypers)
        
        print("\n[Step 4] After flattening (del mlip_hypers + update)")
        print(f"  'mlip_hypers' in config_params: {'mlip_hypers' in config_params}")
        print(f"  'cutoff' in config_params: {'cutoff' in config_params}")
        print(f"  'fit' in config_params: {'fit' in config_params}")
        print(f"  config_params['cutoff']: {config_params.get('cutoff')}")
        print(f"  config_params['fit']['maxiter']: {config_params.get('fit', {}).get('maxiter')}")
        
        # Assertions for the extracted values
        assert mlip_hypers["cutoff"] == EXPECTED_PACE_VALUES["cutoff"], \
            f"cutoff mismatch: expected {EXPECTED_PACE_VALUES['cutoff']}, got {mlip_hypers['cutoff']}"
        assert mlip_hypers["seed"] == EXPECTED_PACE_VALUES["seed"], \
            f"seed mismatch: expected {EXPECTED_PACE_VALUES['seed']}, got {mlip_hypers['seed']}"
        
        # Check fit parameters
        assert mlip_hypers["fit"]["maxiter"] == EXPECTED_PACE_VALUES["fit"]["maxiter"], \
            f"fit.maxiter mismatch: expected {EXPECTED_PACE_VALUES['fit']['maxiter']}, got {mlip_hypers['fit']['maxiter']}"
        assert mlip_hypers["fit"]["optimizer"] == EXPECTED_PACE_VALUES["fit"]["optimizer"]
        assert mlip_hypers["fit"]["loss"]["kappa"] == EXPECTED_PACE_VALUES["fit"]["loss"]["kappa"]
        
        # Check potential parameters (the critical nested ones)
        pot = mlip_hypers["potential"]
        exp_pot = EXPECTED_PACE_VALUES["potential"]
        
        assert pot["embeddings"]["ALL"]["fs_parameters"] == exp_pot["embeddings"]["ALL"]["fs_parameters"], \
            f"fs_parameters mismatch: expected {exp_pot['embeddings']['ALL']['fs_parameters']}, got {pot['embeddings']['ALL']['fs_parameters']}"
        assert pot["embeddings"]["ALL"]["rho_core_cut"] == exp_pot["embeddings"]["ALL"]["rho_core_cut"]
        assert pot["bonds"]["ALL"]["radbase"] == exp_pot["bonds"]["ALL"]["radbase"]
        assert pot["bonds"]["ALL"]["rcut"] == exp_pot["bonds"]["ALL"]["rcut"]
        assert pot["functions"]["ALL"]["nradmax_by_orders"] == exp_pot["functions"]["ALL"]["nradmax_by_orders"]
        
        # Check backend parameters
        assert mlip_hypers["backend"]["evaluator"] == EXPECTED_PACE_VALUES["backend"]["evaluator"]
        assert mlip_hypers["backend"]["batch_size"] == EXPECTED_PACE_VALUES["backend"]["batch_size"]
        
        print("\n[PASS] All assertions passed!")
        print("="*80)


class TestMachineLearningFitIntegration:
    """
    Test how parameters flow from RssConfig through machine_learning_fit to pace_fitting.
    This simulates the actual call chain in the RSS workflow.
    """
    
    @pytest.fixture
    def autoplex_config_file(self, tmp_path):
        """Create a temporary autoplex config file."""
        config_file = tmp_path / "autoplex_config.yaml"
        config_file.write_text(SAMPLE_AUTOPLEX_CONFIG)
        return config_file
    
    @pytest.fixture
    def mock_training_data(self, tmp_path):
        """Create mock training data files."""
        from ase import Atoms
        from ase.io import write
        
        atoms_list = []
        for i in range(5):
            atoms = Atoms(
                'Si2',
                positions=[[0, 0, 0], [1.35 + i * 0.1, 1.35, 1.35]],
                cell=[5.43, 5.43, 5.43],
                pbc=True
            )
            atoms.info['REF_energy'] = -10.0 - i * 0.1
            atoms.info['REF_virial'] = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
            atoms.info['config_type'] = 'bulk'
            atoms.arrays['REF_forces'] = np.array([[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]])
            atoms_list.append(atoms)
        
        train_file = tmp_path / "train.extxyz"
        test_file = tmp_path / "test.extxyz"
        
        write(str(train_file), atoms_list[:4], format='extxyz')
        write(str(test_file), atoms_list[4:], format='extxyz')
        
        return tmp_path

    def test_jobs_py_style_pace_fitting_call(self, autoplex_config_file, mock_training_data, tmp_path, clean_dir):
        """
        Simulate how jobs.py calls pace_fitting.
        
        In jobs.py machine_learning_fit (line 132-141):
            train_test_error = pace_fitting(
                db_dir=database_dir,
                species_list=species_list,
                hyperparameters=hyperparameters.P_ACE,  # <-- PacemakerSettings object
                fit_kwargs=fit_kwargs,
                ...
            )
        
        The 'hyperparameters' comes from MLIP_HYPERS which is defined in __init__.py,
        and 'fit_kwargs' comes from the flattened config_params from RssMaker.make().
        """
        print("\n" + "="*80)
        print("TEST: jobs.py style pace_fitting call")
        print("="*80)
        
        # Load config as RssMaker would
        config = RssConfig.from_file(str(autoplex_config_file))
        
        # Extract hyperparameters as RssMaker.make() does
        config_params = config.model_dump(by_alias=True, exclude_none=True)
        mlip_hypers = config_params["mlip_hypers"][config_params["mlip_type"]]
        del config_params["mlip_hypers"]
        config_params.update(mlip_hypers)
        
        # In jobs.py, hyperparameters.P_ACE is passed directly
        # The mlip_hypers from RssMaker becomes fit_kwargs
        pace_hypers = config.mlip_hypers.P_ACE
        
        print("\n[Setup] Simulating jobs.py call to pace_fitting")
        print(f"  hyperparameters.P_ACE type: {type(pace_hypers).__name__}")
        print(f"  hyperparameters.P_ACE.cutoff: {pace_hypers.cutoff}")
        print(f"  hyperparameters.P_ACE.seed: {pace_hypers.seed}")
        print(f"  hyperparameters.P_ACE.fit: {pace_hypers.fit}")
        print(f"  hyperparameters.P_ACE.potential['embeddings']['ALL']['fs_parameters']: "
              f"{pace_hypers.potential.get('embeddings', {}).get('ALL', {}).get('fs_parameters')}")
        
        # fit_kwargs in jobs.py would be the extra kwargs passed to machine_learning_fit
        # In RSS workflow, these come from the flattened config_params
        # For this test, we'll pass the MLIP-specific params as fit_kwargs
        fit_kwargs = {
            "cutoff": mlip_hypers.get("cutoff"),
            "seed": mlip_hypers.get("seed"),
            "fit": mlip_hypers.get("fit"),
            "potential": mlip_hypers.get("potential"),
            "backend": mlip_hypers.get("backend"),
        }
        
        print(f"\n  fit_kwargs keys: {list(fit_kwargs.keys())}")
        print(f"  fit_kwargs['cutoff']: {fit_kwargs.get('cutoff')}")
        print(f"  fit_kwargs['fit']['maxiter']: {fit_kwargs.get('fit', {}).get('maxiter')}")
        
        os.chdir(tmp_path)
        
        with patch('autoplex.fitting.common.utils.run_pacemaker'):
            with patch('shutil.which', return_value='/usr/bin/pacemaker'):
                try:
                    pace_fitting(
                        db_dir=mock_training_data,
                        species_list=["Si"],
                        hyperparameters=pace_hypers,
                        fit_kwargs=fit_kwargs,
                        isolated_atom_energies={14: -5.0},
                        ref_energy_name="REF_energy",
                        ref_force_name="REF_forces",
                        num_processes_fit=1,
                    )
                except Exception as e:
                    print(f"  [Note] pace_fitting raised: {type(e).__name__}: {e}")
        
        # Check the generated input.yaml
        input_yaml_path = tmp_path / "input.yaml"
        assert input_yaml_path.exists(), "input.yaml was not created!"
        
        generated = loadfn(str(input_yaml_path))
        
        print("\n[Result] Generated input.yaml content:")
        print(f"  Keys: {list(generated.keys())}")
        print(f"  cutoff: {generated.get('cutoff')}")
        print(f"  seed: {generated.get('seed')}")
        print(f"  fit.maxiter: {generated.get('fit', {}).get('maxiter')}")
        print(f"  fit.loss.kappa: {generated.get('fit', {}).get('loss', {}).get('kappa')}")
        print(f"  potential.embeddings.ALL.fs_parameters: {generated.get('potential', {}).get('embeddings', {}).get('ALL', {}).get('fs_parameters')}")
        print(f"  potential.bonds.ALL.radbase: {generated.get('potential', {}).get('bonds', {}).get('ALL', {}).get('radbase')}")
        print(f"  backend.evaluator: {generated.get('backend', {}).get('evaluator')}")
        
        # Verify all expected values
        print("\n[Verification] Comparing with expected values:")
        
        # Top-level
        assert generated["cutoff"] == EXPECTED_PACE_VALUES["cutoff"], \
            f"cutoff: expected {EXPECTED_PACE_VALUES['cutoff']}, got {generated['cutoff']}"
        print(f"  ✓ cutoff: {generated['cutoff']}")
        
        assert generated["seed"] == EXPECTED_PACE_VALUES["seed"]
        print(f"  ✓ seed: {generated['seed']}")
        
        # Fit
        assert generated["fit"]["maxiter"] == EXPECTED_PACE_VALUES["fit"]["maxiter"], \
            f"fit.maxiter: expected {EXPECTED_PACE_VALUES['fit']['maxiter']}, got {generated['fit']['maxiter']}"
        print(f"  ✓ fit.maxiter: {generated['fit']['maxiter']}")
        
        assert generated["fit"]["loss"]["kappa"] == EXPECTED_PACE_VALUES["fit"]["loss"]["kappa"]
        print(f"  ✓ fit.loss.kappa: {generated['fit']['loss']['kappa']}")
        
        # Potential - critical nested values
        pot = generated["potential"]
        exp_pot = EXPECTED_PACE_VALUES["potential"]
        
        assert pot["embeddings"]["ALL"]["fs_parameters"] == exp_pot["embeddings"]["ALL"]["fs_parameters"], \
            f"fs_parameters: expected {exp_pot['embeddings']['ALL']['fs_parameters']}, got {pot['embeddings']['ALL']['fs_parameters']}"
        print(f"  ✓ potential.embeddings.ALL.fs_parameters: {pot['embeddings']['ALL']['fs_parameters']}")
        
        assert pot["embeddings"]["ALL"]["rho_core_cut"] == exp_pot["embeddings"]["ALL"]["rho_core_cut"]
        print(f"  ✓ potential.embeddings.ALL.rho_core_cut: {pot['embeddings']['ALL']['rho_core_cut']}")
        
        assert pot["bonds"]["ALL"]["radbase"] == exp_pot["bonds"]["ALL"]["radbase"]
        print(f"  ✓ potential.bonds.ALL.radbase: {pot['bonds']['ALL']['radbase']}")
        
        assert pot["bonds"]["ALL"]["radparameters"] == exp_pot["bonds"]["ALL"]["radparameters"]
        print(f"  ✓ potential.bonds.ALL.radparameters: {pot['bonds']['ALL']['radparameters']}")
        
        assert pot["bonds"]["ALL"]["rcut"] == exp_pot["bonds"]["ALL"]["rcut"]
        print(f"  ✓ potential.bonds.ALL.rcut: {pot['bonds']['ALL']['rcut']}")
        
        assert pot["functions"]["ALL"]["nradmax_by_orders"] == exp_pot["functions"]["ALL"]["nradmax_by_orders"]
        print(f"  ✓ potential.functions.ALL.nradmax_by_orders: {pot['functions']['ALL']['nradmax_by_orders']}")
        
        # Backend
        assert generated["backend"]["evaluator"] == EXPECTED_PACE_VALUES["backend"]["evaluator"]
        print(f"  ✓ backend.evaluator: {generated['backend']['evaluator']}")
        
        assert generated["backend"]["batch_size"] == EXPECTED_PACE_VALUES["backend"]["batch_size"]
        print(f"  ✓ backend.batch_size: {generated['backend']['batch_size']}")
        
        # Verify NO RSS parameters leaked through
        rss_params = ['initial_kt', 'vasp_ref_file', 'num_processes_rss', 'mlip_type', 'tag']
        for param in rss_params:
            assert param not in generated, f"RSS parameter '{param}' should NOT be in input.yaml!"
        print(f"  ✓ No RSS parameters in input.yaml")
        
        print("\n[PASS] All verifications passed!")
        print("="*80)


class TestPacemakerConfigFlow:
    """Test class for Pacemaker configuration flow."""

    @pytest.fixture
    def autoplex_config_file(self, tmp_path):
        """Create a temporary autoplex config file."""
        config_file = tmp_path / "autoplex_config.yaml"
        config_file.write_text(SAMPLE_AUTOPLEX_CONFIG)
        return config_file

    @pytest.fixture
    def mock_training_data(self, tmp_path):
        """Create mock training data files."""
        from ase import Atoms
        from ase.io import write
        
        atoms_list = []
        for i in range(5):
            atoms = Atoms(
                'Si2',
                positions=[[0, 0, 0], [1.35 + i * 0.1, 1.35, 1.35]],
                cell=[5.43, 5.43, 5.43],
                pbc=True
            )
            atoms.info['REF_energy'] = -10.0 - i * 0.1
            atoms.info['REF_virial'] = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
            atoms.info['config_type'] = 'bulk'
            atoms.arrays['REF_forces'] = np.array([[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]])
            atoms_list.append(atoms)
        
        train_file = tmp_path / "train.extxyz"
        test_file = tmp_path / "test.extxyz"
        
        write(str(train_file), atoms_list[:4], format='extxyz')
        write(str(test_file), atoms_list[4:], format='extxyz')
        
        return tmp_path

    def test_rss_config_loads_mlip_hypers_correctly(self, autoplex_config_file):
        """Test that RssConfig correctly loads all MLIP hyperparameters."""
        config = RssConfig.from_file(str(autoplex_config_file))
        pace_hypers = config.mlip_hypers.P_ACE
        
        print("\n[DEBUG] Loaded P_ACE hyperparameters:")
        print(f"  cutoff: {pace_hypers.cutoff}")
        print(f"  seed: {pace_hypers.seed}")
        print(f"  fit: {pace_hypers.fit}")
        print(f"  potential keys: {list(pace_hypers.potential.keys())}")
        print(f"  potential.embeddings.ALL: {pace_hypers.potential.get('embeddings', {}).get('ALL', {})}")
        
        assert config.mlip_type == "P-ACE"
        assert pace_hypers.cutoff == EXPECTED_PACE_VALUES["cutoff"]
        assert pace_hypers.seed == EXPECTED_PACE_VALUES["seed"]
        assert pace_hypers.fit["maxiter"] == EXPECTED_PACE_VALUES["fit"]["maxiter"]
        assert pace_hypers.potential["embeddings"]["ALL"]["fs_parameters"] == \
            EXPECTED_PACE_VALUES["potential"]["embeddings"]["ALL"]["fs_parameters"]

    def test_pacemaker_settings_not_polluted_by_rss_params(self, autoplex_config_file):
        """Test that PacemakerSettings object is NOT polluted with RSS control parameters."""
        config = RssConfig.from_file(str(autoplex_config_file))
        pace_dict = config.mlip_hypers.P_ACE.model_dump()
        
        rss_params_that_should_not_exist = [
            'initial_kt', 'vasp_ref_file', 'num_processes_rss',
            'max_iteration_number', 'stop_criterion', 'config_types',
            'rss_group', 'scalar_pressure_method', 'hookean_repul',
            'train_from_scratch', 'mlip_type', 'tag', 'num_processes_buildcell',
            'generated_struct_numbers', 'buildcell_options', 'bcur_params',
        ]
        
        for param in rss_params_that_should_not_exist:
            assert param not in pace_dict, \
                f"RSS parameter '{param}' should NOT be in PacemakerSettings!"

    def test_input_yaml_only_allowed_keys(
        self, autoplex_config_file, mock_training_data, tmp_path, clean_dir
    ):
        """Test that input.yaml only contains allowed Pacemaker top-level keys."""
        config = RssConfig.from_file(str(autoplex_config_file))
        pace_hypers = config.mlip_hypers.P_ACE
        
        os.chdir(tmp_path)
        
        with patch('autoplex.fitting.common.utils.run_pacemaker'):
            with patch('shutil.which', return_value='/usr/bin/pacemaker'):
                try:
                    pace_fitting(
                        db_dir=mock_training_data,
                        species_list=["Si"],
                        hyperparameters=pace_hypers,
                        isolated_atom_energies={14: -5.0},
                    )
                except Exception:
                    pass
        
        input_yaml_path = tmp_path / "input.yaml"
        if input_yaml_path.exists():
            generated = loadfn(str(input_yaml_path))
            
            allowed_keys = {"cutoff", "seed", "metadata", "potential", "data", "fit", "backend"}
            actual_keys = set(generated.keys())
            unexpected_keys = actual_keys - allowed_keys
            
            assert len(unexpected_keys) == 0, \
                f"Unexpected keys in input.yaml: {unexpected_keys}"


class TestEndToEndConfigPropagation:
    """End-to-end tests for configuration propagation."""
    
    def test_pacemaker_settings_direct_replacement(self):
        """Verify user-provided nested dicts completely replace defaults."""
        pace_settings = PacemakerSettings(
            cutoff=8.0,
            fit={"maxiter": 800}
        )
        
        print("\n[DEBUG] PacemakerSettings with partial fit:")
        print(f"  cutoff: {pace_settings.cutoff}")
        print(f"  fit: {pace_settings.fit}")
        
        assert pace_settings.fit == {"maxiter": 800}
        assert "optimizer" not in pace_settings.fit
        assert "loss" not in pace_settings.fit

    def test_mlip_hypers_with_alias(self):
        """Test that MLIPHypers handles P-ACE alias correctly."""
        # Test with hyphen (alias)
        hypers_dict = {"P-ACE": {"cutoff": 8.0, "seed": 42}}
        hypers = MLIPHypers(**hypers_dict)
        
        print("\n[DEBUG] MLIPHypers with P-ACE alias:")
        print(f"  hypers.P_ACE.cutoff: {hypers.P_ACE.cutoff}")
        print(f"  hypers.P_ACE.seed: {hypers.P_ACE.seed}")
        
        assert hypers.P_ACE.cutoff == 8.0
        
        # Verify model_dump with alias
        dumped = hypers.model_dump(by_alias=True)
        print(f"  dumped keys: {list(dumped.keys())}")
        print(f"  'P-ACE' in dumped: {'P-ACE' in dumped}")
        
        assert "P-ACE" in dumped
        assert dumped["P-ACE"]["cutoff"] == 8.0


class TestOtherMLIPsNotAffected:
    """Test that the MLIPHypers config changes don't affect other MLIPs."""
    
    def test_mace_settings_work_correctly(self):
        """Verify MACE settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import MACESettings, MLIPHypers, RssConfig
        
        mace = MACESettings(
            batch_size=20,
            max_num_epochs=500,
            lr=0.002,
            r_max=6.0,
        )
        assert mace.batch_size == 20
        assert mace.max_num_epochs == 500
        assert mace.lr == 0.002
        assert mace.r_max == 6.0
        
        hypers = MLIPHypers(MACE=mace)
        assert hypers.MACE.batch_size == 20
        assert hypers.MACE.max_num_epochs == 500
        
        config = RssConfig(
            tag="Si",
            mlip_type="MACE",
            mlip_hypers=MLIPHypers(MACE=MACESettings(batch_size=30))
        )
        assert config.mlip_hypers.MACE.batch_size == 30
        
        # Test model_dump preserves values
        dumped = config.model_dump(by_alias=True, exclude_none=True)
        assert dumped["mlip_hypers"]["MACE"]["batch_size"] == 30

    def test_gap_settings_work_correctly(self):
        """Verify GAP settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import GAPSettings, SoapSettings, MLIPHypers, RssConfig
        
        # Test direct GAPSettings creation with nested settings
        gap = GAPSettings(
            soap=SoapSettings(
                l_max=8,
                n_max=10,
                cutoff=6.0,
            )
        )
        assert gap.soap.l_max == 8
        assert gap.soap.n_max == 10
        assert gap.soap.cutoff == 6.0
        
        # Test via MLIPHypers
        hypers = MLIPHypers(GAP=gap)
        assert hypers.GAP.soap.l_max == 8
        assert hypers.GAP.soap.cutoff == 6.0
        
        # Test via RssConfig
        config = RssConfig(
            tag="Si",
            mlip_type="GAP",
            mlip_hypers=MLIPHypers(
                GAP=GAPSettings(
                    soap=SoapSettings(l_max=12, n_max=14)
                )
            )
        )
        assert config.mlip_hypers.GAP.soap.l_max == 12
        assert config.mlip_hypers.GAP.soap.n_max == 14
        
        # Test model_dump preserves values
        dumped = config.model_dump(by_alias=True, exclude_none=True)
        assert dumped["mlip_hypers"]["GAP"]["soap"]["l_max"] == 12

    def test_nep_settings_work_correctly(self):
        """Verify NEP settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import NEPSettings, MLIPHypers, RssConfig
        
        # Test direct NEPSettings creation
        nep = NEPSettings(
            generation=50000,
            batch=500,
            neuron=100,
            cutoff=[7, 6],
        )
        assert nep.generation == 50000
        assert nep.batch == 500
        assert nep.neuron == 100
        assert nep.cutoff == [7, 6]
        
        # Test via MLIPHypers
        hypers = MLIPHypers(NEP=nep)
        assert hypers.NEP.generation == 50000
        assert hypers.NEP.cutoff == [7, 6]
        
        # Test via RssConfig
        config = RssConfig(
            tag="Si",
            mlip_type="GAP",  # mlip_type doesn't matter for this test
            mlip_hypers=MLIPHypers(NEP=NEPSettings(generation=80000))
        )
        assert config.mlip_hypers.NEP.generation == 80000

    def test_m3gnet_settings_work_correctly(self):
        """Verify M3GNET settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import M3GNETSettings, MLIPHypers
        
        m3gnet = M3GNETSettings(
            cutoff=6.0,
            max_epochs=500,
            batch_size=20,
        )
        assert m3gnet.cutoff == 6.0
        assert m3gnet.max_epochs == 500
        assert m3gnet.batch_size == 20
        
        hypers = MLIPHypers(M3GNET=m3gnet)
        assert hypers.M3GNET.cutoff == 6.0
        assert hypers.M3GNET.max_epochs == 500

    def test_nequip_settings_work_correctly(self):
        """Verify NEQUIP settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import NEQUIPSettings, MLIPHypers
        
        nequip = NEQUIPSettings(
            r_max=5.0,
            num_layers=6,
            max_epochs=5000,
        )
        assert nequip.r_max == 5.0
        assert nequip.num_layers == 6
        assert nequip.max_epochs == 5000
        
        hypers = MLIPHypers(NEQUIP=nequip)
        assert hypers.NEQUIP.r_max == 5.0
        assert hypers.NEQUIP.num_layers == 6

    def test_jace_settings_work_correctly(self):
        """Verify J-ACE settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import JACESettings, MLIPHypers
        
        jace = JACESettings(
            order=4,
            totaldegree=8,
            cutoff=3.0,
        )
        assert jace.order == 4
        assert jace.totaldegree == 8
        assert jace.cutoff == 3.0
        
        # Test with alias
        hypers = MLIPHypers(**{"J-ACE": jace})
        assert hypers.J_ACE.order == 4
        assert hypers.J_ACE.cutoff == 3.0

    def test_mlip_hypers_from_dict_all_types(self):
        """Test that MLIPHypers can be created from dict for all MLIP types."""
        from autoplex.settings import MLIPHypers
        
        # Simulate loading from YAML (all values are dicts)
        hypers_dict = {
            "GAP": {
                "soap": {"l_max": 10, "cutoff": 5.5}
            },
            "MACE": {
                "batch_size": 15,
                "r_max": 5.5,
            },
            "P-ACE": {
                "cutoff": 8.0,
                "seed": 42,
            },
            "NEP": {
                "generation": 60000,
            },
            "M3GNET": {
                "cutoff": 5.5,
            },
            "NEQUIP": {
                "r_max": 4.5,
            },
            "J-ACE": {
                "order": 4,
            },
        }
        
        hypers = MLIPHypers(**hypers_dict)
        
        # Verify all values are correctly set
        assert hypers.GAP.soap.l_max == 10
        assert hypers.MACE.batch_size == 15
        assert hypers.P_ACE.cutoff == 8.0
        assert hypers.NEP.generation == 60000
        assert hypers.M3GNET.cutoff == 5.5
        assert hypers.NEQUIP.r_max == 4.5
        assert hypers.J_ACE.order == 4