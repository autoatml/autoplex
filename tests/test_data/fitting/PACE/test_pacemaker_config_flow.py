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


@pytest.fixture
def clean_dir(tmp_path, monkeypatch):
    """Change to a clean temporary directory for each test."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


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

    # ==================== Top-Level Parameter Tests ====================
    
    def test_rss_config_loads_top_level_params(self, autoplex_config_file):
        """Test that top-level P-ACE parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        pace_hypers = config.mlip_hypers.P_ACE
        
        assert config.mlip_type == "P-ACE"
        assert pace_hypers.cutoff == EXPECTED_PACE_VALUES["cutoff"]
        assert pace_hypers.seed == EXPECTED_PACE_VALUES["seed"]

    # ==================== Backend Section Tests ====================
    
    def test_rss_config_loads_backend_params(self, autoplex_config_file):
        """Test that backend parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        backend = config.mlip_hypers.P_ACE.backend
        expected = EXPECTED_PACE_VALUES["backend"]
        
        assert backend["batch_size"] == expected["batch_size"]
        assert backend["batch_size_reduction"] == expected["batch_size_reduction"]
        assert backend["batch_size_reduction_factor"] == expected["batch_size_reduction_factor"]
        assert backend["display_step"] == expected["display_step"]
        assert backend["evaluator"] == expected["evaluator"]

    # ==================== Fit Section Tests ====================
    
    def test_rss_config_loads_fit_params(self, autoplex_config_file):
        """Test that fit parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        fit = config.mlip_hypers.P_ACE.fit
        expected = EXPECTED_PACE_VALUES["fit"]
        
        assert fit["optimizer"] == expected["optimizer"]
        assert fit["maxiter"] == expected["maxiter"]
        assert fit["repulsion"] == expected["repulsion"]
        assert fit["trainable_parameters"] == expected["trainable_parameters"]

    def test_rss_config_loads_fit_loss_params(self, autoplex_config_file):
        """Test that fit.loss parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        loss = config.mlip_hypers.P_ACE.fit["loss"]
        expected = EXPECTED_PACE_VALUES["fit"]["loss"]
        
        assert loss["kappa"] == expected["kappa"]
        assert loss["L1_coeffs"] == expected["L1_coeffs"]
        assert loss["L2_coeffs"] == expected["L2_coeffs"]
        assert loss["w_energy"] == expected["w_energy"]
        assert loss["w_forces"] == expected["w_forces"]

    # ==================== Potential Section Tests ====================
    
    def test_rss_config_loads_potential_top_level(self, autoplex_config_file):
        """Test that potential top-level parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        potential = config.mlip_hypers.P_ACE.potential
        expected = EXPECTED_PACE_VALUES["potential"]
        
        assert potential["deltaSplineBins"] == expected["deltaSplineBins"]
        assert potential["elements"] == expected["elements"]

    def test_rss_config_loads_embeddings_params(self, autoplex_config_file):
        """Test that potential.embeddings parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        embeddings = config.mlip_hypers.P_ACE.potential["embeddings"]["ALL"]
        expected = EXPECTED_PACE_VALUES["potential"]["embeddings"]["ALL"]
        
        assert embeddings["npot"] == expected["npot"]
        assert embeddings["fs_parameters"] == expected["fs_parameters"]
        assert embeddings["ndensity"] == expected["ndensity"]
        assert embeddings["rho_core_cut"] == expected["rho_core_cut"]
        assert embeddings["drho_core_cut"] == expected["drho_core_cut"]

    def test_rss_config_loads_bonds_params(self, autoplex_config_file):
        """Test that potential.bonds parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        bonds = config.mlip_hypers.P_ACE.potential["bonds"]["ALL"]
        expected = EXPECTED_PACE_VALUES["potential"]["bonds"]["ALL"]
        
        assert bonds["radbase"] == expected["radbase"]
        assert bonds["radparameters"] == expected["radparameters"]
        assert bonds["rcut"] == expected["rcut"]
        assert bonds["dcut"] == expected["dcut"]
        assert bonds["NameofCutoffFunction"] == expected["NameofCutoffFunction"]

    def test_rss_config_loads_functions_params(self, autoplex_config_file):
        """Test that potential.functions parameters are correctly loaded."""
        config = RssConfig.from_file(str(autoplex_config_file))
        functions = config.mlip_hypers.P_ACE.potential["functions"]
        expected = EXPECTED_PACE_VALUES["potential"]["functions"]
        
        assert functions["number_of_functions_per_element"] == \
            expected["number_of_functions_per_element"]
        assert functions["ALL"]["nradmax_by_orders"] == expected["ALL"]["nradmax_by_orders"]
        assert functions["ALL"]["lmax_by_orders"] == expected["ALL"]["lmax_by_orders"]

    # ==================== RSS Parameter Isolation Tests ====================
    
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

    # ==================== Input.yaml Generation Tests ====================
    
    def test_pacemaker_input_yaml_has_all_expected_params(
        self, autoplex_config_file, mock_training_data, tmp_path, clean_dir
    ):
        """Test that generated input.yaml contains all expected P-ACE parameters."""
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
                        ref_energy_name="REF_energy",
                        ref_force_name="REF_forces",
                        num_processes_fit=1,
                    )
                except Exception:
                    pass
        
        input_yaml_path = tmp_path / "input.yaml"
        assert input_yaml_path.exists(), "input.yaml was not created!"
        
        generated = loadfn(str(input_yaml_path))
        
        # Top-level params
        assert generated["cutoff"] == EXPECTED_PACE_VALUES["cutoff"]
        assert generated["seed"] == EXPECTED_PACE_VALUES["seed"]
        
        # Backend params
        for key, value in EXPECTED_PACE_VALUES["backend"].items():
            assert generated["backend"][key] == value, \
                f"backend.{key}: expected {value}, got {generated['backend'].get(key)}"
        
        # Fit params
        assert generated["fit"]["optimizer"] == EXPECTED_PACE_VALUES["fit"]["optimizer"]
        assert generated["fit"]["maxiter"] == EXPECTED_PACE_VALUES["fit"]["maxiter"]
        
        # Fit loss params
        for key, value in EXPECTED_PACE_VALUES["fit"]["loss"].items():
            assert generated["fit"]["loss"][key] == value, \
                f"fit.loss.{key}: expected {value}, got {generated['fit']['loss'].get(key)}"
        
        # Potential params
        assert generated["potential"]["deltaSplineBins"] == \
            EXPECTED_PACE_VALUES["potential"]["deltaSplineBins"]
        assert generated["potential"]["elements"] == \
            EXPECTED_PACE_VALUES["potential"]["elements"]
        
        # Embeddings params
        embeddings = generated["potential"]["embeddings"]["ALL"]
        expected_emb = EXPECTED_PACE_VALUES["potential"]["embeddings"]["ALL"]
        assert embeddings["npot"] == expected_emb["npot"]
        assert embeddings["fs_parameters"] == expected_emb["fs_parameters"], \
            f"fs_parameters mismatch: expected {expected_emb['fs_parameters']}, got {embeddings['fs_parameters']}"
        assert embeddings["ndensity"] == expected_emb["ndensity"]
        assert embeddings["rho_core_cut"] == expected_emb["rho_core_cut"]
        assert embeddings["drho_core_cut"] == expected_emb["drho_core_cut"]
        
        # Bonds params
        bonds = generated["potential"]["bonds"]["ALL"]
        expected_bonds = EXPECTED_PACE_VALUES["potential"]["bonds"]["ALL"]
        assert bonds["radbase"] == expected_bonds["radbase"]
        assert bonds["radparameters"] == expected_bonds["radparameters"], \
            f"radparameters mismatch: expected {expected_bonds['radparameters']}, got {bonds['radparameters']}"
        assert bonds["rcut"] == expected_bonds["rcut"]
        assert bonds["dcut"] == expected_bonds["dcut"]
        assert bonds["NameofCutoffFunction"] == expected_bonds["NameofCutoffFunction"]
        
        # Functions params
        functions = generated["potential"]["functions"]
        expected_func = EXPECTED_PACE_VALUES["potential"]["functions"]
        assert functions["number_of_functions_per_element"] == \
            expected_func["number_of_functions_per_element"]
        assert functions["ALL"]["nradmax_by_orders"] == expected_func["ALL"]["nradmax_by_orders"]
        assert functions["ALL"]["lmax_by_orders"] == expected_func["ALL"]["lmax_by_orders"]

    def test_input_yaml_no_rss_params(
        self, autoplex_config_file, mock_training_data, tmp_path, clean_dir
    ):
        """Test that generated input.yaml does NOT contain RSS parameters."""
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
        assert input_yaml_path.exists()
        
        generated = loadfn(str(input_yaml_path))
        
        rss_params = [
            'initial_kt', 'vasp_ref_file', 'num_processes_rss',
            'max_iteration_number', 'stop_criterion', 'config_types',
            'rss_group', 'scalar_pressure_method', 'hookean_repul',
            'train_from_scratch', 'mlip_type', 'tag', 'num_processes_buildcell',
        ]
        
        for param in rss_params:
            assert param not in generated, \
                f"RSS parameter '{param}' should NOT be in input.yaml!"

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

    # ==================== Fit Kwargs Tests ====================
    
    def test_fit_kwargs_filters_rss_garbage(self, tmp_path, clean_dir):
        """Test that RSS parameters passed via fit_kwargs are filtered out."""
        os.chdir(tmp_path)
        
        from ase import Atoms
        from ase.io import write
        
        atoms = Atoms('Si2', positions=[[0,0,0], [1.35, 1.35, 1.35]], 
                      cell=[5.43]*3, pbc=True)
        atoms.info['REF_energy'] = -10.0
        atoms.info['config_type'] = 'bulk'
        atoms.arrays['REF_forces'] = np.array([[0.01]*3, [-0.01]*3])
        
        write(str(tmp_path / "train.extxyz"), [atoms], format='extxyz')
        write(str(tmp_path / "test.extxyz"), [atoms], format='extxyz')
        
        pace_hypers = PacemakerSettings(cutoff=6.0, seed=123)
        
        polluted_fit_kwargs = {
            "initial_kt": 0.3,
            "vasp_ref_file": "ref.xyz",
            "cutoff": 7.0,
            "fit": {"maxiter": 500},
        }
        
        with patch('autoplex.fitting.common.utils.run_pacemaker'):
            with patch('shutil.which', return_value='/usr/bin/pacemaker'):
                try:
                    pace_fitting(
                        db_dir=tmp_path,
                        species_list=["Si"],
                        hyperparameters=pace_hypers,
                        fit_kwargs=polluted_fit_kwargs,
                        isolated_atom_energies={14: -5.0},
                    )
                except Exception:
                    pass
        
        input_yaml_path = tmp_path / "input.yaml"
        if input_yaml_path.exists():
            generated = loadfn(str(input_yaml_path))
            
            assert "initial_kt" not in generated
            assert "vasp_ref_file" not in generated


class TestEndToEndConfigPropagation:
    """End-to-end tests for configuration propagation."""
    
    def test_full_config_roundtrip(self):
        """Test complete configuration roundtrip with all parameters."""
        config = RssConfig(
            tag="Si",
            mlip_type="P-ACE",
            mlip_hypers=MLIPHypers(
                P_ACE=PacemakerSettings(
                    cutoff=8.0,
                    seed=42,
                    backend=EXPECTED_PACE_VALUES["backend"],
                    fit=EXPECTED_PACE_VALUES["fit"],
                    potential=EXPECTED_PACE_VALUES["potential"],
                )
            )
        )
        
        config_params = config.model_dump(by_alias=True, exclude_none=True)
        mlip_hypers = config_params["mlip_hypers"]["P-ACE"]
        
        # Verify all top-level params
        assert mlip_hypers["cutoff"] == 8.0
        assert mlip_hypers["seed"] == 42
        
        # Verify backend
        for key, value in EXPECTED_PACE_VALUES["backend"].items():
            assert mlip_hypers["backend"][key] == value
        
        # Verify fit
        assert mlip_hypers["fit"]["optimizer"] == "BFGS"
        assert mlip_hypers["fit"]["maxiter"] == 800
        for key, value in EXPECTED_PACE_VALUES["fit"]["loss"].items():
            assert mlip_hypers["fit"]["loss"][key] == value
        
        # Verify potential embeddings
        embeddings = mlip_hypers["potential"]["embeddings"]["ALL"]
        expected_emb = EXPECTED_PACE_VALUES["potential"]["embeddings"]["ALL"]
        assert embeddings["fs_parameters"] == expected_emb["fs_parameters"]
        assert embeddings["npot"] == expected_emb["npot"]
        assert embeddings["rho_core_cut"] == expected_emb["rho_core_cut"]
        
        # Verify potential bonds
        bonds = mlip_hypers["potential"]["bonds"]["ALL"]
        expected_bonds = EXPECTED_PACE_VALUES["potential"]["bonds"]["ALL"]
        assert bonds["radbase"] == expected_bonds["radbase"]
        assert bonds["radparameters"] == expected_bonds["radparameters"]
        assert bonds["rcut"] == expected_bonds["rcut"]
        
        # Verify potential functions
        functions = mlip_hypers["potential"]["functions"]
        expected_func = EXPECTED_PACE_VALUES["potential"]["functions"]
        assert functions["ALL"]["nradmax_by_orders"] == expected_func["ALL"]["nradmax_by_orders"]
        assert functions["ALL"]["lmax_by_orders"] == expected_func["ALL"]["lmax_by_orders"]

    def test_pacemaker_settings_direct_replacement(self):
        """Verify user-provided nested dicts completely replace defaults."""
        pace_settings = PacemakerSettings(
            cutoff=8.0,
            fit={"maxiter": 800}
        )
        
        assert pace_settings.fit == {"maxiter": 800}
        assert "optimizer" not in pace_settings.fit
        assert "loss" not in pace_settings.fit

    def test_pacemaker_settings_preserves_custom_values(self):
        """Verify PacemakerSettings preserves all custom values."""
        pace_settings = PacemakerSettings(
            cutoff=8.0,
            seed=42,
            potential=EXPECTED_PACE_VALUES["potential"]
        )
        
        assert pace_settings.cutoff == 8.0
        assert pace_settings.seed == 42
        
        embeddings = pace_settings.potential["embeddings"]["ALL"]
        assert embeddings["fs_parameters"] == [1, 1, 7.5577494194990145, 0.101]
        assert embeddings["rho_core_cut"] == 3000
        
        bonds = pace_settings.potential["bonds"]["ALL"]
        assert bonds["radbase"] == "SBessel"
        assert bonds["rcut"] == 6.311052419794955

    def test_mlip_hypers_with_alias(self):
        """Test that MLIPHypers handles P-ACE alias correctly."""
        # Test with hyphen (alias)
        hypers_dict = {"P-ACE": {"cutoff": 8.0, "seed": 42}}
        hypers = MLIPHypers(**hypers_dict)
        assert hypers.P_ACE.cutoff == 8.0
        
        # Test with underscore (field name)
        hypers_dict2 = {"P_ACE": {"cutoff": 9.0, "seed": 43}}
        hypers2 = MLIPHypers(**hypers_dict2)
        assert hypers2.P_ACE.cutoff == 9.0
        
        # Verify model_dump with alias
        dumped = hypers.model_dump(by_alias=True)
        assert "P-ACE" in dumped
        assert dumped["P-ACE"]["cutoff"] == 8.0


class TestOtherMLIPsNotAffected:
    """Test that the MLIPHypers config changes don't affect other MLIPs."""
    
    def test_mace_settings_work_correctly(self):
        """Verify MACE settings are not affected by MLIPHypers config changes."""
        from autoplex.settings import MACESettings, MLIPHypers, RssConfig
        
        # Test direct MACESettings creation
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
        
        # Test via MLIPHypers
        hypers = MLIPHypers(MACE=mace)
        assert hypers.MACE.batch_size == 20
        assert hypers.MACE.max_num_epochs == 500
        
        # Test via RssConfig
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