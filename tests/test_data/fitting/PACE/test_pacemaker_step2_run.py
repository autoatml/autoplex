import pytest
import shutil
import yaml
from pathlib import Path
from shutil import which
from autoplex import PacemakerSettings
from autoplex.fitting.common.utils import pace_fitting

pacemaker_avail = which("pacemaker")

@pytest.mark.skipif(not pacemaker_avail, reason="pacemaker executable not found in path")
def test_autoplex_pace_fitting_function(test_dir, tmp_path, monkeypatch):
    """
    Task 2 (Corrected): Directly test the 'pace_fitting' function from autoplex.
    This ensures the internal logic (data conversion -> input.yaml -> pacemaker -> yace conversion)
    works as a cohesive unit.
    """
    # 1. Setup Data Paths
    fitting_dir = test_dir / "fitting/rss_training_dataset"
    source_train = fitting_dir / "train.extxyz"
    pre_xyz = fitting_dir / "pre_xyz_train.extxyz"
    
    if pre_xyz.exists():
        train_filename = "pre_xyz_train.extxyz"
    elif source_train.exists():
        train_filename = "train.extxyz"
    else:
        pytest.skip("Training data not found in fitting/rss_training_dataset")

    test_filename = "test.extxyz"
    has_test = (fitting_dir / test_filename).exists()

    # 2. Prepare Settings
    ref_yaml_path = test_dir / "fitting/PACE/Si_input.yaml"
    if not ref_yaml_path.exists():
        pytest.skip("Si_input.yaml reference file match")

    with open(ref_yaml_path, 'r') as f:
        ref_dict = yaml.safe_load(f)

    # Initialize Settings
    pacemaker_settings = PacemakerSettings()
    
    # Inject minimal settings
    pacemaker_settings.cutoff = ref_dict.get('cutoff', 8.0)
    pacemaker_settings.backend = ref_dict.get('backend', {})
    pacemaker_settings.potential = ref_dict.get('potential', {})
    
    # Speed up fitting
    fit_config = ref_dict.get('fit', {})
    fit_config['maxiter'] = 5 
    if 'ladder_step' in fit_config: del fit_config['ladder_step']
    pacemaker_settings.fit = fit_config
    
    # 3. Execution Environment
    monkeypatch.chdir(tmp_path)
    
    isolated_atom_energies = {14: -0.84696938} # Si
    species = ["Si"]

    print(f"\nCalling pace_fitting in {tmp_path}...")
    
    # --- CORE TEST ---
    result = pace_fitting(
        db_dir=fitting_dir,       
        species_list=species,
        hyperparameters=pacemaker_settings,
        fit_kwargs=None,
        isolated_atom_energies=isolated_atom_energies,
        train_name=train_filename, 
        test_name=test_filename if has_test else "test.extxyz"
    )
    
    # 4. Verification
    print("\nResult from pace_fitting:", result)
    
    # Check intermediate files
    assert (tmp_path / "train.pckl.gzip").exists()
    assert (tmp_path / "input.yaml").exists()
    assert (tmp_path / "pacemaker.log").exists()
    
    # --- FIXED VERIFICATION LOGIC ---
    # The mlip_path returns the DIRECTORY where the potential resides
    fit_dir = Path(result["mlip_path"])
    print(f"Fit Directory returned: {fit_dir}")
    
    assert fit_dir.exists() and fit_dir.is_dir(), "Returned mlip_path should be an existing directory"
    
    # We verify the files exist INSIDE that directory
    if which("pace_yaml2yace"):
        expected_pot = fit_dir / "output_potential.yace"
        assert expected_pot.exists(), "output_potential.yace should exist in the returned directory"
        assert expected_pot.stat().st_size > 0, "output_potential.yace should not be empty"
        print(f"Found valid YACE potential: {expected_pot}")
    else:
        # Fallback if converter not installed
        expected_pot = fit_dir / "output_potential.yaml"
        assert expected_pot.exists(), "output_potential.yaml should exist"
        print(f"Found valid YAML potential (converter missing): {expected_pot}")

    print("\n[SUCCESS] Autoplex pace_fitting function works correctly end-to-end.")