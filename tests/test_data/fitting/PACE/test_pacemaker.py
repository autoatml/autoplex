import pytest
from pathlib import Path
from shutil import which
from jobflow import run_locally
from autoplex.fitting.common.flows import MLIPFitMaker
from ase.io import read

# Check availability
pacemaker_avail = which("pacemaker")

@pytest.mark.skipif(not pacemaker_avail, reason="pacemaker executable not found in path")
def test_pacemaker_fit_maker_rss_data(test_dir, memory_jobstore, clean_dir):
    """
    Test the MLIPFitMaker utilizing the standard RSS training dataset.
    Uses configuration based on a standard Si_input.yaml provided by the user.
    """
    
    # 1. Path to common test data
    database_dir = test_dir / "fitting/rss_training_dataset/"
    train_file = database_dir / "train.extxyz"
    
    if not train_file.exists():
        pytest.skip(f"Test data not found at {train_file}")

    # 2. Get Species
    # We still read the file just to get the species list robustly
    atoms = read(str(train_file), index=0)
    species = sorted(list(set(atoms.get_chemical_symbols()))) # Should be ['Si']

    # 3. Construct the Maker
    # mimicking test_jace_fit_maker structure but with explicit Pacemaker config
    pacemaker_fit = MLIPFitMaker(
        mlip_type="P-ACE",
        num_processes_fit=1,
        apply_data_preprocessing=False,
        ref_energy_name="REF_energy", # User confirmed these keys
        ref_force_name="REF_forces",
    ).make(
        database_dir=database_dir,
        species_list=species, 
        isolated_atom_energies={14: -0.84696938},
        
        # --- Pacemaker Configuration (adapted from Si_input.yaml) ---
        cutoff=6.31, # specific rcut from bond settings in reference yaml
        
        fit={
            "maxiter": 5, # REDUCED for testing speed (orig: 800)
            "optimizer": "BFGS",
            "repulsion": "auto",
            "trainable_parameters": "ALL",
            # "ladder_step": 600, # skipped for short test
            # "ladder_type": "power_order",
            "loss": {
                "L1_coeffs": 1.0e-08,
                "L2_coeffs": 1.0e-08,
                "kappa": 0.75,
                "w0_rad": 1.0e-08,
                "w1_coeffs": 0,
                "w1_rad": 1.0e-08,
                "w2_coeffs": 0,
                "w2_rad": 1.0e-08
            }
        },
        
        potential={
            "elements": species,
            "embeddings": {
                "ALL": {
                    "npot": "FinnisSinclairShiftedScaled",
                    "fs_parameters": [1, 1, 7.5577, 0.101],
                    "ndensity": 2,
                    "rho_core_cut": 3000,
                    "drho_core_cut": 500
                }
            },
            "bonds": {
                "ALL": {
                    "dcut": 0.0189,
                    "radbase": "SBessel",
                    "radparameters": [3.3135],
                    "rcut": 6.311,
                    # "NameofCutoffFunction": "cos" # Optional depending on version
                }
            },
            "functions": {
                "number_of_functions_per_element": 100, # Reduced from 3000 for speed
                "ALL": {
                    "nradmax_by_orders": [15, 6, 4, 2], # Simplified basis for speed
                    "lmax_by_orders": [0, 2, 2, 1]
                }
            }
        },
        
        backend={
            "evaluator": "tensorpot",
            "batch_size_training": 100,
            "batch_size_evaluation": 200,
            "display_step": 1
        }
    )

    # 4. Run Locally
    run_locally(
        pacemaker_fit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # 5. Assertions
    result_path = pacemaker_fit.output["mlip_path"][0].resolve(memory_jobstore)
    assert Path(result_path).exists()
    
    # Check for specific Pacemaker artifacts
    assert (Path(result_path) / "input.yaml").exists()
    assert (Path(result_path) / "pacemaker.log").exists()
    # Check for outcome potential conversion
    # Note: If version <= 0.2.7 conversion might produce 'output_potential.yace' or 'output.yace'
    # Checking for the one standard in utils
    potential_file = Path(result_path) / "output_potential.yace"
    if not potential_file.exists():
         potential_file = Path(result_path) / "output.yace"
    
    assert potential_file.exists(), f"Neither output_potential.yace nor output.yace found in {result_path}"