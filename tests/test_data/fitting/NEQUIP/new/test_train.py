import sys
import pytest
import yaml
from pathlib import Path
from shutil import which
from jobflow import run_locally
from autoplex.fitting.common.flows import MLIPFitMaker
from tests.auto.phonons.test_jobs import fake_run_vasp_kwargs
import subprocess


if sys.version_info[:2] == (3, 10):
    try:
        from nequip.ase import NequIPCalculator
        has_nequip=True
    except:
        has_nequip=False
else:
    try:
        from nequip.integrations.ase import NequIPCalculator
        has_nequip=True
    except:
        has_nequip=False
        
def test_nequip_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"
    
    is_old_nequip = not hasattr(NequIPCalculator, "from_compiled_model")
    
    if is_old_nequip:
        model_kwargs = {
            "r_max":3.14,
            "max_epochs":10,
            "device": "cpu",
        }
    else:
        model_kwargs = {
            "device": "cpu",
            "cutoff_radius": 4,
            "data": {
                "split_dataset": {"train": 0.9, "val": 0.1},
                "train_dataloader": {"num_workers": 1, "batch_size": 1},
            },
            "trainer": {"max_epochs": 1}
        }

    nequipfit = MLIPFitMaker(
        mlip_type="NEQUIP",
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
        **model_kwargs
    )

    run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()