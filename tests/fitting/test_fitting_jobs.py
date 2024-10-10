from __future__ import annotations
from autoplex.fitting.common.flows import MLIPFitMaker
import shutil
from pathlib import Path
from jobflow import run_locally


def test_gap_fit_maker(test_dir, memory_jobstore):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    gapfit = MLIPFitMaker().make(
        auto_delta=True,
        glue_xml=False,
        twob={"cutoff": 6.84,
              "n_sparse": 30,},
        threeb={"cutoff": 2.223},
        soap={"l_max": 8,
              "n_max": 8,
              "atom_sigma": 0.75,
              "n_sparse": 2000,
              "cutoff": 3.42},
        general={"three_body": True},
        apply_data_preprocessing=False,
        database_dir=database_dir    
        )

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_jace_fit_maker(test_dir, memory_jobstore):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    jacefit = MLIPFitMaker(
        mlip_type="J-ACE",
    ).make(
        isolated_atom_energies={14: -0.84696938},
        num_processes_fit=4,
        apply_data_preprocessing=False,
        database_dir=database_dir,
        order=2,
        totaldegree=4,
    )

    responses = run_locally(
        jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(jacefit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_nqeuip_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
       mlip_type="NEQUIP",
    ).make(
        isolated_atom_energies={14: -0.84696938},
        num_processes_fit=1,
        apply_data_preprocessing=False,
        database_dir=database_dir,
        r_max=3.14,
        max_epochs=10,
        device="cpu",
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_m3gnet_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
            mlip_type="M3GNET",
    ).make(
        isolated_atom_energies={14: -0.84696938},
        num_processes_fit=1,
        apply_data_preprocessing=False,
        database_dir=database_dir,
        cutoff=3.0,
        threebody_cutoff=2.0,
        batch_size=1,
        max_epochs=3,
        include_stresses=True,
        hidden_dim=8,
        num_units=8,
        max_l=4,
        max_n=4,
        device="cpu",
        test_equal_to_val=True,
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_mace_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
                mlip_type="MACE",
    ).make(
        isolated_atom_energies={14: -0.84696938},
        num_processes_fit=1,
        apply_data_preprocessing=False,
        database_dir=database_dir,
        model="MACE",
        config_type_weights='{"Default":1.0}',
        hidden_irreps="32x0e + 32x1o",
        r_max=3.0,
        batch_size=5,
        max_num_epochs=10,
        start_swa=5,
        ema_decay=0.99,
        correlation=3,
        loss="huber",
        default_dtype="float32",
        device="cpu",
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)

