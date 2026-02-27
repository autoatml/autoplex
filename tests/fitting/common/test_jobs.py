import pytest
from autoplex.fitting.common.flows import MLIPFitMaker
from pathlib import Path
from jobflow import run_locally

from tests.auto.phonons.test_jobs import fake_run_vasp_kwargs


def test_gap_auto_delta_fit_maker(test_dir, memory_jobstore, clean_dir):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    gapfit = MLIPFitMaker(
        auto_delta=True,
        glue_xml=False,
        apply_data_preprocessing=False,
    ).make(
        twob={"delta": 2.0, "cutoff": 4},
        threeb={"n_sparse": 10},
        database_dir=database_dir,
        general={"two_body": True,
                 "three_body": True}
        )

    run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    
    
def test_gap_fixed_delta_fit_maker(test_dir, memory_jobstore, clean_dir):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    gapfit = MLIPFitMaker(
        auto_delta=False,
        glue_xml=False,
        apply_data_preprocessing=False,
    ).make(
        twob={"delta": 2.0, "cutoff": 4},
        threeb={"delta": 1.5, "n_sparse": 10},
        database_dir=database_dir,
        general={"two_body": True,
                 "three_body": True}
        )

    run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_jace_fit_maker(test_dir, memory_jobstore, clean_dir):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    jacefit = MLIPFitMaker(
        mlip_type="J-ACE",
        num_processes_fit=4,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
        order=2,
        totaldegree=4,
    )

    run_locally(
        jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(jacefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_nequip_fit_maker(test_dir, memory_jobstore, clean_dir):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
        mlip_type="NEQUIP",
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
        r_max=3.14,
        max_epochs=10,
        device="cpu",
    )

    run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_m3gnet_fit_maker(test_dir, memory_jobstore, clean_dir):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    m3gnetfit = MLIPFitMaker(
        mlip_type="M3GNET",
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
        cutoff=3.0,
        threebody_cutoff=2.0,
        batch_size=1,
        max_epochs=3,
        include_stresses=True,
        dim_node_embedding=8,
        dim_edge_embedding=8,
        units=8,
        max_l=4,
        max_n=4,
        device="cpu",
        test_equal_to_val=True,
    )

    run_locally(
        m3gnetfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(m3gnetfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mace_fit_maker(test_dir, memory_jobstore, clean_dir, caplog):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    macefit = MLIPFitMaker(
        mlip_type="MACE",
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
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

    # save the screen output to a file for debugging
    with caplog.at_level("INFO"):
        run_locally(
            macefit, ensure_success=True, create_folders=True, store=memory_jobstore
        )

    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    # check if "Energies, forces and stresses are used for training and validation." is in the output
    #assert any("Energies, forces and stresses are used for training and validation." in message for message in caplog.mess)
    assert "Energies, forces and stresses are used for training and validation." in caplog.text
    assert "Energies are not used for training or validation." not in caplog.text  
    assert "Forces are not used for training or validation." not in caplog.text
    assert "Stresses are not used for training or validation." not in caplog.text

def test_mace_fit_maker_no_stress(test_dir, memory_jobstore, clean_dir, caplog):
    database_dir = test_dir / "fitting/rss_training_dataset_without_virials/"

    macefit = MLIPFitMaker(
        mlip_type="MACE",
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        isolated_atom_energies={14: -0.84696938},
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

    # save the screen output to a file for debugging
    with caplog.at_level("INFO"):
        run_locally(
            macefit, ensure_success=True, create_folders=True, store=memory_jobstore
        )

    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    # check if "Energies, forces and stresses are used for training and validation." is in the output
    #assert any("Energies, forces and stresses are used for training and validation." in message for message in caplog.mess)
    assert "Energies, forces and stresses are used for training and validation." not in caplog.text
    assert "Energies are not used for training or validation." not in caplog.text  
    assert "Forces are not used for training or validation." not in caplog.text
    assert "Stresses are not used for training or validation." in caplog.text


def test_mace_fit_maker_passEOs(test_dir, memory_jobstore, clean_dir, caplog):
    database_dir = test_dir / "fitting/rss_training_dataset/"
    macefit = MLIPFitMaker(
        mlip_type="MACE",
        ref_energy_name="REF_energy",
        ref_force_name="REF_forces",
        ref_virial_name="REF_virial",
        ref_stress_name="REF_stress",   
        num_processes_fit=1,
        apply_data_preprocessing=False,
    
    ).make(
        database_dir=database_dir,
        name="MACE_final",
        foundation_model="small",
        multiheads_finetuning=False,
        r_max = 6,
        loss = "huber",
        energy_weight = 1000.0,
        forces_weight = 1000.0,
        stress_weight = 1.0 ,
        compute_stress=False,
        E0s = "{14:-1.0}", # test if EOs can be passed.
        scaling = "rms_forces_scaling",
        batch_size = 1,
        valid_batch_size=1,
        max_num_epochs = 1,
        ema=True,
        ema_decay = 0.99,
        amsgrad=True,
        default_dtype = "float32",
        restart_latest=True,
        lr = 0.0001,
        patience = 20,
        device = "cpu",
        save_cpu =True,
        seed = 3,
        stress_key = "stress",
    )

    run_locally(
            macefit, ensure_success=True, create_folders=True, store=memory_jobstore
        )

    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    assert "Energies, forces and stresses are used for training and validation." in caplog.text
    assert "Stresses are not used for training or validation." not in caplog.text
    assert "Energies are not used for training or validation." not in caplog.text
    assert "Forces are not used for training or validation." not in caplog.text


def test_mace_finetuning_maker(test_dir, memory_jobstore, clean_dir, caplog):

    # TODO: Replace the finetuning dataset with correct reference keys
    # as per mace logs using 'energy', 'forces' and 'stress' as ref keys is 
    # no longer safe when communicating between MACE and ASE
    
    database_dir = test_dir / "fitting/rss_training_dataset/"

    # TODO: there seems to be something wrong here!
    macefit = MLIPFitMaker(
        mlip_type="MACE",
        ref_energy_name="REF_energy",
        ref_force_name="REF_forces",
        ref_virial_name="REF_virial",
        ref_stress_name="REF_stress",   
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        name="MACE_final",
        foundation_model="small",
        multiheads_finetuning=False,
        r_max = 6,
        loss = "huber",
        energy_weight = 1000.0,
        forces_weight = 1000.0,
        stress_weight = 1.0 ,
        compute_stress=True,
        E0s = "average",
        scaling = "rms_forces_scaling",
        batch_size = 1,
        max_num_epochs = 1,
        ema=True,
        ema_decay = 0.99,
        amsgrad=True,
        default_dtype = "float32",
        restart_latest=True,
        lr = 0.0001,
        patience = 20,
        device = "cpu",
        save_cpu =True,
        seed = 3,
        stress_key = "stress",
    )
    with caplog.at_level("INFO"):
        run_locally(
            macefit, ensure_success=True, create_folders=True, store=memory_jobstore
        )

    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    assert "Energies, forces and stresses are used for training and validation." in caplog.text
    assert "Stresses are not used for training or validation." not in caplog.text
    assert "Energies are not used for training or validation." not in caplog.text
    assert "Forces are not used for training or validation." not in caplog.text

def test_mace_finetuning_maker2(test_dir, memory_jobstore, clean_dir):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    macefit = MLIPFitMaker(
        mlip_type="MACE",
        ref_energy_name=None,
        ref_force_name=None,
        ref_virial_name=None,
        num_processes_fit=1,
        apply_data_preprocessing=False,
    ).make(
        database_dir=database_dir,
        name="MACE_final",
        foundation_model="small",
        multiheads_finetuning=False,
        r_max = 6,
        loss = "huber",
        energy_weight = 1000.0,
        forces_weight = 1000.0,
        stress_weight = 1.0 ,
        compute_stress=True,
        E0s = "average",
        scaling = "rms_forces_scaling",
        batch_size = 1,
        max_num_epochs = 1,
        ema=True,
        ema_decay = 0.99,
        amsgrad=True,
        default_dtype = "float64",
        restart_latest=True,
        lr = 0.0001,
        patience = 20,
        device = "cpu",
        save_cpu =True,
        seed = 3,
    )

    run_locally(
        macefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
