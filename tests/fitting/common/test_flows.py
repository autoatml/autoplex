import pytest
from autoplex.fitting.common.flows import MLIPFitMaker


@pytest.fixture(scope="class")
def fit_input_dict(vasp_test_dir):
    return {
        "mp-22905": {
            "rattled_dir": [
                [
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_1"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_2"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_3"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_dir": [
                [
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "phonon_static_1"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "phonon_static_2"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_data": [],
        },
        "IsolatedAtom": {
            "iso_atoms_dir": [
                [
                    (vasp_test_dir / "Li_iso_atoms" / "Li-statisoatom" / "outputs")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Cl_iso_atoms" / "Cl-statisoatom" / "outputs")
                    .absolute()
                    .as_posix(),
                ]
            ]
        },
    }


@pytest.fixture(scope="class")
def fit_input_dict_glue_xml(vasp_test_dir):
    return {
        "mp-149": {
            "rattled_dir": [
                [
                    (vasp_test_dir / "Si_glue_xml_fit" / "rattled_supercell_1")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "rattled_supercell_2")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "rattled_supercell_3")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "rattled_supercell_4")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "rattled_supercell_5")
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_dir": [
                [
                    (vasp_test_dir / "Si_glue_xml_fit" / "phonon_supercell_1")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "phonon_supercell_2")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "phonon_supercell_3")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "phonon_supercell_4")
                    .absolute()
                    .as_posix(),
                    (vasp_test_dir / "Si_glue_xml_fit" / "phonon_supercell_5")
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_data": [],
        },
        "IsolatedAtom": {
            "iso_atoms_dir": [
                [
                    (vasp_test_dir / "Si_glue_xml_fit" / "iso_atom")
                    .absolute()
                    .as_posix(),
                ]
            ]
        },
    }


def test_mlip_fit_maker(test_dir, clean_dir, memory_jobstore, vasp_test_dir, fit_input_dict):
    from pathlib import Path
    from jobflow import run_locally

    # Test to check if gap fit runs with default hyperparameter sets (i.e. two_body and soap is True)
    gapfit = MLIPFitMaker(apply_data_preprocessing=True).make(
        species_list=["Li", "Cl"],
        fit_input=fit_input_dict,
    )

    _ = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_with_kwargs(
        test_dir, clean_dir, memory_jobstore, vasp_test_dir, fit_input_dict
):
    from pathlib import Path
    from jobflow import run_locally

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker(
        auto_delta=False,
        glue_xml=False,
        apply_data_preprocessing=True,
        split_ratio=0.4,
        regularization=False,
        distillation=True,
        force_max=40,
    ).make(
        species_list=["Li", "Cl"],
        fit_input=fit_input_dict,
        twob={"delta": 2.0, "cutoff": 8},
        threeb={"n_sparse": 100},
    )

    _ = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_with_pre_database_dir(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from ase.io import read
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()
    train_atoms_before = read(test_files_dir / "pre_xyz_train.extxyz", ':')

    test_atoms_before = read(test_files_dir / "pre_xyz_test.extxyz", ':')


    # Test if gap fit runs with pre_database_dir
    gapfit = MLIPFitMaker(
        pre_database_dir=str(test_files_dir),
        apply_data_preprocessing=True,
        pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
    ).make(
        species_list=["Li", "Cl"],
        fit_input=fit_input_dict,
    )

    run_locally(gapfit, ensure_success=True, create_folders=True, store=memory_jobstore)

    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()

    train_atoms = read(Path(gapfit.output["database_dir"].resolve(memory_jobstore))/"train.extxyz",':')
    test_atoms = read(Path(gapfit.output["database_dir"].resolve(memory_jobstore))/"test.extxyz",':')
    assert (len(train_atoms_before) +len(test_atoms_before)+7) == (len(train_atoms) +len(test_atoms))


    train_atoms = read(Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore))/"train.extxyz",':')
    test_atoms = read(Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore))/ "test.extxyz",':')
    assert (len(train_atoms_before) +len(test_atoms_before)+7) == (len(train_atoms) +len(test_atoms))


    train_atoms = read(Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore))/"quip_train.extxyz",':')
    test_atoms = read(Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore))/ "quip_test.extxyz",':')
    assert (len(train_atoms_before) +len(test_atoms_before)+7) == (len(train_atoms) +len(test_atoms))


def test_mlip_fit_maker_jace(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()

    # Test julia-ACE fit runs with pre_database_dir
    jacefit = MLIPFitMaker(
        mlip_type="J-ACE",
        pre_database_dir=str(test_files_dir),
        pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
        apply_data_preprocessing=True,
        num_processes_fit=4,
    ).make(
        isolated_atom_energies={3: -0.28649227, 17: -0.25638457},
        fit_input=fit_input_dict,
        order=3,
        totaldegree=6,
        cutoff=2.0,
        solver="BLR",
    )

    run_locally(
        jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if julia-ACE potential file is generated
    assert Path(jacefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_nequip(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()

    # Test NEQUIP fit runs with pre_database_dir
    nequipfit = MLIPFitMaker(
        mlip_type="NEQUIP",
        pre_database_dir=str(test_files_dir),
        pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
        num_processes_fit=1,
        apply_data_preprocessing=True,
    ).make(
        fit_input=fit_input_dict,
        isolated_atom_energies={3: -0.28649227, 17: -0.25638457},
        r_max=3.14,
        max_epochs=10,
        device="cpu",
    )

    run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if NEQUIP potential file is generated
    assert Path(nequipfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_m3gnet(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()

    # Test if M3GNET fit runs with pre_database_dir
    m3gnetfit = MLIPFitMaker(
        mlip_type="M3GNET",
        pre_database_dir=str(test_files_dir),
        pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
        num_processes_fit=1,
        apply_data_preprocessing=True,
    ).make(
        fit_input=fit_input_dict,
        isolated_atom_energies={3: -0.28649227, 17: -0.25638457},
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

    # check if M3GNET potential file is generated
    assert Path(m3gnetfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_mace(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()

    # Test if MACE fit runs with pre_database_dir
    macefit = MLIPFitMaker(
        mlip_type="MACE",
        pre_database_dir=str(test_files_dir),
        pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
        num_processes_fit=1,
        apply_data_preprocessing=True,
    ).make(
        fit_input=fit_input_dict,
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

    run_locally(
        macefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if MACE potential file is generated
    assert Path(macefit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_glue_xml(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict_glue_xml, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    glue_file = test_dir / "fitting" / "glue.xml"

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker(
        mlip_type="GAP",
        glue_file_path=glue_file,
        apply_data_preprocessing=True,
        auto_delta=False,
        glue_xml=True,
    ).make(
        species_list=["Si"],
        fit_input=fit_input_dict_glue_xml,
        general={"core_param_file": "glue.xml", "core_ip_args": "{IP Glue}"},
    )

    _ = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    assert Path(f"{gapfit.output['mlip_path'][0].resolve(memory_jobstore)}/glue.xml").exists()
    with open(Path(f"{gapfit.output['mlip_path'][0].resolve(memory_jobstore)}/std_gap_out.log"), "r") as file:
        content = file.read()
        assert "core_param_file = glue.xml" in content


def test_mlip_fit_maker_glue_xml_with_other_name(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict_glue_xml, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    glue_file = test_dir / "fitting" / "test_glue.xml"

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker(
        mlip_type="GAP",
        glue_file_path=glue_file,
        auto_delta=False,
        glue_xml=True,
        apply_data_preprocessing=True,
    ).make(
        species_list=["Si"],
        fit_input=fit_input_dict_glue_xml,
        general={"core_param_file": "glue.xml", "core_ip_args": "{IP Glue}"},
    )

    _ = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()


def test_mlip_fit_maker_with_automated_separated_dataset(
        test_dir, memory_jobstore, vasp_test_dir, fit_input_dict, clean_dir
):
    from pathlib import Path
    from jobflow import run_locally

    test_files_dir = Path(test_dir / "fitting").resolve()

    # Test if gap fit runs with pre_database_dir
    gapfit = MLIPFitMaker(
        pre_database_dir=str(test_files_dir),
        pre_xyz_files=["pre_xyz_train_more_data.extxyz", "pre_xyz_test_more_data.extxyz"],
        apply_data_preprocessing=True,
        separated=True
    ).make(
        species_list=["Li", "Cl"],
        fit_input=fit_input_dict,
    )

    run_locally(gapfit, ensure_success=True, create_folders=True, store=memory_jobstore)

    # check if gap potential file is generated
    assert Path(gapfit.output["mlip_path"][0].resolve(memory_jobstore)).exists()
    assert Path(gapfit.output["mlip_path"][1].resolve(memory_jobstore) + "/train.extxyz").exists()
    assert Path(gapfit.output["mlip_path"][2].resolve(memory_jobstore) + "/train.extxyz").exists()
    assert Path(gapfit.output["mlip_path"][2].resolve(memory_jobstore) + "/energy_forces.png").exists()
