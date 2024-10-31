from __future__ import annotations

from jobflow import Flow
from pymatgen.core.structure import Structure
from autoplex.auto.phonons.jobs import (
    get_iso_atom,
    dft_phonopy_gen_data,
    dft_random_gen_data,
    complete_benchmark
)
from autoplex.data.phonons.flows import TightDFTStaticMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.sets.core import TightRelaxSetGenerator
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from jobflow import run_locally
from tests.conftest import memory_jobstore
import pytest
from pytest import approx


@pytest.fixture(scope="class")
def relax_maker():
    return DoubleRelaxMaker.from_relax_maker(
        TightRelaxMaker(
            run_vasp_kwargs={"handlers": {}},
            input_set_generator=TightRelaxSetGenerator(
                user_incar_settings={
                    "ISPIN": 1,
                    "LAECHG": False,
                    "ISMEAR": 0,
                    "ENCUT": 700,
                    "ISYM": 0,
                    "SIGMA": 0.05,
                    "LCHARG": False,  # Do not write the CHGCAR file
                    "LWAVE": False,  # Do not write the WAVECAR file
                    "LVTOT": False,  # Do not write LOCPOT file
                    "LORBIT": 0,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    # to be removed
                    "NPAR": 4,
                }
            ),
        )
    )


@pytest.fixture(scope="class")
def ref_paths():
    return {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "dft static 1/2": "dft_ml_data_generation/phonon_static_1/",
        "dft static 2/2": "dft_ml_data_generation/phonon_static_2/",
        "dft static 1/3": "dft_ml_data_generation/rand_static_1/",
        "dft static 2/3": "dft_ml_data_generation/rand_static_2/",
        "dft static 3/3": "dft_ml_data_generation/rand_static_3/",
    }


@pytest.fixture(scope="class")
def ref_paths_check_sc_mat():
    return {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "dft static 1/2": "dft_ml_data_generation/phonon_static_1_sc_mat/",
        "dft static 2/2": "dft_ml_data_generation/phonon_static_2_sc_mat/",
        "dft static 1/3": "dft_ml_data_generation/rand_static_1_sc_mat/",
        "dft static 2/3": "dft_ml_data_generation/rand_static_2_sc_mat/",
        "dft static 3/3": "dft_ml_data_generation/rand_static_3_sc_mat/",
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs():
    return {
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft static 1/2": {"incar_settings": ["NSW"]},
        "dft static 2/2": {"incar_settings": ["NSW"]},
        "dft static 1/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "poscar", "kpoints", "potcar"],
        },
        "dft static 2/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "poscar", "kpoints", "potcar"],
        },
        "dft static 3/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "poscar", "kpoints", "potcar"],
        },
    }


def test_complete_benchmark(clean_dir, test_dir, memory_jobstore):
    from monty.serialization import loadfn
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from autoplex.fitting.common.flows import MLIPFitMaker
    database_dir = test_dir / "fitting/rss_training_dataset/"
    jobs = []
    gapfit = MLIPFitMaker().make(
        auto_delta=False,
        glue_xml=False,
        twob={"delta": 2.0, "cutoff": 4},
        threeb={"n_sparse": 10},
        preprocessing_data=False,
        database_dir=database_dir,
        separated=True
    )
    dft_data = loadfn(test_dir / "benchmark" / "phonon_doc_si.json")
    dft_doc: PhononBSDOSDoc = dft_data["output"]
    structure = dft_doc.structure

    jobs.append(gapfit)

    bm = complete_benchmark(ibenchmark_structure=0, benchmark_structure=structure, mp_ids=["mp-82"],
                            benchmark_mp_ids=["mp-82"], ml_path=gapfit.output["mlip_path"], ml_model="GAP",
                            dft_references=[dft_doc], add_dft_phonon_struct=False, fit_input=None, symprec=1e-1,
                            phonon_displacement_maker=None, supercell_settings={"min_length": 8},
                            phonon_bulk_relax_maker=None, phonon_static_energy_maker=None,
                            atomwise_regularization_parameter=0.01, )
    jobs.append(bm)

    response = run_locally(Flow(jobs), store=memory_jobstore)
    output = response[bm.output.uuid][1].output[0].resolve(store=memory_jobstore)
    assert output["benchmark_phonon_rmse"] == approx(1.0, abs=0.8)
    assert output["dft_imaginary_modes"] is False
    assert output["ml_imaginary_modes"] is False


def test_get_iso_atom(vasp_test_dir, mock_vasp, clean_dir, memory_jobstore):
    from autoplex.data.phonons.flows import IsoAtomStaticMaker
    structure_list = [
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Si", "Si"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Mo", "C", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.55, 0.55, 0.55]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Mo", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Na", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.55, 0.55, 0.55]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Li"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Cl"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
    ]

    ref_paths = {
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "C-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Mo-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "K-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Si-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Na-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "Li-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Cl-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "C-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Mo-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "K-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Si-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Na-stat_iso_atom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    isolated_atom = get_iso_atom(structure_list, IsoAtomStaticMaker())

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(isolated_atom, create_folders=True, ensure_success=True)

    assert (
            "[Element Li, Element C, Element Mo, Element Na, Element Si, Element Cl, Element K]"
            == f"{responses[isolated_atom.output.uuid][2].output['species']}"
    )
    assert (
            "Li"
            and "C"
            and "Mo"
            and "Na"
            and "Si"
            and "Cl"
            and "K" in f"{responses[isolated_atom.output.uuid][2].output['species']}"
    )


def test_dft_task_doc(
        vasp_test_dir,
        mock_vasp,
        test_dir,
        memory_jobstore,
        relax_maker,
        ref_paths,
        fake_run_vasp_kwargs,
        clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    dft_phonon_workflow = dft_phonopy_gen_data(structure=structure, mp_id="test", displacements=[0.01], symprec=0.1,
                                               phonon_displacement_maker=TightDFTStaticMaker(),
                                               phonon_bulk_relax_maker=relax_maker,
                                               phonon_static_energy_maker=StaticMaker(),
                                               supercell_settings={"min_length": 10, "min_atoms": 20})

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        dft_phonon_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    # check for DFT phonon doc
    assert isinstance(
        dft_phonon_workflow.output.resolve(store=memory_jobstore)["phonon_data"]["001"],
        PhononBSDOSDoc,
    )


def test_dft_phonopy_gen_data_manual_supercell_matrix(
        vasp_test_dir,
        mock_vasp,
        test_dir,
        memory_jobstore,
        relax_maker,
        ref_paths_check_sc_mat,
        fake_run_vasp_kwargs,
        clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    supercell_settings = {
        "test": {
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        },
        "min_length": 1,
        "max_length": 25,
        "min_atoms": 2
    }

    dft_phonon_workflow = dft_phonopy_gen_data(structure=structure, mp_id="test", displacements=[0.01], symprec=0.1,
                                               phonon_displacement_maker=TightDFTStaticMaker(),
                                               phonon_bulk_relax_maker=relax_maker,
                                               phonon_static_energy_maker=StaticMaker(),
                                               supercell_settings=supercell_settings)

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths_check_sc_mat, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        dft_phonon_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    result_structure = dft_phonon_workflow.output.resolve(store=memory_jobstore)['phonon_data']['001'].structure
    assert result_structure.lattice.abc == pytest.approx(structure.lattice.abc, rel=0.005)


def test_dft_random_gen_data_manual_supercell_matrix(
        vasp_test_dir,
        mock_vasp,
        test_dir,
        memory_jobstore,
        relax_maker,
        ref_paths_check_sc_mat,
        fake_run_vasp_kwargs,
        clean_dir
):
    from pathlib import Path
    from atomate2.utils.path import strip_hostname
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    supercell_settings = {
        "test": {
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        },
        "min_length": 1,
        "max_length": 25,
        "min_atoms": 2
    }

    dft_rattled_workflow = dft_random_gen_data(structure=structure, mp_id="test",
                                               volume_custom_scale_factors=[0.95, 1.0, 1.05],
                                               displacement_maker=TightDFTStaticMaker(),
                                               rattled_bulk_relax_maker=relax_maker,
                                               supercell_settings=supercell_settings)

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths_check_sc_mat, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        dft_rattled_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    for path in dft_rattled_workflow.output.resolve(store=memory_jobstore)['rand_struc_dir'][0]:
        result_structure = Structure.from_file(Path(strip_hostname(path)).joinpath("POSCAR.gz"))
        assert result_structure.lattice.abc == pytest.approx(structure.lattice.abc, rel=0.05)
        # high rel error because of volume scaling
