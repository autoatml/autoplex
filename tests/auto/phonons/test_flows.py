import os
import pytest
from monty.serialization import loadfn
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core.structure import Structure
from autoplex.auto.phonons.flows import (
    CompleteDFTvsMLBenchmarkWorkflow,
    CompleteDFTvsMLBenchmarkWorkflowMPSettings,
    IterativeCompleteDFTvsMLBenchmarkWorkflow)
from jobflow import run_locally

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture(scope="class")
def ref_paths():
    return {
        "dft tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/12_test": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/12_test": "dft_ml_data_generation/rand_static_2/",
        "dft rattle static 3/12_test": "dft_ml_data_generation/rand_static_3/",
        "dft rattle static 4/12_test": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 5/12_test": "dft_ml_data_generation/rand_static_5/",
        "dft rattle static 6/12_test": "dft_ml_data_generation/rand_static_6/",
        "dft rattle static 7/12_test": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 8/12_test": "dft_ml_data_generation/rand_static_8/",
        "dft rattle static 9/12_test": "dft_ml_data_generation/rand_static_9/",
        "dft rattle static 10/12_test": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 11/12_test": "dft_ml_data_generation/rand_static_11/",
        "dft rattle static 12/12_test": "dft_ml_data_generation/rand_static_12/",
        "dft tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/12_mp-22905": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/12_mp-22905": "dft_ml_data_generation/rand_static_2/",
        "dft rattle static 3/12_mp-22905": "dft_ml_data_generation/rand_static_3/",
        "dft rattle static 4/12_mp-22905": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 5/12_mp-22905": "dft_ml_data_generation/rand_static_5/",
        "dft rattle static 6/12_mp-22905": "dft_ml_data_generation/rand_static_6/",
        "dft rattle static 7/12_mp-22905": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 8/12_mp-22905": "dft_ml_data_generation/rand_static_8/",
        "dft rattle static 9/12_mp-22905": "dft_ml_data_generation/rand_static_9/",
        "dft rattle static 10/12_mp-22905": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 11/12_mp-22905": "dft_ml_data_generation/rand_static_11/",
        "dft rattle static 12/12_mp-22905": "dft_ml_data_generation/rand_static_12/",
    }


@pytest.fixture(scope="class")
def ref_paths_mpid():
    return {
        "dft tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/12_mp-22905": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/12_mp-22905": "dft_ml_data_generation/rand_static_2/",
        "dft rattle static 3/12_mp-22905": "dft_ml_data_generation/rand_static_3/",
        "dft rattle static 4/12_mp-22905": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 5/12_mp-22905": "dft_ml_data_generation/rand_static_5/",
        "dft rattle static 6/12_mp-22905": "dft_ml_data_generation/rand_static_6/",
        "dft rattle static 7/12_mp-22905": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 8/12_mp-22905": "dft_ml_data_generation/rand_static_8/",
        "dft rattle static 9/12_mp-22905": "dft_ml_data_generation/rand_static_9/",
        "dft rattle static 10/12_mp-22905": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 11/12_mp-22905": "dft_ml_data_generation/rand_static_11/",
        "dft rattle static 12/12_mp-22905": "dft_ml_data_generation/rand_static_12/",
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs():
    return {
        "dft tight relax_test": {"incar_settings": ["NSW"]},
        "dft tight relax 1_test": {"incar_settings": ["NSW"]},
        "dft tight relax 2_test": {"incar_settings": ["NSW"]},
        "dft phonon static 1/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 5/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 6/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 7/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 8/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 9/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 10/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 11/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 12/12_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft tight relax_mp-22905": {"incar_settings": ["NSW"]},
        "dft tight relax 1_mp-22905": {"incar_settings": ["NSW"]},
        "dft tight relax 2_mp-22905": {"incar_settings": ["NSW"]},
        "dft phonon static 1/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 5/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 6/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 7/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 8/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 9/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 10/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 11/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 12/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs_mpid():
    return {
        "dft tight relax_mp-22905": {"incar_settings": ["NSW"]},
        "dft tight relax 1_mp-22905": {"incar_settings": ["NSW"]},
        "dft tight relax 2_mp-22905": {"incar_settings": ["NSW"]},
        "dft phonon static 1/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 5/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 6/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 7/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 8/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 9/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 10/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 11/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 12/12_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
    }


@pytest.fixture(scope="class")
def ref_paths4():
    return {
        "dft tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/4_test": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/4_test": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 3/4_test": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 4/4_test": "dft_ml_data_generation/rand_static_10/",
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs5():
    return {
        "dft tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs4():
    return {
        "dft tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
    }


@pytest.fixture(scope="class")
def ref_paths4_mpid():
    return {
        "dft tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax_test_0": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax_test_1": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax_test2": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax_test3": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test_0": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test_1": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 2_test_0": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 2_test_1": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 1_test2": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_test3": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_test2": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 2_test3": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 1_mp-22905_0": "dft_ml_data_generation/tight_relax_1/",
        "dft tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "dft tight relax 2_mp-22905_0": "dft_ml_data_generation/tight_relax_2/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Cl-stat_iso_atom_0": "Cl_iso_atoms/Cl-statisoatom/",
        "Cl-stat_iso_atom_1": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "Li-stat_iso_atom_0": "Li_iso_atoms/Li-statisoatom/",
        "Li-stat_iso_atom_1": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test_0": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test_1": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test_0": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test_1": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 1/2_test2": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test2_0": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test3": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test3_0": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test2": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test2_0": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test3": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test3_0": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_mp-22905_0": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_mp-22905_0": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/4_test": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/4_test_0": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/1_test_1": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/4_mp-22905": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/4_mp-22905_0": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/4_test": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_test_0": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_test_1": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_mp-22905": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_mp-22905_0": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_mp-22905_1": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 3/4_test": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_test_0": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_test_1": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_mp-22905": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_mp-22905_0": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_mp-22905_1": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 4/4_test": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_test_0": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_test_1": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_mp-22905": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_mp-22905_0": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_mp-22905_1": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 1/4_test2": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/4_test3": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/4_test2": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_test3": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 3/4_test2": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_test3": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 4/4_test2": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_test3": "dft_ml_data_generation/rand_static_10/",
    }

@pytest.fixture(scope="class")
def ref_paths4_mpid_new():
    return {
        "dft tight relax_test": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax_test2": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax_test3": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax_mp-22905": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 1_test": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 2_test": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 1_test2": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 2_test2": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 1_test3": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 2_test3": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 1_mp-22905": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 2_mp-22905": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 2/2_test": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft phonon static 1/2_test2": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 1/2_test3": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 2/2_test2": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft phonon static 2/2_test3": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft rattle static 1/4_test": "dft_ml_data_generation/strict_test/rand_static_1/",
        "dft rattle static 2/4_test": "dft_ml_data_generation/strict_test/rand_static_2/",
        "dft rattle static 3/4_test": "dft_ml_data_generation/strict_test/rand_static_3/",
        "dft rattle static 4/4_test": "dft_ml_data_generation/strict_test/rand_static_4/",
        "dft rattle static 1/4_test2": "dft_ml_data_generation/strict_test/rand_static_5/",
        "dft rattle static 2/4_test2": "dft_ml_data_generation/strict_test/rand_static_6/",
        "dft rattle static 3/4_test2": "dft_ml_data_generation/strict_test/rand_static_7/",
        "dft rattle static 4/4_test2": "dft_ml_data_generation/strict_test/rand_static_8/",
    }

@pytest.fixture(scope="class")
def ref_paths4_mpid_new2():
    return {
        "dft tight relax_test_0": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax_test_1": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax_test_2": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 1_test_0": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 1_test_1": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 1_test_2": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 1_test_3": "dft_ml_data_generation/strict_test/tight_relax_1_test/",
        "dft tight relax 2_test_0": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 2_test_1": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 2_test_2": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "dft tight relax 2_test_3": "dft_ml_data_generation/strict_test/tight_relax_2_test/",
        "Cl-stat_iso_atom_0": "Cl_iso_atoms/Cl-statisoatom/",
        "Cl-stat_iso_atom_1": "Cl_iso_atoms/Cl-statisoatom/",
        "Cl-stat_iso_atom_2": "Cl_iso_atoms/Cl-statisoatom/",
        "Cl-stat_iso_atom_3": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom_0": "Li_iso_atoms/Li-statisoatom/",
        "Li-stat_iso_atom_1": "Li_iso_atoms/Li-statisoatom/",
        "Li-stat_iso_atom_2": "Li_iso_atoms/Li-statisoatom/",
        "Li-stat_iso_atom_3": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test_0": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 1/2_test_1": "dft_ml_data_generation/strict_test/phonon_static_1/",
        "dft phonon static 2/2_test_0": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft phonon static 2/2_test_1": "dft_ml_data_generation/strict_test/phonon_static_2/",
        "dft rattle static 1/4_test_0": "dft_ml_data_generation/strict_test/rand_static_1/",
        "dft rattle static 1/1_test_1": "dft_ml_data_generation/strict_test/rand_static_5/",
        "dft rattle static 1/1_test_2": "dft_ml_data_generation/strict_test/rand_static_6/",
        "dft rattle static 1/1_test_3": "dft_ml_data_generation/strict_test/rand_static_7/",
        "dft rattle static 2/4_test_0": "dft_ml_data_generation/strict_test/rand_static_2/",
        "dft rattle static 3/4_test_0": "dft_ml_data_generation/strict_test/rand_static_3/",
        "dft rattle static 4/4_test_0": "dft_ml_data_generation/strict_test/rand_static_4/",
        }


@pytest.fixture(scope="class")
def ref_paths5_mpid():
    return {
        "dft tight relax 1_test": "MP_finetuning/tight_relax_1/",
        "dft tight relax 2_test": "MP_finetuning/tight_relax_2/",
        "Sn-stat_iso_atom": "MP_finetuning/Sn-stat_iso_atom/",
        "dft phonon static 1/1_test": "MP_finetuning/phonon_static_1/",
        "dft rattle static 1/3_test": "MP_finetuning/rand_static_1/",
        "dft rattle static 2/3_test": "MP_finetuning/rand_static_2/",
        "dft rattle static 3/3_test": "MP_finetuning/rand_static_3/",
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs5_mpid():
    return {
    }


@pytest.fixture(scope="class")
def fake_run_vasp_kwargs4_mpid():
    return {
        "dft tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax_test_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax_test_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test2_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test2_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test3_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_test3_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test2_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test2_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test3_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_test3_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 1_mp-22905_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft tight relax 2_mp-22905_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test_1": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test_2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_mp-22905_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_mp-22905_0": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft rattle static 1/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/4_test_0": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/1_test_1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/4_test_1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },

        "dft rattle static 2/4_test_0": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test_1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },

        "dft rattle static 3/4_test_0": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test_1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_test": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },

        "dft rattle static 4/4_test_0": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },

        "dft rattle static 4/4_test_1": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/4_test2": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test2": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test2": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_test2": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/4_test3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_test3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_test3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_test3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 1/4_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 2/4_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 3/4_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
        "dft rattle static 4/4_mp-22905": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "potcar"],
        },
    }

@pytest.fixture(scope="class")
def fake_run_vasp_kwargs4_mpid_new():
    return {}

@pytest.fixture(scope="class")
def fake_run_vasp_kwargs4_mpid_new2():
    return {}


def test_iterative_complete_dft_vs_ml_benchmark_workflow_gap(vasp_test_dir, mock_vasp, test_dir, memory_jobstore,
                                                             ref_paths4_mpid_new2, fake_run_vasp_kwargs4_mpid_new2,
                                                             clean_dir):
    from ase.io import read
    from pathlib import Path
    # first test with just one iteration (more tests need to be added)
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = IterativeCompleteDFTvsMLBenchmarkWorkflow(
        rms_max=0.2,
        max_iterations=3,
        complete_dft_vs_ml_benchmark_workflow_0=CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2, displacements=[0.01],
                                                                                 split_ratio=0.33,
                                                                                 volume_custom_scale_factors=[0.975,
                                                                                                              1.0,
                                                                                                              1.025,
                                                                                                              1.05],
                                                                                 supercell_settings={"min_length": 8,
                                                                                                     "min_atoms": 20},
                                                                                 apply_data_preprocessing=True),
        complete_dft_vs_ml_benchmark_workflow_1=CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2, displacements=[0.01],
                                                                                 split_ratio=0.33,
                                                                                 volume_custom_scale_factors=[0.975],
                                                                                 supercell_settings={"min_length": 8,
                                                                                                     "min_atoms": 20},
                                                                                 apply_data_preprocessing=True,
                                                                                 add_dft_phonon_struct=False,
                                                                                 num_processes_fit=4,
                                                                                 ),

    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["test"],
        benchmark_structures=[structure],
        rattle_seed=42,
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid_new2, fake_run_vasp_kwargs4_mpid_new2)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    vasp_xyz = read(Path(complete_workflow.output.resolve(memory_jobstore)["pre_database_dir"]) / "vasp_ref.extxyz",":")
    assert len(vasp_xyz) == 10
    assert isinstance(complete_workflow.output.resolve(memory_jobstore)["dft_references"], list)


def test_iterative_complete_dft_vs_ml_benchmark_workflow_gap_add_phonon_false(vasp_test_dir, mock_vasp, test_dir, memory_jobstore,  ref_paths4_mpid_new2, fake_run_vasp_kwargs4_mpid_new2, clean_dir):
    # first test with just one iteration (more tests need to be added)
    from ase.io import read
    from pathlib import Path
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = IterativeCompleteDFTvsMLBenchmarkWorkflow(
        rms_max=0.2,
        max_iterations=3,
        complete_dft_vs_ml_benchmark_workflow_0=CompleteDFTvsMLBenchmarkWorkflow( symprec=1e-2, displacements=[0.01],
                                                                                  split_ratio=0.33,
                                                                                  add_dft_phonon_struct=False,
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        supercell_settings={"min_length": 8, "min_atoms": 20},
        apply_data_preprocessing=True),
        complete_dft_vs_ml_benchmark_workflow_1=CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2, displacements=[0.01],
                                                                                 split_ratio=0.33,
                                                                                 volume_custom_scale_factors=[0.975],
                                                                                 supercell_settings={"min_length": 8,
                                                                                                     "min_atoms": 20},
                                                                                 apply_data_preprocessing=True,
                                                                                 add_dft_phonon_struct=False,
                                                                                 num_processes_fit=4,
                                                                                 ),


    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["test"],
        benchmark_structures=[structure],
        rattle_seed=42,
    )
    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid_new2, fake_run_vasp_kwargs4_mpid_new2)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    vasp_xyz = read(Path(complete_workflow.output.resolve(memory_jobstore)["pre_database_dir"])/"vasp_ref.extxyz",":")
    assert len(vasp_xyz) == 8


def test_complete_dft_vs_ml_benchmark_workflow_gap(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = CompleteDFTvsMLBenchmarkWorkflow(
        symprec=1e-2, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        supercell_settings={"min_length": 8, "min_atoms": 20},
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        2.502641337594289, abs=1.5  # it's kinda fluctuating because of the little data
    )

    # check if soap_default_dict is correctly constructed from
    # n_sparse and delta values in mlip_phonon_default json file
    expected_soap_dict = "atom-wise f=0.1: n_sparse = 6000, SOAP delta = 1.0"
    results_files = glob.glob('job*/results_LiCl.txt')

    for file_path in results_files:
        with open(file_path, 'r') as file:
            results_file = file.read().strip()
            assert expected_soap_dict in results_file, f"Expected soap_dict not found in {file_path}"


def test_complete_dft_vs_gap_benchmark_workflow_database(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = CompleteDFTvsMLBenchmarkWorkflow(
        symprec=1e-2, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        supercell_settings={"min_length": 8, "min_atoms": 20},
        apply_data_preprocessing=True,
        run_fits_on_different_cluster=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow.jobs[5].name == "complete_benchmark_mp-22905"

    assert responses[complete_workflow.jobs[-1].output.uuid][1].output["metrics"][0][0]["benchmark_phonon_rmse"] == pytest.approx(
        2.502641337594289, abs=1.5  # it's kinda fluctuating because of the little data
    )

    # check if soap_default_dict is correctly constructed from
    # n_sparse and delta values in mlip_phonon_default json file
    expected_soap_dict = "atom-wise f=0.1: n_sparse = 6000, SOAP delta = 1.0"
    results_files = glob.glob('job*/results_LiCl.txt')

    for file_path in results_files:
        with open(file_path, 'r') as file:
            results_file = file.read().strip()
            assert expected_soap_dict in results_file, f"Expected soap_dict not found in {file_path}"


def test_complete_dft_vs_ml_benchmark_workflow_m3gnet(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_m3gnet = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["M3GNET"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        benchmark_structures=[structure],
        fit_kwargs_list=[{
            "cutoff": 3.0,
            "threebody_cutoff": 2.0,
            "batch_size": 1,
            "max_epochs": 3,
            "include_stresses": True,
            "dim_node_embedding": 8,
            "dim_edge_embedding": 8,
            "units": 8,
            "max_l": 4,
            "max_n": 4,
            "device": "cpu",
            "test_equal_to_val": True,
        }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_m3gnet,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert complete_workflow_m3gnet.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_m3gnet.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        5.2622804443539355, abs=3.0  # bad fit data, fluctuates between 4 and 7
    )


def test_complete_dft_vs_ml_benchmark_workflow_m3gnet_finetuning(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_m3gnet = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["M3GNET"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        benchmark_structures=[structure],
        fit_kwargs_list=[{
            "batch_size": 1,
            "max_epochs": 1,
            "include_stresses": True,
            "device": "cpu",
            "test_equal_to_val": True,
            "foundation_model": "M3GNet-MP-2021.2.8-DIRECT-PES",
            "use_foundation_model_element_refs": True,
        }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_m3gnet,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_m3gnet.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_m3gnet.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        4.6, abs=0.5,
    )

def test_complete_dft_vs_ml_benchmark_workflow_nep(
        vasp_test_dir, mock_vasp, mock_nep, test_dir, memory_jobstore,
        ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_nep = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["NEP"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        benchmark_structures=[structure],
        fit_kwargs_list=[{
            "generation": 1000,
            "batch": 1,
            }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # mock nep
    ref_paths_nep = {"machine_learning_fit": "LiCl_complete_dft_vs_ml_benchmark_workflow"}
    fake_run_nep_kwargs = {
        "machine_learning_fit": {"nep_settings": ["generation", "batch", "type_weight"], "check_nep_inputs": True}}
    mock_nep(ref_paths_nep, fake_run_nep_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_nep,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert complete_workflow_nep.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_nep.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        3.8951576702856716
    )


def test_complete_dft_vs_ml_benchmark_workflow_mace(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_mace = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["MACE"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}},
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        fit_kwargs_list=[{
            "model": "MACE",
            "config_type_weights": '{"Default":1.0}',
            "hidden_irreps": "32x0e + 32x1o",
            "r_max": 3.0,
            "batch_size": 5,
            "max_num_epochs": 10,
            "start_swa": 5,
            "ema_decay": 0.99,
            "correlation": 3,
            "loss": "huber",
            "default_dtype": "float32",
            "device": "cpu",
        }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_mace,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_mace.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        5.391879137001022, abs=3.0
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_mace_finetuning(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_mace = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["MACE"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}},
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        fit_kwargs_list=[{
            "model": "MACE",
            "name": "MACE_final",
            "foundation_model": "small",
            "multiheads_finetuning": False,
            "r_max": 6,
            "loss": "huber",
            "energy_weight": 1000.0,
            "forces_weight": 1000.0,
            "stress_weight": 1.0,
            "compute_stress": True,
            "E0s": "average",
            "scaling": "rms_forces_scaling",
            "batch_size": 1,
            "max_num_epochs": 1,
            "ema": True,
            "ema_decay": 0.99,
            "amsgrad": True,
            "default_dtype": "float64",
            "restart_latest": True,
            "lr": 0.0001,
            "patience": 20,
            "device": "cpu",
            "save_cpu": True,
            "seed": 3,
        }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_mace,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_mace.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.45, abs=0.4
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_mace_finetuning_mp_settings(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths5_mpid, fake_run_vasp_kwargs5_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "MP_finetuning" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_mace = CompleteDFTvsMLBenchmarkWorkflowMPSettings(
        ml_models=["MACE"],
        volume_custom_scale_factors=[0.95, 1.00, 1.05], rattle_type=0, distort_type=0,
        symprec=1e-3, supercell_settings={"min_length": 6, "max_length": 10, "min_atoms": 10, "max_atoms": 300, },
        displacements=[0.01],
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}},
        add_dft_rattled_struct=True,
        apply_data_preprocessing=True,
        split_ratio=0.3,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["test"],
        benchmark_structures=[structure],
        fit_kwargs_list=[{
            "model": "MACE",
            "name": "MACE_final",
            "foundation_model": "small",
            "multiheads_finetuning": False,
            "r_max": 6,
            "loss": "huber",
            "energy_weight": 1000.0,
            "forces_weight": 1000.0,
            "stress_weight": 1.0,
            "compute_stress": True,
            "E0s": "average",
            "scaling": "rms_forces_scaling",
            "batch_size": 1,
            "max_num_epochs": 10,
            "ema": True,
            "ema_decay": 0.99,
            "amsgrad": True,
            "default_dtype": "float64",
            "restart_latest": True,
            "lr": 0.0001,
            "patience": 20,
            "device": "cpu",
            "save_cpu": True,
            "seed": 3,
        }]
    )
    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths5_mpid, fake_run_vasp_kwargs5_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_mace,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_mace.jobs[5].name == "complete_benchmark_test"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.45, abs=0.4
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_nequip(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_nequip = CompleteDFTvsMLBenchmarkWorkflow(
        ml_models=["NEQUIP"],
        symprec=1e-2, supercell_settings={"min_length": 8, "min_atoms": 20}, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}},
        apply_data_preprocessing=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        fit_kwargs_list=[{
            "r_max": 4.0,
            "num_layers": 4,
            "l_max": 2,
            "num_features": 32,
            "num_basis": 8,
            "invariant_layers": 2,
            "invariant_neurons": 64,
            "batch_size": 1,
            "learning_rate": 0.005,
            "max_epochs": 1,
            "device": "cpu",
        }]
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_nequip,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_nequip.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_nequip.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        5.633069137001022, abs=3.0
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_two_mpids(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid_new, fake_run_vasp_kwargs4_mpid_new, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_two_mpid = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                                  supercell_settings={"min_length": 8, "min_atoms": 20},
                                                                  displacements=[0.01],
                                                                  volume_custom_scale_factors=[0.975, 1.0, 1.025,
                                                                                               1.05],
                                                                  apply_data_preprocessing=True,
                                                                  ).make(
        structure_list=[structure, structure],
        mp_ids=["test", "test2"],
        benchmark_mp_ids=["mp-22905", "test3"],
        benchmark_structures=[structure, structure],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid_new, fake_run_vasp_kwargs4_mpid_new)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_two_mpid,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_two_mpid.jobs[8].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_two_mpid.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.7126017685370398, abs=0.5
    )


def test_complete_dft_vs_ml_benchmark_workflow_with_hploop(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_hploop = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                                supercell_settings={"min_length": 8, "min_atoms": 20},
                                                                displacements=[0.01],
                                                                volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
                                                                hyper_para_loop=True,
                                                                atomwise_regularization_list=[0.01],
                                                                n_sparse_list=[3000, 5000],
                                                                soap_delta_list=[1.0],
                                                                apply_data_preprocessing=True,
                                                                ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_hploop,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_hploop.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_hploop.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        2.002641337594289, abs=1.0  # it's kinda fluctuating because of the little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_with_sigma_regularization_hploop(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_sigma_hploop = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2, supercell_settings={"min_length": 8,
                                                                                                        "min_atoms": 20},
                                                                      displacements=[0.01],
                                                                      volume_custom_scale_factors=[0.975, 1.0, 1.025,
                                                                                                   1.05],
                                                                      hyper_para_loop=True,
                                                                      atomwise_regularization_list=[0.01],
                                                                      n_sparse_list=[3000, 5000],
                                                                      soap_delta_list=[1.0],
                                                                      apply_data_preprocessing=True,
                                                                      regularization=True,
                                                                      ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_sigma_hploop,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_sigma_hploop.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sigma_hploop.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        1.511743561686686, abs=1.0  # it's kinda fluctuating because of the little data
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/without_regularization/train.extxyz"))
    assert reg_specific_file_exists


def test_complete_dft_vs_ml_benchmark_workflow_with_sigma_regularization(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_sigma = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                               supercell_settings={"min_length": 8, "min_atoms": 20},
                                                               displacements=[0.01],
                                                               volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
                                                               summary_filename_prefix="test_results_",
                                                               apply_data_preprocessing=True,
                                                               regularization=True,
                                                               ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        fit_kwargs_list=[{"soap": {"delta": 3.0, "l_max": 12, "n_max": 10, "n_sparse": 8000, "f0": 0.0}}],
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_sigma,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_sigma.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sigma.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.6205293987404107, abs=0.3
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/without_regularization/train.extxyz"))
    assert reg_specific_file_exists

    # check if soap_default_dict is correctly constructed from n_sparse and delta values in user fit parameter input
    expected_soap_dict = "atom-wise f=0.1: n_sparse = 8000, SOAP delta = 3.0"

    results_files = glob.glob('job*/test_results_LiCl.txt')
    for file_path in results_files:
        with open(file_path, 'r') as file:
            results_file = file.read().strip()
            assert expected_soap_dict in results_file in results_file, f"Expected soap_dict not found in {file_path}"


def test_complete_dft_vs_ml_benchmark_workflow_separated(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_sep = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                             run_fits_on_different_cluster=True,
                                                             supercell_settings={"min_length": 8, "min_atoms": 20},
                                                             displacements=[0.01],
                                                             volume_custom_scale_factors=[0.975, 1.0, 1.025,
                                                                                          1.05],
                                                             apply_data_preprocessing=True,
                                                             separated=True,
                                                             ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",

    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_sep,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_sep.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sep.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.8709764794814768, abs=0.5
    )


def test_complete_dft_vs_ml_benchmark_workflow_separated_sigma_reg_hploop_three_mpids(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_sep_3 = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                               supercell_settings={"min_length": 8, "min_atoms": 20},
                                                               displacements=[0.01],
                                                               volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
                                                               hyper_para_loop=True,
                                                               atomwise_regularization_list=[0.01],
                                                               n_sparse_list=[3000, 5000],
                                                               soap_delta_list=[1.0],
                                                               apply_data_preprocessing=True,
                                                               regularization=True,
                                                               separated=True,
                                                               ).make(
        structure_list=[structure, structure, structure],
        mp_ids=["test", "test2", "test3"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",

    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_sep_3,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert responses[complete_workflow_sep_3.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.8709764794814768, abs=0.5
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/without_regularization/train.extxyz"))
    assert reg_specific_file_exists


def test_complete_dft_vs_ml_benchmark_workflow_separated_sigma_reg_hploop(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_sep = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                             supercell_settings={"min_length": 8, "min_atoms": 20},
                                                             displacements=[0.01],
                                                             volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
                                                             hyper_para_loop=True, atomwise_regularization_list=[0.01],
                                                             n_sparse_list=[3000, 5000], soap_delta_list=[1.0],
                                                             apply_data_preprocessing=True,
                                                             regularization=True,
                                                             separated=True,
                                                             ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",

    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_sep,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_sep.jobs[5].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sep.jobs[-1].output.uuid][1].output["metrics"][0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.8709764794814768, abs=0.5
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/without_regularization/train.extxyz"))
    assert reg_specific_file_exists


class TestCompleteDFTvsMLBenchmarkWorkflow:
    def test_add_data_to_dataset_workflow(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
            fake_run_vasp_kwargs,
            ref_paths,
    ):
        import pytest

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)

        add_data_workflow = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],

            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=None,
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )

        # automatically use fake VASP and write POTCAR.spec during the test
        mock_vasp(ref_paths, fake_run_vasp_kwargs)

        # run the flow or job and ensure that it finished running successfully
        responses = run_locally(
            add_data_workflow,
            create_folders=True,
            ensure_success=True,
            store=memory_jobstore,
        )

        assert responses[add_data_workflow.jobs[-1].output.uuid][
                   1
               ].output["metrics"][0][0][
                   "benchmark_phonon_rmse"] == pytest.approx(0.4841808019705598, abs=0.5)

    def test_add_data_workflow_with_dft_reference(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
            fake_run_vasp_kwargs,
            ref_paths,
    ):

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)
        dft_data = loadfn(test_dir / "benchmark" / "PhononBSDOSDoc_LiCl.json")
        dft_reference: PhononBSDOSDoc = dft_data["output"]

        add_data_workflow_with_dft_reference = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            add_dft_phonon_struct=False,
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=[dft_reference],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )

        # automatically use fake VASP and write POTCAR.spec during the test
        mock_vasp(ref_paths, fake_run_vasp_kwargs)

        _ = run_locally(
            add_data_workflow_with_dft_reference,
            create_folders=True,
            ensure_success=True,
            store=memory_jobstore,
        )

        for job, uuid in add_data_workflow_with_dft_reference.iterflow():
            assert job.name != "dft_phonopy_gen_data_mp-22905"

        for job, uuid in add_data_workflow_with_dft_reference.iterflow():
            assert job.name != "dft tight relax 1_mp-22905"

    def test_add_data_workflow_add_phonon_false(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
            fake_run_vasp_kwargs,
            ref_paths,
    ):

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)

        add_data_workflow_add_phonon_false = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            add_dft_phonon_struct=False,
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=None,
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )

        for job, uuid in add_data_workflow_add_phonon_false.iterflow():
            assert job.name != "dft_phonopy_gen_data_mp-22905"

    def test_add_data_workflow_add_random_false(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
            fake_run_vasp_kwargs,
            ref_paths,
    ):

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)

        add_data_workflow_add_random_false = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            add_dft_rattled_struct=False,
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=None,
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )

        for job, uuid in add_data_workflow_add_random_false.iterflow():
            assert job.name != "dft_random_gen_data_mp-22905"

    def test_add_data_workflow_with_same_mpid(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
            fake_run_vasp_kwargs_mpid,
            ref_paths_mpid,
    ):

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)

        add_data_workflow_with_same_mpid = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["mp-22905"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=None,
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )

        for job, uuid in add_data_workflow_with_same_mpid.iterflow():
            assert job.name != "dft tight relax 1_mp-22905"

    def test_workflow_with_different_makers(
            self,
            vasp_test_dir,
            mock_vasp,
            test_dir,
            memory_jobstore,
            clean_dir,
    ):
        from autoplex.data.phonons.flows import IsoAtomStaticMaker
        from autoplex.data.phonons.flows import TightDFTStaticMaker
        from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
        from atomate2.vasp.sets.core import StaticSetGenerator

        ref_paths = {
            "dft tight relax_mp-22905": "dft_ml_data_generation/tight_relax_ISPIN2/",
            # it's not a DoubleRelaxMaker in the test
            "dft static_mp-22905": "dft_ml_data_generation/tight_relax_ISPIN2/",
            "Cl-stat_iso_atom": "Cl_iso_atoms_ISMEAR1/Cl-statisoatom/",
            "Li-stat_iso_atom": "Li_iso_atoms_ISMEAR1/Li-statisoatom/",
            "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1_ISPIN2/",
            "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2_ISPIN2/",
            "dft rattle static 1/12_mp-22905": "dft_ml_data_generation/rand_static_1_ISPIN2/",
            "dft rattle static 2/12_mp-22905": "dft_ml_data_generation/rand_static_2_ISPIN2/",
            "dft rattle static 3/12_mp-22905": "dft_ml_data_generation/rand_static_3_ISPIN2/",
            "dft rattle static 4/12_mp-22905": "dft_ml_data_generation/rand_static_4_ISPIN2/",
            "dft rattle static 5/12_mp-22905": "dft_ml_data_generation/rand_static_5_ISPIN2/",
            "dft rattle static 6/12_mp-22905": "dft_ml_data_generation/rand_static_6_ISPIN2/",
            "dft rattle static 7/12_mp-22905": "dft_ml_data_generation/rand_static_7_ISPIN2/",
            "dft rattle static 8/12_mp-22905": "dft_ml_data_generation/rand_static_8_ISPIN2/",
            "dft rattle static 9/12_mp-22905": "dft_ml_data_generation/rand_static_9_ISPIN2/",
            "dft rattle static 10/12_mp-22905": "dft_ml_data_generation/rand_static_10_ISPIN2/",
            "dft rattle static 11/12_mp-22905": "dft_ml_data_generation/rand_static_11_ISPIN2/",
            "dft rattle static 12/12_mp-22905": "dft_ml_data_generation/rand_static_12_ISPIN2/",
        }

        fake_run_vasp_kwargs = {
            "dft tight relax_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft static_mp-22905": {"incar_settings": ["ISPIN"]},
            "Cl-stat_iso_atom": {"incar_settings": ["ISMEAR"]},
            "Li-stat_iso_atom": {"incar_settings": ["ISMEAR"]},
            "dft phonon static 1/2_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft phonon static 2/2_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 1/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 2/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 3/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 4/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 5/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 6/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 7/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 8/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 9/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 10/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 11/12_mp-22905": {"incar_settings": ["ISPIN"]},
            "dft rattle static 12/12_mp-22905": {"incar_settings": ["ISPIN"]},
        }

        path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
        structure = Structure.from_file(path_to_struct)

        test_iso_atom_static_input_set = StaticSetGenerator(
            user_kpoints_settings={"grid_density": 1},
            user_incar_settings={
                "ISPIN": 2,
                "ISMEAR": 1,
            },
        )
        test_static_iso_atom_maker = IsoAtomStaticMaker(
            name="test_iso_atom_maker",  # this will always be overwritten by "{specie}-stat_iso_atom"
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_displacement_maker = TightDFTStaticMaker(
            name="test_displacement_maker",  # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set
        )
        test_rattled_bulk_relax_maker = TightRelaxMaker(
            name="test_bulk_rattled_maker",  # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_phonon_bulk_relax_maker = TightRelaxMaker(
            name="test_bulk_phonon_maker",  # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_phonon_static_energy_maker = StaticMaker(
            name="test_phonon_static_energy_maker",  # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_different_makers_wf = CompleteDFTvsMLBenchmarkWorkflow(
            n_structures=3,
            symprec=1e-2,
            supercell_settings={"min_length": 8, "min_atoms": 20},
            displacements=[0.01],
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
            displacement_maker=test_displacement_maker,
            phonon_bulk_relax_maker=test_phonon_bulk_relax_maker,
            phonon_static_energy_maker=test_phonon_static_energy_maker,
            rattled_bulk_relax_maker=test_rattled_bulk_relax_maker,
            isolated_atom_maker=test_static_iso_atom_maker,
            apply_data_preprocessing=True,
        ).make(
            structure_list=[structure],
            mp_ids=["mp-22905"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            dft_references=None,
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
            # reduce unit test run time
        )
        mock_vasp(ref_paths, fake_run_vasp_kwargs)

        responses = run_locally(
            test_different_makers_wf,
            create_folders=True,
            ensure_success=True,
            store=memory_jobstore,
        )

        assert "test_phonon_static_energy_maker" not in str(responses)
        assert "test_bulk_phonon_maker" not in str(responses)
        assert "test_bulk_rattled_maker" not in str(responses)


def test_phonon_dft_ml_data_generation_flow(
        vasp_test_dir, mock_vasp, clean_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, test_dir
):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)
    structure_list = [structure]
    mp_ids = ["mp-22905"]

    flow_data_generation = CompleteDFTvsMLBenchmarkWorkflow(
        n_structures=3, supercell_settings={"min_length": 10, "min_atoms": 20}, symprec=1e-2,
        apply_data_preprocessing=True,
    ).make(structure_list=structure_list,
           mp_ids=mp_ids,
           benchmark_structures=structure_list,
           benchmark_mp_ids=mp_ids,
           pre_xyz_files=["vasp_ref.extxyz"],
           pre_database_dir=test_dir / "fitting" / "ref_files",
           fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
           # reduce unit test run time
           )

    flow_data_generation_without_rattled_structures = CompleteDFTvsMLBenchmarkWorkflow(
        n_structures=3, supercell_settings={"min_length": 10, "min_atoms": 20}, symprec=1e-2,
        add_dft_rattled_struct=False,
        apply_data_preprocessing=True,
    ).make(structure_list=structure_list,
           mp_ids=mp_ids,
           benchmark_structures=structure_list,
           benchmark_mp_ids=mp_ids,
           pre_xyz_files=["vasp_ref.extxyz"],
           pre_database_dir=test_dir / "fitting" / "ref_files",
           fit_kwargs_list=[{"general": {"two_body": True, "three_body": False, "soap": False}}]
           # reduce unit test run time
           )
    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        flow_data_generation,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    responses_worattled = run_locally(
        flow_data_generation_without_rattled_structures,
        create_folders=True,
        ensure_success=False,  # only two phonon calcs are not enough for this to pass
        store=memory_jobstore,
    )
    counter = 0
    counter_wor = 0
    for _ in flow_data_generation.iterflow():
        counter += 1
    for _ in flow_data_generation_without_rattled_structures.iterflow():
        counter_wor += 1
    assert counter == 9
    assert counter_wor == 8


# TODO testing cell_factor_sequence


def test_supercell_test_runs(vasp_test_dir, clean_dir, memory_jobstore, test_dir):
    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from autoplex.auto.phonons.flows import DFTSupercellSettingsMaker

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    structure_list = [structure]
    mp_ids = ["mp-22905"]

    autoplex_flow = DFTSupercellSettingsMaker(supercell_settings={"min_length": 10, "min_atoms": 10},
                                              DFT_Maker=ForceFieldStaticMaker(force_field_name="CHGNet")).make(
        structure_list=structure_list, mp_ids=mp_ids, )

    responses_flow = run_locally(autoplex_flow)
    assert responses_flow[autoplex_flow.jobs[-1].output.uuid][1].replace[0].name == "Force field static"
    # seems that the current atomate2 implementation doesn't distinguish in the FF flow names
