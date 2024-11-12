from __future__ import annotations
import os 
os.environ["OMP_NUM_THREADS"] = "1"
import pytest
from monty.serialization import loadfn
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core.structure import Structure
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow, CompleteDFTvsMLBenchmarkWorkflowMPSettings
from jobflow import Response, job
from autoplex.data.rss.jobs import do_rss_single_node, do_rss_multi_node
from autoplex.data.common.jobs import sample_data, collect_dft_data, preprocess_data
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.fitting.common.flows import MLIPFitMaker
from typing import Optional, Dict, Any
from pathlib import Path
from jobflow import run_locally, Flow


@pytest.fixture(scope="class")
def ref_paths():
    return {
        "tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "static_test": "dft_ml_data_generation/static/",
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
        "tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "static_mp-22905": "dft_ml_data_generation/static/",
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
        "tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "static_mp-22905": "dft_ml_data_generation/static/",
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
        "tight relax_test": {"incar_settings": ["NSW"]},
        "tight relax 1_test": {"incar_settings": ["NSW"]},
        "tight relax 2_test": {"incar_settings": ["NSW"]},
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
        "tight relax_mp-22905": {"incar_settings": ["NSW"]},
        "tight relax 1_mp-22905": {"incar_settings": ["NSW"]},
        "tight relax 2_mp-22905": {"incar_settings": ["NSW"]},
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
        "tight relax_mp-22905": {"incar_settings": ["NSW"]},
        "tight relax 1_mp-22905": {"incar_settings": ["NSW"]},
        "tight relax 2_mp-22905": {"incar_settings": ["NSW"]},
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
        "tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "static_test": "dft_ml_data_generation/static/",
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
        "tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
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
        "tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
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
        "tight relax_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax_test2": "dft_ml_data_generation/tight_relax_1/",
        "tight relax_test3": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_test": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_test": "dft_ml_data_generation/tight_relax_2/",
        "tight relax 1_test2": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 1_test3": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_test2": "dft_ml_data_generation/tight_relax_2/",
        "tight relax 2_test3": "dft_ml_data_generation/tight_relax_2/",
        "static_test": "dft_ml_data_generation/static/",
        "static_test2": "dft_ml_data_generation/static/",
        "static_test3": "dft_ml_data_generation/static/",
        "tight relax 1_mp-22905": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2_mp-22905": "dft_ml_data_generation/tight_relax_2/",
        "static_mp-22905": "dft_ml_data_generation/static/",
        "Cl-stat_iso_atom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-stat_iso_atom": "Li_iso_atoms/Li-statisoatom/",
        "dft phonon static 1/2_test": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 1/2_test2": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 1/2_test3": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_test2": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 2/2_test3": "dft_ml_data_generation/phonon_static_2/",
        "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1/",
        "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2/",
        "dft rattle static 1/4_test": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 1/4_mp-22905": "dft_ml_data_generation/rand_static_1/",
        "dft rattle static 2/4_test": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 2/4_mp-22905": "dft_ml_data_generation/rand_static_4/",
        "dft rattle static 3/4_test": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 3/4_mp-22905": "dft_ml_data_generation/rand_static_7/",
        "dft rattle static 4/4_test": "dft_ml_data_generation/rand_static_10/",
        "dft rattle static 4/4_mp-22905": "dft_ml_data_generation/rand_static_10/",
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
def ref_paths5_mpid():
    return {
        "MP GGA relax 1_test": "MP_finetuning/tight_relax_1/",
        "MP GGA relax 2_test": "MP_finetuning/tight_relax_2/",
        "Sn-stat_iso_atom": "MP_finetuning/Sn-stat_iso_atom/",
        "static_test": "MP_finetuning/static_test/",
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
        "tight relax_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_test3": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 1/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft phonon static 2/2_mp-22905": {"incar_settings": ["NSW", "ISMEAR"]},
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




def test_complete_dft_vs_ml_benchmark_workflow_gap(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):
    import glob

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = CompleteDFTvsMLBenchmarkWorkflow(
        symprec=1e-2, displacements=[0.01],
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
        supercell_settings={"min_length": 8, "min_atoms": 20}).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        apply_data_preprocessing=True,
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

    print("RMSE: ", responses[complete_workflow.jobs[-1].output.uuid][1].output[0][0]["benchmark_phonon_rmse"])

    assert complete_workflow.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow.jobs[-1].output.uuid][1].output[0][0]["benchmark_phonon_rmse"] == pytest.approx(
        2.502641337594289, abs=1.5  # it's kinda fluctuating because of the little data
    )

    # check if soap_default_dict is correctly constructed from
    # n_sparse and delta values in mlip_phonon_default json file
    expected_soap_dict = "{'f=0.1': {'n_sparse': 6000, 'delta': 0.5}}"
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
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
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

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_m3gnet,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    assert complete_workflow_m3gnet.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_m3gnet.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        5.2622804443539355, abs=1.0  # bad fit data
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
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}}
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
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

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_mace,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_mace.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output[0][0][
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
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}}
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
        use_defaults_fitting=False,
        model="MACE",
        name="MACE_final",
        foundation_model="small",
        multiheads_finetuning=False,
        r_max=6,
        loss="huber",
        energy_weight=1000.0,
        forces_weight=1000.0,
        stress_weight=1.0,
        compute_stress=True,
        E0s="average",
        scaling="rms_forces_scaling",
        batch_size=1,
        max_num_epochs=1,
        ema=True,
        ema_decay=0.99,
        amsgrad=True,
        default_dtype="float64",
        restart_latest=True,
        lr=0.0001,
        patience=20,
        device="cpu",
        save_cpu=True,
        seed=3,
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

    assert complete_workflow_mace.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.45, abs=0.4
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_mace_finetuning_MP_settings(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths5_mpid, fake_run_vasp_kwargs5_mpid, clean_dir
):

    path_to_struct = vasp_test_dir / "MP_finetuning" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_mace = CompleteDFTvsMLBenchmarkWorkflowMPSettings(
        ml_models=["MACE"],
        volume_custom_scale_factors=[0.95,1.00,1.05], rattle_type=0, distort_type=0,
        symprec=1e-3, supercell_settings={"min_length": 6, "max_length":10, "min_atoms":10, "max_atoms":300,}, displacements=[0.01],
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}},
        add_dft_random_struct=True,
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["test"],
        benchmark_structures=[structure],
        apply_data_preprocessing=True,
        use_defaults_fitting=False,
        split_ratio=0.3,
        model="MACE",
        name="MACE_final",
        foundation_model="small",
        multiheads_finetuning=False,
        r_max=6,
        loss="huber",
        energy_weight=1000.0,
        forces_weight=1000.0,
        stress_weight=1.0,
        compute_stress=True,
        E0s="average",
        scaling="rms_forces_scaling",
        batch_size=1,
        max_num_epochs=10,
        ema=True,
        ema_decay=0.99,
        amsgrad=True,
        default_dtype="float64",
        restart_latest=True,
        lr=0.0001,
        patience=20,
        device="cpu",
        save_cpu=True,
        seed=3,
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

    assert complete_workflow_mace.jobs[4].name == "complete_benchmark_test"
    assert responses[complete_workflow_mace.jobs[-1].output.uuid][1].output[0][0][
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
        benchmark_kwargs={"calculator_kwargs": {"device": "cpu"}}
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
        r_max=4.0,
        num_layers=4,
        l_max=2,
        num_features=32,
        num_basis=8,
        invariant_layers=2,
        invariant_neurons=64,
        batch_size=1,
        learning_rate=0.005,
        max_epochs=1,
        default_dtype="float32",
        device="cpu",
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

    assert complete_workflow_nequip.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_nequip.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        5.633069137001022, abs=3.0
        # result is so bad because hyperparameter quality is reduced to a minimum to save time
        # and too little data
    )


def test_complete_dft_vs_ml_benchmark_workflow_two_mpids(
        vasp_test_dir, mock_vasp, test_dir, memory_jobstore, ref_paths4_mpid, fake_run_vasp_kwargs4_mpid, clean_dir
):

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow_two_mpid = CompleteDFTvsMLBenchmarkWorkflow(symprec=1e-2,
                                                                  supercell_settings={"min_length": 8, "min_atoms": 20},
                                                                  displacements=[0.01],
                                                                  volume_custom_scale_factors=[0.975, 1.0, 1.025,
                                                                                               1.05], ).make(
        structure_list=[structure, structure],
        mp_ids=["test", "test2"],
        benchmark_mp_ids=["mp-22905", "test3"],
        benchmark_structures=[structure, structure],
        apply_data_preprocessing=True,
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths4_mpid, fake_run_vasp_kwargs4_mpid)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        complete_workflow_two_mpid,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert complete_workflow_two_mpid.jobs[6].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_two_mpid.jobs[-1].output.uuid][1].output[0][0][
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
                                                                soap_delta_list=[1.0], ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        apply_data_preprocessing=True,
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

    assert complete_workflow_hploop.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_hploop.jobs[-1].output.uuid][1].output[0][0][
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
                                                                      soap_delta_list=[1.0], ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        apply_data_preprocessing=True,
        **{"regularization": True},
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

    assert complete_workflow_sigma_hploop.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sigma_hploop.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        1.511743561686686, abs=1.0  # it's kinda fluctuating because of the little data
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/train_wo_sigma.extxyz"))
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
                                                               volume_custom_scale_factors=[0.975, 1.0, 1.025,1.05],
                                                               summary_filename_prefix="test_results_",).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        apply_data_preprocessing=True,
        **{"regularization": True,
           "soap": {"delta": 3.0, "l_max": 12, "n_max": 10, "n_sparse": 8000, "f0": 0.0}},
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

    assert complete_workflow_sigma.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sigma.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.6205293987404107, abs=0.3
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/train_wo_sigma.extxyz"))
    assert reg_specific_file_exists

    # check if soap_default_dict is correctly constructed from n_sparse and delta values in user fit parameter input
    expected_soap_dict = "{'f=0.1': {'delta': 3.0, 'n_sparse': 8000}}"

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
                                                             supercell_settings={"min_length": 8, "min_atoms": 20},
                                                             displacements=[0.01],
                                                             volume_custom_scale_factors=[0.975, 1.0, 1.025,
                                                                                          1.05], ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
        **{"separated": True},
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

    assert complete_workflow_sep.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sep.jobs[-1].output.uuid][1].output[0][0][
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
                                                               soap_delta_list=[1.0], ).make(
        structure_list=[structure, structure, structure],
        mp_ids=["test", "test2", "test3"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
        **{"regularization": True, "separated": True},
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

    assert responses[complete_workflow_sep_3.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.8709764794814768, abs=0.5
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/train_wo_sigma.extxyz"))
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
                                                             n_sparse_list=[3000, 5000], soap_delta_list=[1.0], ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids=["mp-22905"],
        benchmark_structures=[structure],
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir=test_dir / "fitting" / "ref_files",
        apply_data_preprocessing=True,
        **{"regularization": True, "separated": True},
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

    assert complete_workflow_sep.jobs[4].name == "complete_benchmark_mp-22905"
    assert responses[complete_workflow_sep.jobs[-1].output.uuid][1].output[0][0][
               "benchmark_phonon_rmse"] == pytest.approx(
        0.8709764794814768, abs=0.5
    )

    # regularization specific test
    reg_specific_file_exists = any(glob.glob("job*/train_wo_sigma.extxyz"))
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
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=None,
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
               ].output[0][0][
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
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=[dft_reference],
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
            assert job.name != "tight relax 1_mp-22905"

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
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=None,
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
            add_dft_random_struct=False,
            volume_custom_scale_factors=[0.975, 0.975, 0.975, 1.0, 1.0, 1.0, 1.025, 1.025, 1.025, 1.05, 1.05, 1.05],
        ).make(
            structure_list=[structure],
            mp_ids=["test"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=None,
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
        ).make(
            structure_list=[structure],
            mp_ids=["mp-22905"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=None,
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
        )

        for job, uuid in add_data_workflow_with_same_mpid.iterflow():
            assert job.name != "tight relax 1_mp-22905"

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
            "tight relax_mp-22905": "dft_ml_data_generation/tight_relax_ISPIN2/",
            # it's not a DoubleRelaxMaker in the test
            "static_mp-22905": "dft_ml_data_generation/tight_relax_ISPIN2/",
            "Cl-stat_iso_atom": "Cl_iso_atoms_ISMEAR1/Cl-statisoatom/",
            "Li-stat_iso_atom": "Li_iso_atoms_ISMEAR1/Li-statisoatom/",
            "dft phonon static 1/2_mp-22905": "dft_ml_data_generation/phonon_static_1_ISPIN2/",
            "dft phonon static 2/2_mp-22905": "dft_ml_data_generation/phonon_static_2_ISPIN2/",
            "dft rattle static 1/12_mp-22905": "dft_ml_data_generation/rand_static_1_ISPIN2/",
            "dft rattle static 2/12_mp-22905": "dft_ml_data_generation/rand_static_1_ISPIN2/",
            "dft rattle static 3/12_mp-22905": "dft_ml_data_generation/rand_static_1_ISPIN2/",
            "dft rattle static 4/12_mp-22905": "dft_ml_data_generation/rand_static_4_ISPIN2/",
            "dft rattle static 5/12_mp-22905": "dft_ml_data_generation/rand_static_4_ISPIN2/",
            "dft rattle static 6/12_mp-22905": "dft_ml_data_generation/rand_static_4_ISPIN2/",
            "dft rattle static 7/12_mp-22905": "dft_ml_data_generation/rand_static_8_ISPIN2/",
            "dft rattle static 8/12_mp-22905": "dft_ml_data_generation/rand_static_8_ISPIN2/",
            "dft rattle static 9/12_mp-22905": "dft_ml_data_generation/rand_static_8_ISPIN2/",
            "dft rattle static 10/12_mp-22905": "dft_ml_data_generation/rand_static_12_ISPIN2/",
            "dft rattle static 11/12_mp-22905": "dft_ml_data_generation/rand_static_12_ISPIN2/",
            "dft rattle static 12/12_mp-22905": "dft_ml_data_generation/rand_static_12_ISPIN2/",
        }

        fake_run_vasp_kwargs = {
            "tight relax_mp-22905": {"incar_settings": ["ISPIN"]},
            "static_mp-22905": {"incar_settings": ["ISPIN"]},
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
            name="test_displacement_maker", # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set
        )
        test_rattled_bulk_relax_maker = TightRelaxMaker(
            name="test_bulk_rattled_maker", # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_phonon_bulk_relax_maker = TightRelaxMaker(
            name="test_bulk_phonon_maker", # overwritten by autoplex default
            input_set_generator=test_iso_atom_static_input_set,
        )
        test_phonon_static_energy_maker = StaticMaker(
            name="test_phonon_static_energy_maker", # overwritten by autoplex default
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
        ).make(
            structure_list=[structure],
            mp_ids=["mp-22905"],
            benchmark_mp_ids=["mp-22905"],
            benchmark_structures=[structure],
            pre_xyz_files=["vasp_ref.extxyz"],
            pre_database_dir=test_dir / "fitting" / "ref_files",
            apply_data_preprocessing=True,
            dft_references=None,
            **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
    ).make(structure_list=structure_list,
           mp_ids=mp_ids,
           benchmark_structures=structure_list,
           benchmark_mp_ids=mp_ids,
           pre_xyz_files=["vasp_ref.extxyz"],
           pre_database_dir=test_dir / "fitting" / "ref_files",
           apply_data_preprocessing=True,
           **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
           )

    flow_data_generation_without_rattled_structures = CompleteDFTvsMLBenchmarkWorkflow(
        n_structures=3, supercell_settings={"min_length": 10, "min_atoms": 20}, symprec=1e-2,
        add_dft_random_struct=False,
        volume_custom_scale_factors=[0.975, 1.0, 1.025, 1.05],
    ).make(structure_list=structure_list,
           mp_ids=mp_ids,
           benchmark_structures=structure_list,
           benchmark_mp_ids=mp_ids,
           pre_xyz_files=["vasp_ref.extxyz"],
           pre_database_dir=test_dir / "fitting" / "ref_files",
           apply_data_preprocessing=True,
           **{"general": {"two_body": True, "three_body": False, "soap": False}}  # reduce unit test run time
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
    for job, uuid in flow_data_generation.iterflow():
        counter += 1
    for job, uuid in flow_data_generation_without_rattled_structures.iterflow():
        counter_wor += 1
    assert counter == 7
    assert counter_wor == 6


# TODO testing cell_factor_sequence


def test_supercell_test_runs(vasp_test_dir, clean_dir, memory_jobstore, test_dir
                             ):
    from atomate2.forcefields.jobs import CHGNetStaticMaker
    from autoplex.auto.phonons.flows import DFTSupercellSettingsMaker

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    structure_list = [structure]
    mp_ids = ["mp-22905"]

    autoplex_flow = DFTSupercellSettingsMaker(supercell_settings={"min_length": 10, "min_atoms": 10},
                                              DFT_Maker=CHGNetStaticMaker()).make(
        structure_list=structure_list, mp_ids=mp_ids, )

    responses_flow = run_locally(autoplex_flow)
    assert responses_flow[autoplex_flow.jobs[-1].output.uuid][1].replace[0].name == "MLFF.CHGNet static"


@job
def mock_rss(input_dir: str = None,
             selection_method: str = 'cur',
             num_of_selection: int = 3,
             bcur_params: Optional[str] = None,
             random_seed: int = None,
             e0_spin: bool = False,
             isolated_atom: bool = True,
             dimer: bool = True,
             dimer_range: list = None,
             dimer_num: int = None,
             custom_incar: Optional[str] = None,
             vasp_ref_file: str = 'vasp_ref.extxyz',
             rss_group: str = 'initial',
             test_ratio: float = 0.1,
             regularization: bool = True,
             distillation: bool = True,
             f_max: float = 200,
             pre_database_dir: Optional[str] = None,
             mlip_type: str = 'GAP',
             ref_energy_name: str = "REF_energy",
             ref_force_name: str = "REF_forces",
             ref_virial_name: str = "REF_virial",
             num_processes_fit: int = None,
             kt: float = None,
             **fit_kwargs,):
    
    job2 = sample_data(selection_method=selection_method, 
                    num_of_selection=num_of_selection, 
                    bcur_params=bcur_params,
                    dir=input_dir,
                    random_seed=random_seed)
    job3 = DFTStaticLabelling(e0_spin=e0_spin, 
                       isolated_atom=isolated_atom, 
                       dimer=dimer,
                       dimer_range=dimer_range,
                       dimer_num=dimer_num,
                       custom_incar=custom_incar, 
                       ).make(structures=job2.output)
    job4 = collect_dft_data(vasp_ref_file=vasp_ref_file, 
                             rss_group=rss_group, 
                             vasp_dirs=job3.output)
    job5 = preprocess_data(test_ratio=test_ratio, 
                              regularization=regularization, 
                              distillation=distillation, 
                              force_max=f_max, 
                              vasp_ref_dir=job4.output['vasp_ref_dir'], pre_database_dir=pre_database_dir)
    job6 = MLIPFitMaker(mlip_type=mlip_type, 
                        ref_energy_name=ref_energy_name,
                        ref_force_name=ref_force_name,
                        ref_virial_name=ref_virial_name,
                        ).make(database_dir=job5.output, 
                               isolated_atom_energies=job4.output['isolated_atom_energies'],
                               num_processes_fit=num_processes_fit,
                               apply_data_preprocessing=False,
                               **fit_kwargs)
    job_list = [job2, job3, job4, job5, job6]

    return Response(
        replace=Flow(job_list),
        output={
            'test_error': job6.output['test_error'],
            'pre_database_dir': job5.output,
            'mlip_path': job6.output['mlip_path'],
            'isolated_atom_energies': job4.output['isolated_atom_energies'],
            'current_iter': 0,
            'kt': kt
        },
    )


@job
def mock_do_rss_iterations(input: Dict[str, Optional[Any]] = {'test_error': None,
                                                         'pre_database_dir': None,
                                                         'mlip_path': None,
                                                         'isolated_atom_energies': None,
                                                         'current_iter': None,
                                                         'kt': 0.6},
                      input_dir: str = None,
                      selection_method1: str = 'cur',
                      selection_method2: str = 'bcur1s',
                      num_of_selection1: int = 3,
                      num_of_selection2: int = 5,
                      bcur_params: Optional[str] = None,
                      random_seed: int = None,
                      mlip_type: str = 'GAP',
                      scalar_pressure_method: str ='exp',
                      scalar_exp_pressure: float = 100,
                      scalar_pressure_exponential_width: float = 0.2,
                      scalar_pressure_low: float = 0,
                      scalar_pressure_high: float = 50,
                      max_steps: int = 10,
                      force_tol: float = 0.1,
                      stress_tol: float = 0.1,
                      Hookean_repul: bool = False,
                      write_traj: bool = True,
                      num_processes_rss: int = 4,
                      device: str = "cpu",
                      stop_criterion: float = 0.01,
                      max_iteration_number: int = 9,
                      **fit_kwargs,):

    if input['test_error'] is not None and input['test_error'] > stop_criterion and input['current_iter'] < max_iteration_number:
        if input['kt'] > 0.15:
            kt = input['kt'] - 0.1
        else:
            kt = 0.1
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print(f'The error of {current_iter}th iteration:', input['test_error'])

        bcur_params['kt'] = kt

        job2 = sample_data(selection_method=selection_method1, 
                        num_of_selection=num_of_selection1, 
                        bcur_params=bcur_params,
                        dir=input_dir,
                        random_seed=random_seed)
        job3 = do_rss_single_node(mlip_type=mlip_type, 
                      iteration_index=f'{current_iter}th', 
                      mlip_path=input['mlip_path'], 
                      structures=job2.output,
                      scalar_pressure_method=scalar_pressure_method,
                      scalar_exp_pressure=scalar_exp_pressure,
                      scalar_pressure_exponential_width=scalar_pressure_exponential_width,
                      scalar_pressure_low=scalar_pressure_low,
                      scalar_pressure_high=scalar_pressure_high,
                      max_steps=max_steps,
                      force_tol=force_tol,
                      stress_tol=stress_tol,
                      hookean_repul=Hookean_repul,
                      write_traj=write_traj,
                      num_processes_rss=num_processes_rss,
                      device=device)
        job4 = sample_data(selection_method=selection_method2, 
                        num_of_selection=num_of_selection2, 
                        bcur_params=bcur_params,
                        traj_path=job3.output,
                        random_seed=random_seed,
                        isolated_atom_energies=input["isolated_atom_energies"])
        
        job_list = [job2, job3, job4]

        return Response(detour=job_list, output=job4.output)
    

@job
def mock_do_rss_iterations_multi_jobs(input: Dict[str, Optional[Any]] = {'test_error': None,
                                                         'pre_database_dir': None,
                                                         'mlip_path': None,
                                                         'isolated_atom_energies': None,
                                                         'current_iter': None,
                                                         'kt': 0.6},
                      input_dir: str = None,
                      selection_method1: str = 'cur',
                      selection_method2: str = 'bcur1s',
                      num_of_selection1: int = 3,
                      num_of_selection2: int = 5,
                      bcur_params: Optional[str] = None,
                      random_seed: int = None,
                      mlip_type: str = 'GAP',
                      scalar_pressure_method: str ='exp',
                      scalar_exp_pressure: float = 100,
                      scalar_pressure_exponential_width: float = 0.2,
                      scalar_pressure_low: float = 0,
                      scalar_pressure_high: float = 50,
                      max_steps: int = 10,
                      force_tol: float = 0.1,
                      stress_tol: float = 0.1,
                      Hookean_repul: bool = False,
                      write_traj: bool = True,
                      num_processes_rss: int = 4,
                      device: str = "cpu",
                      stop_criterion: float = 0.01,
                      max_iteration_number: int = 9,
                      num_groups: int = 2,
                      remove_traj_files: bool = True,
                      **fit_kwargs,):

    if input['test_error'] is not None and input['test_error'] > stop_criterion and input['current_iter'] < max_iteration_number:
        if input['kt'] > 0.15:
            kt = input['kt'] - 0.1
        else:
            kt = 0.1
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print(f'The error of {current_iter}th iteration:', input['test_error'])

        bcur_params['kT'] = kt

        job2 = sample_data(selection_method=selection_method1, 
                        num_of_selection=num_of_selection1, 
                        bcur_params=bcur_params,
                        dir=input_dir,
                        random_seed=random_seed)
        job3 = do_rss_multi_node(mlip_type=mlip_type, 
                      iteration_index=f'{current_iter}th', 
                      mlip_path=input['mlip_path'], 
                      structure=job2.output,
                      scalar_pressure_method=scalar_pressure_method,
                      scalar_exp_pressure=scalar_exp_pressure,
                      scalar_pressure_exponential_width=scalar_pressure_exponential_width,
                      scalar_pressure_low=scalar_pressure_low,
                      scalar_pressure_high=scalar_pressure_high,
                      max_steps=max_steps,
                      force_tol=force_tol,
                      stress_tol=stress_tol,
                      hookean_repul=Hookean_repul,
                      write_traj=write_traj,
                      num_processes_rss=num_processes_rss,
                      device=device,
                      num_groups=num_groups,)
        job4 = sample_data(selection_method=selection_method2, 
                        num_of_selection=num_of_selection2, 
                        bcur_params=bcur_params,
                        traj_path=job3.output,
                        random_seed=random_seed,
                        isolated_atom_energies=input["isolated_atom_energies"],
                        remove_traj_files=remove_traj_files)
        
        job_list = [job2, job3, job4]

        return Response(detour=job_list, output=job4.output)
    

def test_mock_workflow(test_dir, mock_vasp, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    # atoms = read(test_files_dir, index=':')
    # structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    ref_paths = {
        **{f"static_bulk_{i}": f"rss/Si_bulk_{i+1}/" for i in range(18)},
        "static_isolated_0": "rss/Si_isolated_1/",
        "static_dimer_0": "rss/Si_dimer_1/",
        "static_dimer_1": "rss/Si_dimer_2/",
        "static_dimer_2": "rss/Si_dimer_3/",
    }

    fake_run_vasp_kwargs = {
        "static_isolated_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_1": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_2": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job1=mock_rss(input_dir=test_files_dir,
                  selection_method='cur',
                  num_of_selection=18,
                  bcur_params={'soap_paras': {'l_max': 3,
                                    'n_max': 3,
                                    'atom_sigma': 0.5,
                                    'cutoff': 4.0,
                                    'cutoff_transition_width': 1.0,
                                    'zeta': 4.0,
                                    'average': True,
                                    'species': True,
                                    },
                 },
                 random_seed=42,
                 e0_spin=True,
                 isolated_atom=True,
                 dimer=False,
                 dimer_range=None,
                 dimer_num=None,
                 custom_incar={
                        "ADDGRID": None, 
                        "ENCUT": 200,
                        "EDIFF": 1E-04,
                        "ISMEAR": 0,
                        "SIGMA": 0.05,
                        "PREC": "Normal",
                        "ISYM": None,
                        "KSPACING": 0.3,
                        "NPAR": 8,
                        "LWAVE": "False",
                        "LCHARG": "False",
                        "ENAUG": None,
                        "GGA": None,
                        "ISPIN": None,
                        "LAECHG": None,
                        "LELF": None,
                        "LORBIT": None,
                        "LVTOT": None,
                        "NSW": None,
                        "SYMPREC": None,
                        "NELM": 50,
                        "LMAXMIX": None,
                        "LASPH": None,
                        "AMIN": None,
                    },
                 vasp_ref_file='vasp_ref.extxyz',
                 gap_rss_group='initial',
                 test_ratio=0.1,
                 regularization=True,
                 distillation=True,
                 f_max=0.7,
                 pre_database_dir=None,
                 mlip_type='GAP',
                 ref_energy_name="REF_energy",
                 ref_force_name="REF_forces",
                 ref_virial_name="REF_virial",
                 num_processes_fit=4,
                 kt=0.6
                )

    job2 = mock_do_rss_iterations(input=job1.output,
                      input_dir=test_files_dir,
                      selection_method1='cur',
                      selection_method2='bcur1s',
                      num_of_selection1=5,
                      num_of_selection2=3,
                      bcur_params={'soap_paras': {'l_max': 3,
                                   'n_max': 3,
                                   'atom_sigma': 0.5,
                                   'cutoff': 4.0,
                                   'cutoff_transition_width': 1.0,
                                   'zeta': 4.0,
                                   'average': True,
                                   'species': True,
                                   },
                                   'frac_of_bcur': 0.8,
                                   'bolt_max_num': 3000,
                                   'kernel_exp': 4.0, 
                                   'energy_label': 'energy'},
                      random_seed=None,
                      e0_spin=False,
                      isolated_atom=False,
                      dimer=False,
                      dimer_range=None,
                      dimer_num=None,
                      custom_incar=None,
                      vasp_ref_file='vasp_ref.extxyz',
                      rss_group='initial',
                      test_ratio=0.1,
                      regularization=True,
                      distillation=True,
                      f_max=200,
                      pre_database_dir=None,
                      mlip_type='GAP',
                      ref_energy_name="REF_energy",
                      ref_force_name="REF_forces",
                      ref_virial_name="REF_virial",
                      num_processes_fit=None,
                      scalar_pressure_method='exp',
                      scalar_exp_pressure=100,
                      scalar_pressure_exponential_width=0.2,
                      scalar_pressure_low=0,
                      scalar_pressure_high=50,
                      max_steps=100,
                      force_tol=0.6,
                      stress_tol=0.6,
                      Hookean_repul=False,
                      write_traj=True,
                      num_processes_rss=4,
                      device="cpu",
                      stop_criterion=0.01,
                      max_iteration_number=9
                      )

    response = run_locally(
        Flow([job1, job2]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    ) 

    assert Path(job1.output["mlip_path"].resolve(memory_jobstore)).exists()

    selected_atoms = job2.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 3


def test_mock_workflow_multi_node(test_dir, mock_vasp, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    # atoms = read(test_files_dir, index=':')
    # structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    ref_paths = {
        **{f"static_bulk_{i}": f"rss/Si_bulk_{i+1}/" for i in range(18)},
        "static_isolated_0": "rss/Si_isolated_1/",
        "static_dimer_0": "rss/Si_dimer_1/",
        "static_dimer_1": "rss/Si_dimer_2/",
        "static_dimer_2": "rss/Si_dimer_3/",
    }

    fake_run_vasp_kwargs = {
        "static_isolated_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_1": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
        "static_dimer_2": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}}, 
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job1=mock_rss(input_dir=test_files_dir,
                  selection_method='cur',
                  num_of_selection=18,
                  bcur_params={'soap_paras': {'l_max': 3,
                                    'n_max': 3,
                                    'atom_sigma': 0.5,
                                    'cutoff': 4.0,
                                    'cutoff_transition_width': 1.0,
                                    'zeta': 4.0,
                                    'average': True,
                                    'species': True,
                                    },
                 },
                 random_seed=42,
                 e0_spin=True,
                 isolated_atom=True,
                 dimer=False,
                 dimer_range=None,
                 dimer_num=None,
                 custom_incar={
                        "ADDGRID": None, 
                        "ENCUT": 200,
                        "EDIFF": 1E-04,
                        "ISMEAR": 0,
                        "SIGMA": 0.05,
                        "PREC": "Normal",
                        "ISYM": None,
                        "KSPACING": 0.3,
                        "NPAR": 8,
                        "LWAVE": "False",
                        "LCHARG": "False",
                        "ENAUG": None,
                        "GGA": None,
                        "ISPIN": None,
                        "LAECHG": None,
                        "LELF": None,
                        "LORBIT": None,
                        "LVTOT": None,
                        "NSW": None,
                        "SYMPREC": None,
                        "NELM": 50,
                        "LMAXMIX": None,
                        "LASPH": None,
                        "AMIN": None,
                    },
                 vasp_ref_file='vasp_ref.extxyz',
                 gap_rss_group='initial',
                 test_ratio=0.1,
                 regularization=True,
                 distillation=True,
                 f_max=0.7,
                 pre_database_dir=None,
                 mlip_type='GAP',
                 ref_energy_name="REF_energy",
                 ref_force_name="REF_forces",
                 ref_virial_name="REF_virial",
                 num_processes_fit=4,
                 kt=0.6
                )

    job2 = mock_do_rss_iterations_multi_jobs(input=job1.output,
                      input_dir=test_files_dir,
                      selection_method1='cur',
                      selection_method2='bcur1s',
                      num_of_selection1=5,
                      num_of_selection2=3,
                      bcur_params={'soap_paras': {'l_max': 3,
                                   'n_max': 3,
                                   'atom_sigma': 0.5,
                                   'cutoff': 4.0,
                                   'cutoff_transition_width': 1.0,
                                   'zeta': 4.0,
                                   'average': True,
                                   'species': True,
                                   },
                                   'frac_of_bcur': 0.8,
                                   'bolt_max_num': 3000,
                                   'kernel_exp': 4.0, 
                                   'energy_label': 'energy'},
                      random_seed=None,
                      e0_spin=False,
                      isolated_atom=True,
                      dimer=False,
                      dimer_range=None,
                      dimer_num=None,
                      custom_incar=None,
                      vasp_ref_file='vasp_ref.extxyz',
                      rss_group='initial',
                      test_ratio=0.1,
                      regularization=True,
                      distillation=True,
                      f_max=200,
                      pre_database_dir=None,
                      mlip_type='GAP',
                      ref_energy_name="REF_energy",
                      ref_force_name="REF_forces",
                      ref_virial_name="REF_virial",
                      num_processes_fit=None,
                      scalar_pressure_method='exp',
                      scalar_exp_pressure=100,
                      scalar_pressure_exponential_width=0.2,
                      scalar_pressure_low=0,
                      scalar_pressure_high=50,
                      max_steps=100,
                      force_tol=0.6,
                      stress_tol=0.6,
                      Hookean_repul=False,
                      write_traj=True,
                      num_processes_rss=4,
                      device="cpu",
                      stop_criterion=0.01,
                      max_iteration_number=9,
                      num_groups=2,
                      remove_traj_files=True,
                      )

    response = run_locally(
        Flow([job1, job2]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    ) 

    assert Path(job1.output["mlip_path"].resolve(memory_jobstore)).exists()

    selected_atoms = job2.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 3
