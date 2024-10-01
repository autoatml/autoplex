import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
)
from atomate2.vasp.flows.phonons import PhononMaker


def test_phonon_wf_only_displacements3(mock_vasp, clean_dir):
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "phonon static 1/1": "Si_phonons_2/phonon_static_1_1",
        "static": "Si_phonons_2/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "phonon static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Generate job
    job = PhononMaker(
        min_length=3.0,
        bulk_relax_maker=None,
        born_maker=None,
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={"tstep": 100},
        get_supercell_size_kwargs={"step_size": 0.5}
    ).make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # !!! validation on the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5774.603553771463,
            5616.334060911681,
            4724.766198084037,
            3044.208072582665,
            696.3373193497828,
        ],
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    assert responses[job.jobs[-1].uuid][1].output.thermal_displacement_data is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures, [0, 100, 200, 300, 400]
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.has_imaginary_modes, False
    )
    assert responses[job.jobs[-1].uuid][1].output.force_constants is None
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert np.isclose(
        responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.74555232
    )
    assert responses[job.jobs[-1].uuid][1].output.born is None
    assert responses[job.jobs[-1].uuid][1].output.epsilon_static is None
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        (
            (0, 0.5000000000000001, 0.5000000000000001),
            (0.5000000000000001, 0.0, 0.5000000000000001),
            (0.5000000000000001, 0.5000000000000001, 0.0),
        ),
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            4.786689009990746,
            13.02544271435008,
            20.360935069065423,
            26.39830736008501,
        ],
    )
    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            8.047577577838897,
            15.97117761314484,
            19.97051059716143,
            21.87494655884403,
        ],
    )

    assert np.allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5774.603553771463,
            6095.002960937394,
            7329.854739783488,
            9152.488591840654,
            11255.660261586278,
        ],
    )


def test_cleanup_auxiliary_files(test_dir, clean_dir):
    import os
    import shutil

    # for some reason, not all the auxiliary files are removed at the end
    os.chdir(test_dir / "data" / "ref_data")
    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            file_path = os.path.join(os.getcwd(), file_name)
            try:
                # Check if it's a file or directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
