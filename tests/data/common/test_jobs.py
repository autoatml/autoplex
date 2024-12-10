import os
from pymatgen.core.structure import Structure
from autoplex.data.common.jobs import (
    generate_randomized_structures,
    convert_to_extxyz,
    plot_force_distribution,
    sample_data
)
from jobflow import run_locally
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

os.environ["OMP_NUM_THREADS"] = "1"


# test distort_type=0, i.e. volume distortion
def test_generate_randomized_structures_distort_type_0():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rattledts_job = generate_randomized_structures(
        structure=structure,
        distort_type=0,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        rattle_type=0)

    responses = run_locally(rattledts_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 11 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


# test distort_type=1, i.e. angle distortion
def test_generate_randomized_structures_distort_type_1():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rattledts_job = generate_randomized_structures(
        structure=structure,
        distort_type=1,
        n_structures=10,
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rattledts_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 10 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


# test distort_type=2, i.e. simultaneous volume and angle distortion
def test_generate_randomized_structures_distort_type_2():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rattledts_job = generate_randomized_structures(
        structure=structure,
        distort_type=2,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rattledts_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 11 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


def test_convert_to_extxyz(test_dir, memory_jobstore, clean_dir):
    class Job:  # create a dummy job object
        def __init__(self, dir_name):
            self.output = self.Output(dir_name)

        class Output:
            def __init__(self, dir_name):
                self.dir_name = dir_name

            def as_dict(self):
                return {"dir_name": self.dir_name}

    dir = test_dir / "pkls"
    file = "LiCl_10.pkl"
    job = Job(dir)

    conv = convert_to_extxyz(job.output, file, "bulk", "10")

    responses = run_locally(
        conv, create_folders=True, ensure_success=False, store=memory_jobstore
        # atomate2 switched from pkl to json files for the trajectories, therefore success False.
        # This job is also not needed for the main workflow
    )


def test_plot_force_distribution(test_dir, memory_jobstore, clean_dir):
    import glob
    import os

    dir_path = test_dir / "fitting"
    plot = plot_force_distribution(1.0, str(dir_path))
    run_locally(
        plot, create_folders=False, ensure_success=False, store=memory_jobstore
    )
    assert os.path.isfile("total_data.png")
    files_remove = glob.glob("*.png")
    for file in files_remove:
        try:
            os.remove(file)
            print(f'Removed file: {file}')
        except Exception as e:
            print(f'Error removing file {file}: {e}')


def test_sampling_cur_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job_sample = sample_data(
        selection_method='cur',
        num_of_selection=5,
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
        structure=structures,
    )

    response = run_locally(
        job_sample,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job_sample.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_bcur1s_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job_sample = sample_data(selection_method='bcur1s',
                             num_of_selection=5,
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
                                          'energy_label': 'REF_energy'
                                          },
                             structure=structures,
                             isolated_atom_energies={14: -0.84696938},
                             random_seed=42)

    response = run_locally(
        job_sample,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job_sample.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_random_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job_sample = sample_data(selection_method='random',
                             num_of_selection=5,
                             structure=structures,
                             random_seed=42)

    response = run_locally(
        job_sample,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job_sample.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_uniform_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job_sample = sample_data(selection_method='uniform',
                             num_of_selection=5,
                             structure=structures,
                             random_seed=42)

    response = run_locally(
        job_sample,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job_sample.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))
