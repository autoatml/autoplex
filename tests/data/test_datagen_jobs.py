from pymatgen.core.structure import Structure
from autoplex.data.common.jobs import generate_randomized_structures, convert_to_extxyz, plot_force_distribution
from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle
)
from autoplex.data.phonons.jobs import reduce_supercell_size
import numpy as np
import pytest

@pytest.fixture(scope="class")
def mp_1200830():
    return Structure.from_dict(  # mp-1200830
        {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {
            'matrix': [[-6.2141887, -6.2141887, 6.2141887], [-6.2141887, 6.2141887, -6.2141887],
                       [6.2141887, -6.2141887, -6.2141887]], 'pbc': (True, True, True), 'a': 10.763290556220392,
            'b': 10.763290556220392, 'c': 10.763290556220392, 'alpha': 109.47122063449069, 'beta': 109.47122063449069,
            'gamma': 109.47122063449069, 'volume': 959.8719531108837}, 'properties': {}, 'sites': [
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.39175378, 0.13262899],
             'xyz': [-6.498293142493029, -3.277792458677283, 1.629429316776457], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.74087421, 0.60824622],
                              'xyz': [-3.2777862444885835, -1.6294355309651567, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.86737101, 0.25912579],
             'xyz': [-7.843618016776459, -0.28411687087042936, -2.9363962413227167], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.25912579, 0.86737101],
                              'xyz': [-0.28411687087042914, -7.843618016776459, -2.9363962413227167],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.13262899, 0.39175378],
             'xyz': [-3.2777924586772835, -6.498293142493028, 1.6294293167764566], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.60824622, 0.74087421],
                              'xyz': [-1.6294355309651563, -3.2777862444885835, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.65396589, 0.86737101],
             'xyz': [-0.28411687087042914, -2.9363962413227167, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.78659388, 0.39175378],
                              'xyz': [-3.2777924586772835, 1.6294293167764566, -6.498293142493028],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.3948401, 0.74087421],
             'xyz': [-1.6294355309651563, -5.930078043318272, -3.2777862444885835], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.78659388, 0.13262899],
                              'xyz': [-6.498293142493029, 1.629429316776457, -3.277792458677283],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.3948401, 0.60824622],
             'xyz': [-3.2777862444885835, -5.930078043318272, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.65396589, 0.25912579],
                              'xyz': [-7.843618016776459, -2.9363962413227167, -0.28411687087042936],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.60824622, 0.3948401],
             'xyz': [-5.930078043318271, -3.2777862444885835, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.25912579, 0.65396589],
                              'xyz': [-2.9363962413227167, -7.843618016776459, -0.28411687087042914],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.13262899, 0.78659388],
             'xyz': [1.629429316776457, -6.498293142493029, -3.277792458677283], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.74087421, 0.3948401],
                              'xyz': [-5.930078043318271, -1.6294355309651567, -3.2777862444885835],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.86737101, 0.65396589],
             'xyz': [-2.9363962413227167, -0.28411687087042914, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.39175378, 0.78659388],
                              'xyz': [1.629429316776457, -3.277792458677283, -6.498293142493029],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.34603411, 0.6051599],
             'xyz': [0.28411065668172863, -2.936402455511417, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.21340612, 0.34603411],
                              'xyz': [-2.936402455511417, -4.584753169034843, 0.28411065668172863],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.6051599, 0.21340612],
             'xyz': [-4.584753169034843, 0.28411065668172886, -2.936402455511417], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.21340612, 0.6051599],
                              'xyz': [0.28411065668172863, -4.584753169034843, -2.936402455511417],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.6051599, 0.34603411],
             'xyz': [-2.936402455511417, 0.28411065668172863, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.34603411, 0.21340612],
                              'xyz': [-4.584753169034843, -2.936402455511417, 0.28411065668172886],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.21736945],
             'xyz': [1.350774779915215, -1.350774779915215, -1.350774779915215], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.21736945, -0.0],
                              'xyz': [-1.350774779915215, 1.350774779915215, -1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78263055, 0.78263055, 0.78263055],
             'xyz': [-4.863413920084786, -4.863413920084786, -4.863413920084786], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21736945, 0.0, 0.0],
                              'xyz': [-1.350774779915215, -1.350774779915215, 1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.0, 0.0, -0.0], 'xyz': [0.0, 0.0, 0.0],
             'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.35932015, 0.22356876],
             'xyz': [-0.8435847537472929, 0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.86424861, 0.64067985],
                              'xyz': [-5.370603946252707, -2.592007022122684, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.77643124, 0.13575139],
             'xyz': [-8.806195722122682, -0.843584753747293, -0.843584753747293], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.13575139, 0.77643124],
                              'xyz': [-0.8435847537472929, -8.806195722122684, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.22356876, 0.35932015],
             'xyz': [0.8435847537472929, -0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.64067985, 0.86424861],
                              'xyz': [-2.592007022122684, -5.370603946252707, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13575139, 0.77643124, 0.77643124],
             'xyz': [-0.8435847537472929, -0.8435847537472929, -8.806195722122684], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, -0.0, 0.35932015],
                              'xyz': [0.8435847537472929, -3.6221816778773173, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, -0.0, 0.22356876],
             'xyz': [-0.8435847537472929, -3.6221816778773173, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86424861, 0.64067985, 0.64067985],
                              'xyz': [-5.370603946252707, -5.370603946252707, -2.592007022122684],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, 0.22356876, -0.0],
             'xyz': [-3.6221816778773173, -0.8435847537472929, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, 0.35932015, -0.0],
                              'xyz': [-3.6221816778773173, 0.8435847537472929, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'}]})


# test distort_type=0, i.e. volume distortion
def test_generate_randomized_structures_distort_type_0():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=0,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

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
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=1,
        n_structures=10,
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

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
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=2,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 11 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


def test_mc_rattle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    mc_rattle_job = mc_rattle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(mc_rattle_job) == 10
    for struct in mc_rattle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the rattled structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check if lattice parameters are unchanged
        assert (struct.lattice.matrix).all() == (structure.lattice.matrix).all()
        # check if atom positions are reasonably close to positions before rattling
        assert np.allclose(struct.frac_coords, structure.frac_coords, atol=0.05)


def test_std_rattle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    std_rattle_job = std_rattle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(std_rattle_job) == 10
    for struct in std_rattle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the rattled structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check if lattice parameters are unchanged
        assert (struct.lattice.matrix).all() == (structure.lattice.matrix).all()
        # check if atom positions are reasonably close to positions before rattling
        assert np.allclose(struct.frac_coords, structure.frac_coords, atol=0.05)


def test_random_vary_angle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    random_vary_angle_job = random_vary_angle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(random_vary_angle_job) == 10
    for struct in random_vary_angle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are reasonably close to those before distorting
        assert np.allclose((struct.lattice.matrix).all(), (structure.lattice.matrix).all(), atol=0.5)


# adapt to check for each input possible e.g. inputting range/manual scale_factors?
def test_scale_cell():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    scale_cell_job = scale_cell(structure=structure, volume_scale_factor_range=[0.90, 1.10], n_structures=10)

    # check if correct number of structures are generated
    assert len(scale_cell_job) == 11
    for struct in scale_cell_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are within +-10% of original value
        assert np.allclose(np.abs(np.array(struct.lattice.abc) - np.array(structure.lattice.abc)), 0,
                           atol=0.1 * np.array(structure.lattice.abc))


def test_convert_to_extxyz(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
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
    from jobflow import run_locally
    import glob
    import os

    dir = test_dir / "fitting"

    plot = plot_force_distribution(1.0, str(dir))

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

def test_supercell_check(mp_1200830, memory_jobstore):
    from jobflow import run_locally
    expected_matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    new_matrix_job = reduce_supercell_size(
        min_length=18,
        max_length=25,
        min_atoms=50,
        max_atoms=500,
        fallback_min_length=12,
        structure=mp_1200830,
        step_size=1.0
    )

    run_locally(new_matrix_job, create_folders=False, ensure_success=False, store=memory_jobstore)

    assert new_matrix_job.output.resolve(memory_jobstore) == expected_matrix
