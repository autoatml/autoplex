import os 
os.environ["OMP_NUM_THREADS"] = "1"
from pymatgen.core.structure import Structure
from autoplex.data.common.jobs import generate_randomized_structures, convert_to_extxyz, plot_force_distribution
from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle
)
from autoplex.data.phonons.jobs import reduce_supercell_size_job
import numpy as np
import pytest
from autoplex.data.rss.jobs import RandomizedStructure, do_rss_single_node, do_rss_multi_node
from autoplex.data.common.jobs import sample_data
from jobflow import run_locally, Flow
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np


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
    expected_matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    new_matrix_job = reduce_supercell_size_job(
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


def test_extract_elements():
    rs = RandomizedStructure()
    elements = rs._extract_elements("SiO2")
    assert elements == {"Si": 1, "O": 2}
    
    elements = rs._extract_elements("H2O")
    assert elements == {"H": 2, "O": 1}
    
    elements = rs._extract_elements("C6H12O6")
    assert elements == {"C": 6, "H": 12, "O": 6}


def test_make_species():
    rs = RandomizedStructure()
    elements = {"Si": 1, "O": 2}
    species = rs._make_species(elements)
    assert species == "Si%NUM=1,O%NUM=2"
    
    elements = {"H": 2, "O": 1}
    species = rs._make_species(elements)
    assert species == "H%NUM=2,O%NUM=1"


def test_is_metal():
    from pymatgen.core import Element
    assert Element("Fe").is_metal == True
    assert Element("Si").is_metal == False


def test_make_minsep():
    rs = RandomizedStructure()
    radii = {"Si": 1.1, "O": 0.66}
    minsep = rs._make_minsep(radii)
    assert "Si-Si=1.7600000000000002" in minsep  # r1 * 1.8
    assert "Si-O=1.4080000000000004" in minsep  # (r1 + r2) / 2 * 1.5
    assert "O-O=1.056" in minsep   # r1 * 1.8


def test_update_buildcell_options():
    rs = RandomizedStructure()
    options = {'VARVOL': 20, 'SPECIES': 'Si%NUM=1,O%NUM=2'}
    buildcell_parameters = ['VARVOL=15',
                            'NFORM=1-7',
                            ]
    buildcell_update = rs._update_buildcell_option(options, buildcell_parameters)
    print("Updated buildcell parameters:", buildcell_update)
    assert 'VARVOL=20' in buildcell_update
    assert 'SPECIES=Si%NUM=1,O%NUM=2' in buildcell_update


def test_output_from_scratch(memory_jobstore, clean_dir):
    from ase.io import read
    from pathlib import Path
    import shutil
    job = RandomizedStructure(struct_number=3,
                              tag='SiO2',
                              output_file_name='random_structs.extxyz',
                              buildcell_option={'VARVOL': 20,
                                                'SYMMOPS':'1-2'},
                              num_processes=4).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert len(read(job.output.resolve(memory_jobstore), index=":")) == 3

        
def test_fragment_buildcell(test_dir, memory_jobstore, clean_dir):
    from ase.io import read
    from pathlib import Path
    import shutil
    import numpy as np
    from ase.build import molecule
    from ase.io import write
    
    ice_density = 0.0307 # molecules/A^3
    h2o = molecule('H2O')
    h2o.arrays['fragment_id'] = np.array([0,0,0])
    h2o.cell = np.ones(3)*20
    write(f'{test_dir}/data/h2o.xyz', h2o)
    
    job = RandomizedStructure(struct_number=4,
                              tag='water',
                              output_file_name='random_h20_structs.extxyz',
                              buildcell_option={'TARGVOL': f'{1/ice_density*0.8}-{1/ice_density*1.2}',
                                                'SYMMOPS': '1-4',
                                                'NFORM': '500',
                                                'MINSEP': '2.0',
                                                'SLACK': 0.25,
                                                'OVERLAP': 0.1,
                                                'SYSTEM': 'Cubi'
                                                },
                              fragment_file=os.path.join(f'{test_dir}/data', 'h2o.xyz'),
                              fragment_numbers=None,
                              remove_tmp_files=True,
                              num_processes=4).make()
    
    _ = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    ats = read(job.output.resolve(memory_jobstore), index=":")
    assert len(ats) == 4 and np.all(ats[0].positions[0] != ats[0].positions[1])


def test_output_from_cell_seed(test_dir, memory_jobstore, clean_dir):
    from ase.io import read
    from pathlib import Path
    import shutil
    test_files_dir = test_dir / "data/SiO2.cell"
    job = RandomizedStructure(struct_number=3,
                              cell_seed_path=test_files_dir,
                              num_processes=3).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert len(read(job.output.resolve(memory_jobstore),index=":")) == 3


def test_build_multi_randomized_structure(memory_jobstore, clean_dir):
    from autoplex.data.rss.flows import BuildMultiRandomizedStructure
    from autoplex.data.common.utils import flatten
    from pathlib import Path
    import shutil
    bcur_params={'soap_paras': {'l_max': 3,
                                'n_max': 3,
                                'atom_sigma': 0.5,
                                'cutoff': 4.0,
                                'cutoff_transition_width': 1.0,
                                'zeta': 4.0,
                                'average': True,
                                'species': True,
                                },
                }
    generate_structure = BuildMultiRandomizedStructure(tag="Si",
        generated_struct_numbers=[50,50],
        buildcell_options=[{'VARVOL': 20, 
                            'VARVOL_RANGE': '0.75 1.25',
                            'NATOM': '{6,8,10,12,14,16,18,20,22,24}',
                            'NFORM': '1'}, 
                           {'SYMMOPS':'1-2',
                            'NATOM': '{7,9,11,13,15,17,19,21,23}',
                            'NFORM': '1'}],
        num_processes=8,
        initial_selection_enabled=True,
        selected_struct_numbers=[8,2],
        bcur_params=bcur_params,
        random_seed=None).make()

    job = Flow(generate_structure, output=generate_structure.output) 
    responses = run_locally(job, 
                            ensure_success=True, 
                            create_folders=True, 
                            store=memory_jobstore)

    structures = job.output.resolve(memory_jobstore)

    n_atoms = [struct.num_sites for struct in flatten(structures, recursive=False)]

    assert max(n_atoms) < 25

    even_count = sum(1 for n in n_atoms if n % 2 == 0)
    odd_count = sum(1 for n in n_atoms if n % 2 != 0)

    assert even_count == 8
    assert odd_count == 2


def test_vasp_static(test_dir, memory_jobstore, clean_dir):
    from autoplex.data.common.jobs import preprocess_data
    test_files_dir = test_dir / "data/rss.extxyz"

    job = preprocess_data(test_ratio=0.1,
                             regularization=True,
                             distillation=True,
                             force_max=0.7,
                             vasp_ref_dir=test_files_dir,
                             pre_database_dir=None,)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    path_to_training_data = job.output.resolve(memory_jobstore)
    atom_train = read(os.path.join(path_to_training_data, 'train.extxyz'), index=":")
    atom_test = read(os.path.join(path_to_training_data, 'test.extxyz'), index=":")

    atoms = atom_train + atom_test
    f_component_max = []
    for at in atoms:
        forces = np.abs(at.arrays["REF_forces"])
        f_component_max.append(np.max(forces))

    assert len(atom_train) == 12
    assert len(atom_test) == 2
    assert "energy_sigma" in atom_train[0].info
    assert max(f_component_max) < 0.7


def test_gap_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    job = do_rss_single_node(mlip_type='GAP',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 2


def test_gap_rss_multi_jobs(test_dir, memory_jobstore, clean_dir):
    from ase.units import GPa
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:2:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    job = do_rss_multi_node(mlip_type='GAP',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=1000,
                force_tol=0.01,
                stress_tol=0.0001,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938},
                num_groups=2,)
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i:
            output_filter.append(i)
   
    assert len(output_filter) == 2

    ats = read(output_filter[0][0])

    enthalpy_pseudo = ats.info["enthalpy"]
    enthalpy_cal = ats.get_potential_energy() + ats.info["RSS_applied_pressure"]*GPa*ats.get_volume()
    
    assert round(enthalpy_pseudo,3) == round(enthalpy_cal,3)


def test_jace_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:1:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/JACE"

    job = do_rss_single_node(mlip_type='J-ACE',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.01,
                stress_tol=0.0001,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_nequip_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/NEQUIP"

    job = do_rss_single_node(mlip_type='NEQUIP',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_m3gnet_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/M3GNET/m3gnet_results/training"

    job = do_rss_single_node(mlip_type='M3GNET',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_mace_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/MACE"

    job = do_rss_single_node(mlip_type='MACE',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_sampling_cur_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job = sample_data(
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
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_bcur1s_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='bcur1s',
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
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_random_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='random',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_uniform_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='uniform',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))
