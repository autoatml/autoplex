import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from autoplex.data.common.utils import (
    energy_plot,
    force_plot,
    plot_energy_forces,
    filter_outlier_energy,
    filter_outlier_forces,
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle
)


fig, ax_list = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(15, 20)


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


def test_energy_forces(clean_dir, test_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    plot_energy_forces(
        title="regularization 0.1",
        energy_limit=0.0005,
        force_limit=0.15,
        train_name='train_Si.extxyz',
        test_name='test_Si.extxyz'
    )
    assert os.path.isfile("energy_forces_Si.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_energy_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    energy = energy_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Energy on training data")
    plt.savefig("test_energy.png")
    assert os.path.isfile("test_energy.png")


def test_force_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    force = force_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Si", "Force on training data - Si", )
    plt.savefig("test_force.png")
    assert os.path.isfile("test_force.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_filter_outliers(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    filter_energy = filter_outlier_energy("train_Si.extxyz", "quip_train_Si.extxyz", 0.005)
    filter_forces = filter_outlier_forces("train_Si.extxyz", "quip_train_Si.extxyz", "Si", 0.1)

    assert os.path.isfile("filtered_in_energy_Si.extxyz")
    assert os.path.isfile("filtered_out_energy_Si.extxyz")
    assert os.path.isfile("outliers_energy_Si.extxyz")
    assert os.path.isfile("filtered_in_force_Si.extxyz")
    assert os.path.isfile("filtered_out_force_Si.extxyz")
    assert os.path.isfile("outliers_force_Si.extxyz")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_scale_cell(vasp_test_dir):
    from autoplex.data.common.utils import scale_cell

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    scaled_cell = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1.05])
    scaled_cell_1 = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1.0])
    scaled_cell_1_2 = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1])
    scaled_cell_2 = scale_cell(structure=structure, volume_custom_scale_factors=[0.95, 1.05, 1.10])
    scaled_cell_2_2 = scale_cell(structure=structure, volume_custom_scale_factors=[0.95, 1.0, 1.05, 1.10])

    assert len(scaled_cell) == 11
    assert len(scaled_cell_1) == 10
    assert len(scaled_cell_1_2) == 10
    assert len(scaled_cell_2) == 3
    assert len(scaled_cell_2_2) == 4
