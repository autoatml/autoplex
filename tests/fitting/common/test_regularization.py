"""Unit tests for the regularization functions in fitting.common."""

from ase.atoms import Atom
from ase.io import read

from autoplex.fitting.common.regularization import (
    calculate_hull_nd,
    get_convex_hull,
    get_e_distance_to_hull,
    get_e_distance_to_hull_nd,
    get_intersect,
    get_mole_frac,
    label_stoichiometry_volume,
    piecewise_linear,
    point_in_triangle_2D,
    point_in_simplex_nd,
    set_custom_sigma,
)


def test_set_sigma(test_dir):
    """Tests regularization."""
    # data setup
    test_atoms = read(test_dir / "fitting/pre_xyz_train_more_data.extxyz", ":")
    isol_es = {3: -0.28649227, 17: -0.28649227}
    reg_minmax = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]

    # test series of options for set_sigma

    atoms_modi = set_custom_sigma(
        test_atoms,
        reg_minmax,
        scheme="linear-hull",
    )
    assert atoms_modi[2].info["energy_sigma"] == 0.001

    atoms_modi = set_custom_sigma(
        test_atoms,
        reg_minmax,
        scheme="linear-hull",
        config_type_override={"test": [1e-4, 1e-4, 1e-4]},
    )
    assert atoms_modi[2].info["energy_sigma"] == 1e-4

    atoms_modi[0].info["REF_energy"] += 20
    for atoms in atoms_modi[:3]:
        atoms.set_cell([10, 10, 10])
    for atoms in atoms_modi[4:]:
        atoms.set_cell([11, 11, 11])
    atoms_modi = set_custom_sigma(
        test_atoms,
        reg_minmax,
        scheme="linear-hull",
        max_energy=0.05,
        isolated_atom_energies=isol_es,
        element_order=[3, 17],
    )
    assert len(atoms_modi) < len(test_atoms)

    atoms_modi[0].append(Atom("Li", [1, 1, 1]))
    atoms_modi = set_custom_sigma(
        test_atoms,
        reg_minmax,
        scheme="volume-stoichiometry",
        isolated_atom_energies=isol_es,
        element_order=[3, 17],
    )
    assert True  # TODO: modify this to test actual condition


def test_auxiliary_functions(test_dir, memory_jobstore, clean_dir):
    """Tests auxiliary functions related to convex hull."""
    import numpy as np
    from ase.io import read

    file = test_dir / "fitting" / "ref_files" / "quip_train.extxyz"
    atoms = read(file, ":")

    # Define the arrays
    array1 = np.array(
        [
            [15.2266087, -3.81106994],
            [15.2266087, -3.80983557],
            [16.2004607, -3.81927384],
            [8000.0, -0.28663766],
        ]
    )

    array2 = np.array(
        [
            [15.2266087, -3.80983557],
            [15.2266087, -3.81106994],
            [16.2004607, -3.81927264],
            [16.2004607, -3.81927384],
            [16.4281758, -3.81869979],
            [8000.0, -0.28663766],
            [17.6913485, -3.80636951],
            [17.6913485, -3.80665250],
            [19.0176670, -3.77969777],
            [8000.0, -0.27567309],
        ]
    )

    array3 = np.array(
        [
            [0.00000000e00, 8.00000000e03, -1.45390000e-04],
            [5.00000000e-01, 1.76913485e01, -3.53493109e00],
            [5.00000000e-01, 1.52266087e01, -3.53839715e00],
            [5.00000000e-01, 1.62004607e01, -3.54783542e00],
            [5.00000000e-01, 1.62004607e01, -3.54783422e00],
            [5.00000000e-01, 1.76913485e01, -3.53521408e00],
            [5.00000000e-01, 1.64281758e01, -3.54726137e00],
            [5.00000000e-01, 1.90176670e01, -3.50825935e00],
            [5.00000000e-01, 1.52266087e01, -3.53963152e00],
            [1.00000000e00, 8.00000000e03, -1.92885200e-02],
        ]
    )

    lower_half_hull_points, points = get_convex_hull(atoms, energy_name="REF_energy")

    # Function to compare sets of arrays with allclose after sorting
    def arrays_allclose(array1, array2):
        array1_sorted = np.array(sorted(array1, key=lambda x: tuple(x)))
        array2_sorted = np.array(sorted(array2, key=lambda x: tuple(x)))
        return np.allclose(array1_sorted, array2_sorted)

    assert arrays_allclose(lower_half_hull_points, array1)
    assert arrays_allclose(points, array2)

    label = label_stoichiometry_volume(
        atoms_list=atoms,
        isolated_atom_energies={3: -0.28649227, 17: -0.25638457},
        energy_name="REF_energy",
        element_order=[3, 17],
    )
    assert arrays_allclose(label, array3)

    calc_hull = calculate_hull_nd(points)
    calc_hull_ND = calculate_hull_nd(label)

    fraction_list = [[1.0]] + [[0.0]] + [[0.5]] * 8
    for atom, fraction in zip(atoms, fraction_list):
        get_e_dist_hull = get_e_distance_to_hull(
            calc_hull, atom, energy_name="REF_energy"
        )
        assert get_e_dist_hull == 0
        get_e_dist_hull_ND = get_e_distance_to_hull_nd(
            calc_hull_ND, atom, {3: -0.28649227, 17: -0.25638457}, "REF_energy"
        )
        assert round(get_e_dist_hull_ND) == 0
        getmole_frac = get_mole_frac(atom, element_order=[3, 17])
        assert getmole_frac == fraction

    point1, point2, point3, point4 = (1, 5), (2, 9), (8, 7), (9, 3)
    get_inter = get_intersect(point1, point2, point3, point4)
    assert get_inter == (4.75, 20.0)
    point_2d = point_in_triangle_2D(point1, point2, point3, point4)
    assert point_2d is False

    # Define test values
    vals = [
        (1.0, [1.0, 2.0, 3.0]),
        (2.0, [2.0, 3.0, 4.0]),
        (3.0, [3.0, 4.0, 5.0]),
        (4.0, [4.0, 5.0, 6.0]),
    ]

    # Define test values
    x = 2.5
    expected_result = np.array([2.5, 3.5, 4.5])

    piece_lin = piecewise_linear(x, vals)
    assert np.allclose(piece_lin, expected_result)

    # Define a test case for 2D (Triangle)
    point_2D_inside = np.array([0.5, 0.5])
    region_2D = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    point_2D_outside = np.array([1.5, 1.5])

    # Test 2D case
    inside_result_2D = point_in_simplex_nd(point_2D_inside, *region_2D)
    outside_result_2D = point_in_simplex_nd(point_2D_outside, *region_2D)

    # Point point_2D_inside inside region:
    assert inside_result_2D
    # Point point_2D_outside outside region:
    assert not outside_result_2D

    # Define a test case for 3D (Tetrahedron)
    point_3D_inside = np.array([0.25, 0.25, 0.25])
    region_3D = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    point_3D_outside = np.array([1.0, 1.0, 1.0])

    # Test 3D case
    inside_result_3D = point_in_simplex_nd(point_3D_inside, *region_3D)
    outside_result_3D = point_in_simplex_nd(point_3D_outside, *region_3D)

    # Point point_3D_inside inside region:
    assert inside_result_3D
    # Point point_3D_outside outside region:
    assert not outside_result_3D


def test_regularization_for_three_element_system(test_dir, memory_jobstore, clean_dir):
    """Tests auxiliary functions related to three_element_system."""
    import numpy as np
    from ase.io import read
    import numpy as np

    file = test_dir / "fitting" / "ref_files" / "test_regularization.extxyz"
    atoms = read(file, ":")

    points = label_stoichiometry_volume(
        atoms,
        isolated_atom_energies={3: 0.1, 15: 0.2, 16: 0.3},
        energy_name="energy",
        element_order=[3, 15, 16],
    )

    hull = calculate_hull_nd(points)

    des = [
        get_e_distance_to_hull_nd(
            hull,
            at,
            energy_name="energy",
            isolated_atom_energies={3: 0.1, 15: 0.2, 16: 0.3},
            element_order=[3, 15, 16],
        )
        for at in atoms
    ]
    
    print("Values in des:", des)
    assert all(d >= -1e-3 for d in des)

    reg_minmax = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]

    atoms_modi = set_custom_sigma(
        atoms,
        reg_minmax,
        scheme="volume-stoichiometry",
        isolated_atom_energies={3: 0.1, 15: 0.2, 16: 0.3},
        element_order=[3, 15, 16],
        energy_name="energy",
    )
    assert atoms_modi[2].info["energy_sigma"] == 1e-3
