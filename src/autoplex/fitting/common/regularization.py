"""Functions for automatic regularization and weighting of training data."""

import ast
import logging
import traceback
from contextlib import suppress

import numpy as np
from ase import Atoms
from scipy.spatial import ConvexHull

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# adapted from MorrowChem's RSS routines.
def set_custom_sigma(
    atoms: list[Atoms],
    reg_minmax: list[tuple],
    isolated_atom_energies: dict | None = None,
    scheme: str = "linear-hull",
    energy_name: str = "REF_energy",
    force_name: str = "REF_forces",
    virial_name: str = "REF_virial",
    element_order: list | None = None,
    max_energy: float = 20.0,
    config_type_override: dict | None = None,
    retain_existing_sigma: bool = False,
) -> list[Atoms]:
    """
    Handle automatic regularisation based on distance to convex hull, amongst other things.

    TODO: Need to make sure this works for full multi-stoichiometry systems.

    Parameters
    ----------
    atoms: (list of ase.Atoms)
        List of atoms objects to set reg. for. Usually fitting database
    reg_minmax: (list of tuples)
        List of tuples of (min, max) values for energy, force, virial sigmas
    scheme: (str)
        Method to use for regularization. Options are: linear_hull, volume-stoichiometry
        linear_hull: for single-composition system, use 2D convex hull (E, V)
        volume-stoichiometry: for multi-composition system, use 3D convex hull of (E, V, mole-fraction)
    energy_name: (str)
        Name of energy key in atoms.info
    force_name: (str)
        Name of force key in atoms.arrays
    virial_name: (str)
        Name of virial key in atoms.info
    isolated_atom_energies: (dict)
        Dictionary of isolated energies for each atomic number.
        Only needed for volume-x scheme e.g. {14: '-163.0', 8:'-75.0'}
        for SiO2
    element_order: (list)
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2)
    max_energy: (float)
        Ignore any structures with energy above hull greater than this value (eV)
    config_type_override: (dict)
        Give custom regularization for specific configuration types
    retain_existing_sigma: bool
        Whether to keep the current sigma values for specific configuration types.
        If set to True, existing sigma values for specific configurations will remain unchanged.

    e.g. reg_minmax = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]
    [(emin, emax), (semin, semax), (sfmin, sfmax), (ssvmin, svmax)]

    """
    sigs = [[], [], []]  # type: ignore
    atoms_modi = []

    if config_type_override is None:
        config_type_override = {
            "IsolatedAtom": (1e-4, 0.0, 0.0),
            "dimer": (0.1, 0.5, 0.0),
        }

    for at in atoms:
        for config_type, sigs_override in config_type_override.items():
            if at.info["config_type"] == config_type:
                at.info["energy_sigma"] = sigs_override[0]
                at.info["force_sigma"] = sigs_override[1]
                at.info["virial_sigma"] = sigs_override[2]
                atoms_modi.append(at)

        if at.info["config_type"] == "IsolatedAtom":
            at.calc = None  # TODO side-effect alert
            del at.info["force_sigma"]
            del at.info["virial_sigma"]
            continue

        if at.info["config_type"] == "dimer":
            with suppress(Exception):
                del at.info[virial_name]

            continue

    isolated_atom_energies = isolated_atom_energies or {}

    if scheme == "linear-hull":
        logging.info("Regularising with linear hull")
        hull, points = get_convex_hull(atoms, energy_name=energy_name)
        get_e_distance_func = get_e_distance_to_hull

    elif scheme == "volume-stoichiometry":
        logging.info("Regularising with 3D volume-mole fraction hull")
        if len(isolated_atom_energies) == 0:
            raise ValueError("Need to supply dictionary of isolated energies.")

        isolated_atom_energies = {
            ast.literal_eval(k) if isinstance(k, str) else k: v
            for k, v in isolated_atom_energies.items()
        }
        points = label_stoichiometry_volume(
            atoms, isolated_atom_energies, energy_name, element_order=element_order
        )  # label atoms with volume and mole fraction
        hull = calculate_hull_nd(points)
        get_e_distance_func = get_e_distance_to_hull_nd  # type: ignore

    points = {}
    for group in sorted(
        {  # check if set comprehension is running as supposed to
            at.info.get("rss_group")
            for at in atoms
            if not at.info.get("rss_nonperiodic")
        }
    ):
        points[group] = []

    for at in atoms:
        try:
            # skip non-periodic configs, volume is meaningless
            if at.info.get("rss_nonperiodic"):
                continue
            points[at.info.get("rss_group")].append(at)
        except Exception:
            pass

    for group, atoms_group in points.items():
        logging.info(f"group: {group}")

        for val in atoms_group:

            if retain_existing_sigma and "energy_sigma" in val.info:
                atoms_modi.append(val)
                continue

            de = get_e_distance_func(
                hull,
                val,
                energy_name=energy_name,
                isolated_atom_energies=isolated_atom_energies,
                element_order=element_order,
            )

            if de > max_energy:
                # don't even fit if too high
                continue
            if val.info["config_type"] not in config_type_override:
                if group == "initial":
                    sigs[0].append(reg_minmax[1][1])
                    sigs[1].append(reg_minmax[2][1])
                    sigs[2].append(reg_minmax[3][1])
                    val.info["energy_sigma"] = reg_minmax[1][1]
                    val.info["force_sigma"] = reg_minmax[2][1]
                    val.info["virial_sigma"] = reg_minmax[3][1]
                    atoms_modi.append(val)
                    continue

                if de <= reg_minmax[0][0]:
                    sigs[0].append(reg_minmax[1][0])
                    sigs[1].append(reg_minmax[2][0])
                    sigs[2].append(reg_minmax[3][0])
                    val.info["energy_sigma"] = reg_minmax[1][0]
                    val.info["force_sigma"] = reg_minmax[2][0]
                    val.info["virial_sigma"] = reg_minmax[3][0]
                    atoms_modi.append(val)

                elif de >= reg_minmax[0][1]:
                    sigs[0].append(reg_minmax[1][1])
                    sigs[1].append(reg_minmax[2][1])
                    sigs[2].append(reg_minmax[3][1])
                    val.info["energy_sigma"] = reg_minmax[1][1]
                    val.info["force_sigma"] = reg_minmax[2][1]
                    val.info["virial_sigma"] = reg_minmax[3][1]
                    atoms_modi.append(val)

                else:
                    [e, f, v] = piecewise_linear(
                        de,
                        [
                            (
                                0.1,
                                [reg_minmax[1][0], reg_minmax[2][0], reg_minmax[3][0]],
                            ),
                            (
                                1.0,
                                [reg_minmax[1][1], reg_minmax[2][1], reg_minmax[3][1]],
                            ),
                        ],
                    )
                    sigs[0].append(e)
                    sigs[1].append(f)
                    sigs[2].append(v)
                    val.info["energy_sigma"] = e
                    val.info["force_sigma"] = f
                    val.info["virial_sigma"] = v
                    atoms_modi.append(val)

    labels = ["E", "F", "V"]
    data_type = [np.array(sig) for sig in sigs]  # [e, f, v]

    for label, data in zip(labels, data_type):
        if len(data) == 0:
            logging.info(
                "No automatic regularisation performed (no structures requested)"
            )
            continue
        if label == "E":
            # Report of the regularisation statistics
            logging.info(
                f"Automatic regularisation statistics for {len(data)} structures:\n"
            )
            logging.info(
                "{:>20s}{:>20s}{:>20s}{:>20s}{:>20s}".format(
                    "", "Mean", "Std", "Nmin", "Nmax"
                )
            )
        logging.info(
            f"{label:>20s}"
            f"{data.mean():>20.4f}"
            f"{data.std():>20.4f}"
            f"{(data == data.min()).sum():>20d}"
            f"{(data == data.max()).sum():>20d}"
        )

    return atoms_modi


def get_convex_hull(
    atoms, energy_name="energy", **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the simple linear (E,V) convex hull.

    Parameters
    ----------
    atoms: list
        List of atoms objects.
    energy_name: str
        Name of the energy key in atoms.info (typically a DFT energy).

    Returns
    -------
    tuple
        A tuple containing two elements:
        - lower_half_hull_points: list of points (volume, energy) in the convex hull (lower half only).
        - p: list of all points for testing purposes.

    """
    points_list = []
    failed_count = 0

    for atom in atoms:
        if atom.info["config_type"] in ["IsolatedAtom", "dimer"]:
            continue
        try:
            volume_per_atom = atom.get_volume() / len(atom)
            energy_per_atom = (
                atom.info[energy_name] / len(atom)
                if energy_name != "energy"
                else atom.get_potential_energy() / len(atom)
            )
            points_list.append((volume_per_atom, energy_per_atom))
        except KeyError:
            failed_count += 1

    if failed_count > 0:
        raise ValueError(
            f"Convex hull failed to include {failed_count}/{len(atoms)} structures"
        )

    points = np.array(points_list)
    points = points.T[:, np.argsort(points.T[0])].T  # sort by volume axis

    hull = ConvexHull(points)  # generate full convex hull
    hull_points = points[hull.vertices]

    min_x_index = np.argmin(hull_points[:, 0])
    max_x_index = np.argmax(hull_points[:, 0])

    lower_half_hull = []
    i = min_x_index

    while True:
        lower_half_hull.append(hull.vertices[i])
        i = (i + 1) % len(hull.vertices)
        if i == max_x_index:
            lower_half_hull.append(hull.vertices[i])
            break

    lower_half_hull_points = points[lower_half_hull]

    lower_half_hull_points = lower_half_hull_points[
        lower_half_hull_points[:, 1] <= np.max(lower_half_hull_points[:, 1])
    ]

    return lower_half_hull_points, points


def get_e_distance_to_hull(
    hull: np.array, atoms, energy_name="energy", **kwargs
) -> float:
    """
    Calculate the distance of a structure to the linear convex hull in energy.

    Parameters
    ----------
    hull: (np.array)
        Points in the convex hull
    atoms: (Atoms)
        Structure to calculate distance to hull
    energy_name: (str)
        Name of the energy key in atoms.info (typically a DFT energy)

    """
    volume = atoms.get_volume() / len(atoms)
    energy = (
        atoms.info[energy_name] / len(atoms)
        if energy_name != "energy"
        else atoms.get_potential_energy() / len(atoms)
    )
    tp = np.array([volume, energy])
    hull_ps = hull.points if isinstance(hull, ConvexHull) else hull

    if any(
        np.isclose(hull_ps[:], tp).all(1)
    ):  # if the point is on the hull, return 0.0
        return 0.0

    nearest = np.searchsorted(hull_ps.T[0], tp, side="right")[
        0
    ]  # find the nearest convex hull point

    return (
        energy
        - get_intersect(
            tp,  # get intersection of the vertical line (energy axis)
            tp + np.array([0, 1]),  # and the line between the nearest hull points
            hull_ps[(nearest - 1) % len(hull_ps.T[0])],
            hull_ps[nearest % len(hull_ps.T[0])],
        )[1]
    )


def get_intersect(a1, a2, b1, b2) -> tuple[float, float] | tuple:
    """
    Return the point of intersection of the lines passing through a2,a1 and b2,b1.

    a1: [x, y]
        A point on the first line
    a2: [x, y]
        Another point on the first line
    b1: [x, y]
        A point on the second line
    b2: [x, y]
        Another point on the second line

    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return x / z, y / z


def get_mole_frac(atoms, element_order=None) -> float | int:
    """
    Calculate the mole-fraction of a structure.

    Parameters
    ----------
    atoms: (Atoms)
        Structure to calculate mole-fraction of
    element_order: (list)
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    Returns
    -------
    (x2, x3...): (array of float)
        Reduced mole-fraction of structure - first element n = 1-sum(others)

    """
    element, cts = np.unique(atoms.get_atomic_numbers(), return_counts=True)

    if element_order is None and len(element) < 3:  # compatibility with old version
        x = cts[1] / sum(cts) if len(element) == 2 else 1

    else:  # new version, requires element_order, recommended for all new calculations
        if element_order is None:
            element_order = element  # use default order
        not_in = [i for i in element_order if i not in element]
        for i in not_in:
            element = np.insert(element, -1, i)
            cts = np.insert(cts, -1, 0)

        cts = np.array(
            [cts[np.argwhere(element == i).squeeze()] for i in element_order]
        )
        element = np.array(
            [element[np.argwhere(element == i).squeeze()] for i in element_order]
        )

        x = cts[1:] / sum(cts)

    return x


def label_stoichiometry_volume(
    atoms_list: list[Atoms],
    isolated_atom_energies: dict,
    energy_name: str,
    element_order: list | None = None,
) -> np.ndarray:
    """
    Calculate the stoichiometry, energy, and volume coordinates for forming the convex hull.

    Parameters
    ----------
    atoms_list: (list[Atoms])
        List of atoms objects
    isolated_atom_energies: (dict)
        Dictionary of isolated atom energies {atomic_number: energy}
    energy_name: (str)
        Name of energy key in atoms.info (typically a DFT energy)
    element_order: (list | None)
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    """
    isolated_atom_energies = {
        ast.literal_eval(k) if isinstance(k, str) else k: v
        for k, v in isolated_atom_energies.items()
    }
    points_list = []
    for atom in atoms_list:
        try:
            volume = atom.get_volume() / len(atom)
            # make energy relative to isolated atoms
            energy = (
                atom.info[energy_name]
                - sum([isolated_atom_energies[j] for j in atom.get_atomic_numbers()])
                if energy_name != "energy"
                else atom.get_potential_energy()
                - sum([isolated_atom_energies[j] for j in atom.get_atomic_numbers()])
            ) / len(atom)
            mole_frac = get_mole_frac(atom, element_order=element_order)
            points_list.append(np.hstack((mole_frac, volume, energy)))
        except KeyError:
            traceback.print_exc()
    points = np.array(points_list)
    return points[np.lexsort(points.T[::-1])]


def point_in_triangle_2D(p1, p2, p3, pn) -> bool:
    """
    Check if a point is inside a triangle in 2D.

    Parameters
    ----------
    p1: (tuple)
        Coordinates of first point
    p2: (tuple)
        Coordinates of second point
    p3: (tuple)
        Coordinates of third point
    pn: (tuple)
        Coordinates of point to check

    """
    ep = 1e-4
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = pn

    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    c = 1 - a - b

    return (
        0 - ep <= a
        and a <= 1 + ep
        and 0 - ep <= b
        and b <= 1 + ep
        and 0 - ep <= c
        and c <= 1 + ep
    )


def point_in_simplex_nd(pn, *preg) -> bool:
    """
    Check if a point is inside a region of hyperplanes in N dimensions.

    Make a little convex hull in N-1 D and check this.

    Parameters
    ----------
    pn:
        Point to check (in ND)
    *preg:
        A list of points defining an (N-1)D simplex.

    """
    preg = np.array(preg)
    pn = np.array(pn)
    if preg.shape[1] != pn.shape[0]:
        raise ValueError(
            f"Point and points must have the same dimensionality. Got {pn.shape[0]} and {preg.shape[1]}."
        )
    hull = ConvexHull(preg)
    return np.all(np.dot(hull.equations[:, :-1], pn) + hull.equations[:, -1] <= 1e-10)


def calculate_hull_nd(points_nd) -> ConvexHull:
    """
    Calculate the convex hull in ND (N>=3).

    Parameters
    ----------
    points_nd:
        Point in ND.

    Returns
    -------
    Convex hull in ND.

    """
    p0 = np.array(
        [
            (points_nd[:, i].max() - points_nd[:, i].min()) / 2 + points_nd[:, i].min()
            for i in range(points_nd.shape[1] - 1)
        ]
        + [-1e6]
    )  # test point to get the visible facets from below
    pn = np.vstack((p0, points_nd))
    remove_dim = []

    for i in range(points_nd.shape[1]):
        if np.all(points_nd.T[i, 0] == points_nd.T[i, :]):
            pn = np.delete(pn, i, axis=1)
            logging.info(f"Convex hull lower dimensional - removing dimension {i}")
            remove_dim.append(i)

    hull = ConvexHull(pn, qhull_options="QG0")
    logging.info("done calculating hull")
    hull.remove_dim = remove_dim

    return hull


def get_e_distance_to_hull_nd(
    hull, atoms, isolated_atom_energies=None, energy_name="energy", element_order=None
) -> float:
    """
    Calculate the energy distance to the N-dimensional convex hull.

    Parameters
    ----------
    hull:
        Convex hull.
    atoms: (ase.Atoms)
        Structure to calculate mole-fraction of
    isolated_atom_energies: (dict)
        Dictionary of isolated atom energies
    energy_name: (str)
        Name of energy key in atoms.info (typically a DFT energy)
    element_order: (list)
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    """
    isolated_atom_energies = {
        ast.literal_eval(k) if isinstance(k, str) else k: v
        for k, v in isolated_atom_energies.items()
    }
    mole_frac = get_mole_frac(atoms, element_order=element_order)
    energy = (
        atoms.info[energy_name]
        - sum([isolated_atom_energies[j] for j in atoms.get_atomic_numbers()])
        if energy_name != "energy"
        else atoms.get_potential_energy()
        - sum([isolated_atom_energies[j] for j in atoms.get_atomic_numbers()])
    ) / len(atoms)
    volume = atoms.get_volume() / len(atoms)
    sp = np.hstack([mole_frac, volume, energy])

    for i in hull.remove_dim:
        sp = np.delete(sp, i)

    if len(sp[:-1]) == 1:
        logging.info("doing convexhull analysis in 1D")
        return get_e_distance_to_hull(hull, atoms, energy_name=energy_name)

    for _, visible_facet in enumerate(hull.simplices[hull.good]):
        if point_in_simplex_nd(sp[:-1], *hull.points[visible_facet][:, :-1]):
            n_d = hull.points[visible_facet]

            if n_d.shape[1] == 3:
                norm = np.cross(n_d[2] - n_d[0], n_d[1] - n_d[0])
                plane_norm = norm / np.linalg.norm(norm)
            else:
                relative_vectors = n_d[:-1] - n_d[0]
                plane_norm = np.linalg.lstsq(
                    relative_vectors, np.ones(relative_vectors.shape[0]), rcond=None
                )[0]

            plane_constant = np.dot(plane_norm, n_d[0])

            return (
                energy
                - (plane_constant - np.dot(plane_norm[:-1], sp[:-1])) / plane_norm[-1]
            )

    logging.info("Failed to find distance to hull in ND")
    return 1e6


def piecewise_linear(x, vals) -> np.ndarray:
    """
    Piecewise linear.

    Parameters
    ----------
    x:
        The x value.
    vals:
        The values

    """
    i = np.searchsorted([v[0] for v in vals], x)
    f0 = (vals[i][0] - x) / (vals[i][0] - vals[i - 1][0])
    return f0 * np.array(vals[i - 1][1]) + (1.0 - f0) * np.array(vals[i][1])
