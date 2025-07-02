"""Utility functions for training data jobs."""

import logging
import os
import random
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.atoms import Atom
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory as AseTrajectory
from ase.io import write
from ase.units import GPa
from hiphive.structure_generation import generate_mc_rattled_structures
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from quippy import descriptors
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from autoplex.fitting.common.regularization import (
    calculate_hull_nd,
    get_convex_hull,
    get_e_distance_to_hull,
    get_e_distance_to_hull_nd,
    label_stoichiometry_volume,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def flatten(atoms_object: Atoms | Iterable, recursive: bool = False) -> list[Atoms]:
    """
    Flatten an iterable fully, but excluding Atoms objects.

    Parameters
    ----------
    atoms_object: Atoms or Iterable
        An Atoms object or an iterable containing Atoms objects.
    recursive: bool
        If set to True, the function will recursively flatten the iterable.

    Returns
    -------
    A flattened list containing only Atoms objects.

    """
    iteration_list = []

    if recursive:
        for element in atoms_object:
            if isinstance(element, Iterable) and not isinstance(
                element, (str, bytes, ase.atoms.Atoms, ase.Atoms)
            ):
                iteration_list.extend(flatten(element, recursive=True))
            else:
                iteration_list.append(element)
        return iteration_list

    return [item for sublist in atoms_object for item in sublist]


def rms_dict(x_ref: np.ndarray | list, x_pred: np.ndarray | list) -> dict:
    """Compute RMSE and standard deviation of predictions with reference data.

    Adapted and adjusted from libatoms GAP tutorial page
    https://libatoms.github.io/GAP/gap_fitting_tutorial.html#make-simple-plots-of-the-energies-and-forces-on-the-EMT-and-GAP-datas

    Parameters
    ----------
    x_ref: np.ndarray.
        List of reference data.
    x_pred: np.ndarray.
        List of prediction.
    Note that x_ref and x_pred should be of same shape.

    Returns
    -------
    dict
        A dict with RMSE and std deviation of predictions.
    """
    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError("WARNING: not matching shapes in rms")
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))
    return {"rmse": average, "std": std_}


def to_ase_trajectory(
    traj_obj, filename: str = "atoms.traj", store_magmoms=False
) -> AseTrajectory:  # adapted from ase
    """
    Convert to an ASE .Trajectory.

    Parameters
    ----------
    traj_obj:
        Trajectory object.
    filename : str | None
        Name of the file to write the ASE trajectory to.
        If None, no file is written.
    store_magmoms:
        The bool to store magnetic moments.
    """
    for idx in range(len(traj_obj["atom_positions"])):
        atoms = Atoms(symbols=list(traj_obj["atomic_number"]))  # .atoms.copy()
        atoms.set_positions(traj_obj["atom_positions"][idx])
        atoms.set_cell(traj_obj["cell"][idx])
        atoms.set_pbc("T T T")

        kwargs = {
            "energy": traj_obj["energy"][idx],
            "forces": traj_obj["forces"][idx],
            "stress": traj_obj["stresses"][idx],
        }
        if store_magmoms:
            kwargs["magmom"] = traj_obj.magmoms[idx]

        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
        with AseTrajectory(filename, "a" if idx > 0 else "w", atoms=atoms) as f:
            f.write()

    return AseTrajectory(filename, "r")


def scale_cell(
    structure: Structure,
    volume_scale_factor_range: list[float] | None = None,
    n_structures: int = 10,
    volume_custom_scale_factors: list[float] | None = None,
) -> list[Structure]:
    """
    Take in a pymatgen Structure object and generates stretched or compressed structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
    n_structures : int.
        If specified a range, the number of structures to be generated with
        volume distortions equally spaced between min and max.
    volume_custom_scale_factors : list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].

    Returns
    -------
    Response.output.
        Stretched or compressed structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_cells = []

    if volume_custom_scale_factors is not None:
        scale_factors_defined = volume_custom_scale_factors
        warnings.warn("Using your custom lattice scale factors", stacklevel=2)
    if volume_custom_scale_factors is None:
        if volume_scale_factor_range is None:
            volume_scale_factor_range = [0.90, 1.1]

        scale_factors_defined = np.arange(
            volume_scale_factor_range[0],
            volume_scale_factor_range[1]
            + (volume_scale_factor_range[1] - volume_scale_factor_range[0])
            / (n_structures - 1),
            (volume_scale_factor_range[1] - volume_scale_factor_range[0])
            / (n_structures - 1),
        )

        if not np.isclose(scale_factors_defined, 1.0).any():
            scale_factors_defined = np.append(scale_factors_defined, 1)
            scale_factors_defined = np.sort(scale_factors_defined)

        warnings.warn(
            f"Generated lattice scale factors {scale_factors_defined} within your range",
            stacklevel=2,
        )

    for scale_factor in scale_factors_defined:
        # make copy of ground state
        cell = atoms.copy()
        # set lattice parameter scale factor
        lattice_scale_factor = scale_factor ** (1 / 3)
        # scale cell volume and atomic positions
        cell.set_cell(lattice_scale_factor * atoms.get_cell(), scale_atoms=True)
        # store scaled cell
        distorted_cells.append(AseAtomsAdaptor.get_structure(cell))
    return distorted_cells


def check_distances(structure: Structure, min_distance: float = 1.5) -> bool:
    """
    Take in a pymatgen Structure object and check minimum distances between atoms using minimum image convention.

    Useful after distorting cell angles and rattling to check atoms aren't too close.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between any two atoms. Default= 1.5A.

    Returns
    -------
    Response.output.
        "True" if atoms are sufficiently spaced out i.e. all pairwise interatomic distances > min_distance.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)

    for i in range(len(atoms)):
        indices = [j for j in range(len(atoms)) if j != i]
        distances = atoms.get_distances(i, indices, mic=True)
        for distance in distances:
            if distance < min_distance:
                warnings.warn("Atoms too close.", stacklevel=2)
                return False
    return True


def random_vary_angle(
    structure: Structure,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    w_angle: list[float] | None = None,
    n_structures: int = 8,
    angle_max_attempts: int = 1000,
) -> list[Structure]:
    """
    Take in a pymatgen Structure object and generates angle-distorted structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between atoms. Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    n_structures: int.
        Number of angle-distorted structures to generate.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.

    Returns
    -------
    Response.output.
        Angle-distorted structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_angle_cells = []
    generated_structures = 0  # counter to keep track of generated structures

    if w_angle is None:
        w_angle = [0, 1, 2]

    while generated_structures < n_structures:
        attempts = 0
        # make copy of ground state
        atoms_copy = atoms.copy()

        # stretch lattice parameters by 3% before changing angles
        # helps atoms to not be too close
        distorted_cells = scale_cell(
            AseAtomsAdaptor.get_structure(atoms_copy),
            volume_custom_scale_factors=[1.03],
        )

        distorted_supercells: Atoms = AseAtomsAdaptor.get_atoms(distorted_cells[0])

        # getting stretched supercell out of array
        newcell = distorted_supercells.cell.cellpar()

        # current angles
        alpha = atoms_copy.cell.cellpar()[3]
        beta = atoms_copy.cell.cellpar()[4]
        gamma = atoms_copy.cell.cellpar()[5]

        # convert angle distortion scale
        angle_percentage_scale = angle_percentage_scale / 100
        min_scale = 1 - angle_percentage_scale
        max_scale = 1 + angle_percentage_scale

        while attempts < angle_max_attempts:
            attempts += 1
            # new random angles within +-10% (default) of current angle
            new_alpha = random.randint(int(alpha * min_scale), int(alpha * max_scale))
            new_beta = random.randint(int(beta * min_scale), int(beta * max_scale))
            new_gamma = random.randint(int(gamma * min_scale), int(gamma * max_scale))

            newvalues = [new_alpha, new_beta, new_gamma]

            for wang, newv in zip(w_angle, newvalues):
                newcell[wang + 3] = newv

            # converting newcell back into an Atoms object so future functions work
            # scaling atoms to new distorted cell
            atoms_copy.set_cell(newcell, scale_atoms=True)

            # if successful structure generated, i.e. atoms are not too close, then break loop
            if check_distances(AseAtomsAdaptor.get_structure(atoms_copy), min_distance):
                # store scaled cell
                distorted_angle_cells.append(AseAtomsAdaptor.get_structure(atoms_copy))
                generated_structures += 1
                break  # break the inner loop if successful
        else:
            raise RuntimeError(
                "Maximum attempts (1000) reached without distorting angles successfully."
            )

    return distorted_angle_cells


def std_rattle(
    structure: Structure,
    n_structures: int = 5,
    rattle_std: float = 0.01,
    rattle_seed: int = 42,
) -> list[Structure]:
    """
    Take in a pymatgen Structure object and generates rattled structures.

    Uses standard ASE rattle.

    Parameters
    ----------
    structure : Structure.
        The pymatgen structures object.
    n_structures: int.
        Number of rattled structures to generate.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution). Default=0.01.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated. Default= 42.

    Returns
    -------
    Response.output.
        Rattled structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    rattled_xtals = []
    for i in range(n_structures):
        if i == 0:
            copy = atoms.copy()
            copy.rattle(stdev=rattle_std, seed=rattle_seed)
            rattled_xtals.append(AseAtomsAdaptor.get_structure(copy))
        if i > 0:
            rattle_seed = rattle_seed + 1
            copy = atoms.copy()
            copy.rattle(stdev=rattle_std, seed=rattle_seed)
            rattled_xtals.append(AseAtomsAdaptor.get_structure(copy))
    return rattled_xtals


def mc_rattle(
    structure: Structure,
    n_structures: int = 5,
    rattle_std: float = 0.003,
    min_distance: float = 1.5,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
) -> list[Structure]:
    """
    Take in a pymatgen Structure object and generates rattled structures.

    Randomly draws displacements with a MC trial step that penalises displacements leading to very small interatomic
    distances.
    Displacements generated will roughly be n_iter**0.5 * rattle_std for small values of n_iter.
    See https://hiphive.materialsmodeling.org/moduleref/structures.html?highlight=generate_mc_rattled_structures for
    more details.

    Parameters
    ----------
    structure : Structure.
        The pymatgen structures object.
    n_structures: int.
        Number of rattled structures to generate.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution). N.B. this value is not connected to the final
        average displacement for the structures. Default= 0.003.
    min_distance: float.
        Minimum separation of any two atoms in the rattled structures. Used for computing the probability for each
        rattle move. Default= 1.5A.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated. Default= 42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations. Larger number of iterations will generate larger displacements. Default=10.

    Returns
    -------
    Response.output.
        Monte-Carlo rattled structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    mc_rattle = generate_mc_rattled_structures(
        atoms=atoms,
        n_structures=n_structures,
        rattle_std=rattle_std,
        d_min=min_distance,
        seed=rattle_seed,
        n_iter=rattle_mc_n_iter,
    )
    return [AseAtomsAdaptor.get_structure(xtal) for xtal in mc_rattle]


def filter_outlier_energy(
    in_file: str, out_file: str, criteria: float = 0.0005
) -> None:
    """
    Filter data outliers via energy criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    criteria:
        Energy filter threshold.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms:
        kwargs = {
            "energy": atoms.info["REF_energy"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")

    atoms_in = []
    atoms_out = []
    outliers = []

    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        en_in = at_in.get_potential_energy() / len(at_in.get_chemical_symbols())
        en_out = at_out.get_potential_energy() / len(at_out.get_chemical_symbols())
        en_error = rms_dict(en_in, en_out)
        if abs(en_error["rmse"]) < criteria:
            atoms_in.append(at_in)
            atoms_out.append(at_out)
        else:
            outliers.append(at_in)

    output_files = {
        "filtered_in_energy": atoms_in,
        "filtered_out_energy": atoms_out,
        "outliers_energy": outliers,
    }

    # Iterate over output files and write them
    for suffix, atom_list in output_files.items():
        path = Path(in_file).with_name(Path(in_file).name.replace("train", suffix))
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        write(path, atom_list, append=True)


def filter_outlier_forces(
    in_file: str, out_file: str, symbol="Si", criteria: float = 0.1
) -> None:
    """
    Filter data outliers via force criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    symbol:
        Chemical symbol.
    criteria:
        Force filter threshold.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms[:-1]:
        kwargs = {
            "forces": atoms.arrays["REF_forces"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")
    atoms_in = []
    atoms_out = []
    outliers = []
    # extract data for only one species
    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        # get the symbols
        sym_all = at_in.get_chemical_symbols()
        # add force for each atom
        force_error = []
        for j, sym in enumerate(sym_all):
            if sym in symbol:
                in_force = at_in.get_forces()[j]
                out_force = at_out.arrays["force"][j]
                rms = rms_dict(in_force, out_force)
                force_error.append(rms["rmse"])
        at_in.info["max RMSE"] = max(force_error) if force_error else 0
        at_in.info["avg RMSE"] = (
            sum(force_error) / len(force_error) if force_error else 0
        )
        at_in.info["RMSE"] = force_error
        if not any(np.any(value > criteria) for value in force_error):
            atoms_in.append(at_in)
            atoms_out.append(at_out)
        else:
            outliers.append(at_in)

    output_files = {
        "filtered_in_force": atoms_in,
        "filtered_out_force": atoms_out,
        "outliers_force": outliers,
    }

    # Iterate over output files and write them
    for suffix, atom_list in output_files.items():
        path = Path(in_file).with_name(Path(in_file).name.replace("train", suffix))
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        write(path, atom_list, append=True)


def energy_plot(
    in_file: str | Path,
    out_file: str | Path,
    ax: plt.Axes,
    title: str = "Plot of energy",
    label: str = "energy",
) -> float:
    """
    Plot the distribution of energy per atom on the output vs the input.

    Adapted and adjusted from libatoms GAP tutorial page
    https://libatoms.github.io/GAP/gap_fitting_tutorial.html#make-simple-plots-of-the-energies-and-forces-on-the-EMT-and-GAP-datas

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    ax:
        Panel position in plt.subplots.
    title:
        Title of the plot.
    label:
        Legend label

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    out_atoms = ase.io.read(out_file, ":")
    atoms_in: list = []
    atoms_out: list = []
    for at_in, at_out in zip(in_atoms, out_atoms):
        kwargs = {
            "energy": at_in.info["REF_energy"],
        }
        at_in.calc = SinglePointCalculator(atoms=at_in, **kwargs)
        for at, atoms in [(at_in, atoms_in), (at_out, atoms_out)]:
            if at.info["config_type"] not in {"isolated_atom", "IsolatedAtom"}:
                atoms.append(at)
    # list energies
    ener_in = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in atoms_in
    ]
    ener_out = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in atoms_out
    ]
    # scatter plot of the data
    ax.scatter(ener_in, ener_out, label=label)
    # get the appropriate limits for the plot
    for_limits = np.array(ener_in + ener_out)
    elim = (for_limits.min() - 0.01, for_limits.max() + 0.01)
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    # add line of slope 1 for reference
    ax.plot(elim, elim, c="k")
    # set labels
    ax.set_ylabel("energy by GAP / eV")
    ax.set_xlabel("energy by DFT / eV")
    # set title
    ax.set_title(title)
    # set legend
    ax.legend(loc="upper right")
    # add text about RMSE
    _rms = rms_dict(ener_in, ener_out)
    rmse_text = (
        "RMSE:\n"
        + str(np.round(_rms["rmse"], 7))
        + " +- "
        + str(np.round(_rms["std"], 9))
        + "eV/atom"
    )
    ax.text(
        0.9,
        0.1,
        rmse_text,
        transform=ax.transAxes,
        fontsize="large",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    return _rms["rmse"]


def force_plot(
    in_file: str | Path,
    out_file: str | Path,
    ax: plt.Axes,
    symbol: str = "Si",
    title: str = "Plot of force",
    label: str = "force for ",
) -> float:
    """
    Plot the distribution of force components per atom on the output vs the input.

    Only plots for the given atom type(s).

    Adapted and adjusted from libatoms GAP tutorial page
    https://libatoms.github.io/GAP/gap_fitting_tutorial.html#make-simple-plots-of-the-energies-and-forces-on-the-EMT-and-GAP-datas

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    ax:
        Panel position in plt.subplots.
    symbol:
        Chemical symbol.
    title:
        Title of the plot.
    label:
        Legend label

    """
    in_atoms = ase.io.read(in_file, ":")

    for atoms in in_atoms[:-1]:
        kwargs = {
            "forces": atoms.arrays["REF_forces"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")
    # extract data for only one species
    in_force, out_force = [], []
    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        # get the symbols
        sym_all = at_in.get_chemical_symbols()
        # add force for each atom
        for j, sym in enumerate(sym_all):
            if sym in symbol:
                in_force.append(at_in.get_forces()[j])
                # out_force.append(at_out.get_forces()[j]) \
                out_force.append(
                    at_out.arrays["force"][j]
                )  # because QUIP and ASE use different names
    # convert to np arrays, much easier to work with
    # in_force = np.array(in_force)
    # out_force = np.array(out_force)
    # scatter plot of the data
    ax.scatter(in_force, out_force, label=label + symbol)
    # get the appropriate limits for the plot
    for_limits = np.array(in_force + out_force)
    flim = (for_limits.min() - 1, for_limits.max() + 1)
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    # add line of
    ax.plot(flim, flim, c="k")
    # set labels
    ax.set_ylabel("force by GAP / (eV/Å)")
    ax.set_xlabel("force by DFT / (eV/Å)")
    # set title
    ax.set_title(title)
    # set legend
    ax.legend(loc="upper right")
    # add text about RMSE
    _rms = rms_dict(in_force, out_force)
    rmse_text = (
        "RMSE:\n"
        + str(np.round(_rms["rmse"], 3))
        + " +- "
        + str(np.round(_rms["std"], 5))
        + "eV/Å"
    )
    ax.text(
        0.9,
        0.1,
        rmse_text,
        transform=ax.transAxes,
        fontsize="large",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    return _rms["rmse"]


def plot_energy_forces(
    title: str,
    energy_limit: float,
    force_limit: float,
    species_list: list | None = None,
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
) -> None:
    """
    Plot energy and forces of the data.

    Adapted and adjusted from libatoms GAP tutorial page
    https://libatoms.github.io/GAP/gap_fitting_tutorial.html#make-simple-plots-of-the-energies-and-forces-on-the-EMT-and-GAP-datas

    Parameters
    ----------
    title: str
        Title of the plot.
    energy_limit: float
        Energy limit for data filtering.
    force_limit: list
        Force limit for data filtering.
    species_list: str
        List of species.
    train_name: str
        Name of the training data file.
    test_name: str
        Name of the test data file.
    """
    quip_train_file = train_name.replace("train", "quip_train")
    quip_test_file = test_name.replace("test", "quip_test")
    path = Path(train_name)
    if species_list is None:
        species_list = ["Si"]
    fig, ax_list = plt.subplots(nrows=3, ncols=2, gridspec_kw={"hspace": 0.3})
    fig.set_size_inches(10, 15)
    ax_list = ax_list.flat[:]
    rmse = ["Energy and forces and train and test data\n"]

    pretty_species_list = (
        str(species_list).replace("['", "").replace("']", "").replace("'", "")
    )

    energy_rmse_train = energy_plot(
        train_name, quip_train_file, ax_list[0], "Energy on training data"
    )
    rmse.append(f"Energy train: {energy_rmse_train}")
    for species in species_list:
        force_rmse_train = force_plot(
            train_name,
            quip_train_file,
            ax_list[1],
            species,
            f"Force on training data - {pretty_species_list}",
        )
        rmse.append(f"Force train {species}: {force_rmse_train}")
    energy_rmse_test = energy_plot(
        test_name, quip_test_file, ax_list[2], "Energy on test data"
    )
    rmse.append(f"Energy test: {energy_rmse_test}")

    filter_outlier_energy(train_name, quip_train_file, energy_limit)
    # filter_outlier_energy(test_name, quip_test_file, energy_limit)
    for species in species_list:
        force_rmse_test = force_plot(
            test_name,
            quip_test_file,
            ax_list[3],
            species,
            f"Force on test data - {pretty_species_list}",
        )
        rmse.append(f"Force test {species}: {force_rmse_test}")
        filter_outlier_forces(train_name, quip_train_file, species, force_limit)
        # filter_outlier_forces(test_name, quip_test_file, species, force_limit)

    energy_plot(
        path.with_name(path.name.replace("train", "filtered_in_energy")),
        path.with_name(path.name.replace("train", "filtered_out_energy")),
        ax_list[4],
        "Energy on filtered data",
        "energy-filtered data, energy",
    )
    for species in species_list:
        force_plot(
            path.with_name(path.name.replace("train", "filtered_in_force")),
            path.with_name(path.name.replace("train", "filtered_out_force")),
            ax_list[5],
            species,
            f"Force on filtered data - {pretty_species_list}",
            "energy-filtered data, force for ",
        )
    energy_plot(
        path.with_name(path.name.replace("train", "filtered_in_energy")),
        path.with_name(path.name.replace("train", "filtered_out_energy")),
        ax_list[4],
        "Energy on filtered data",
        "force-filtered data, energy",
    )
    for species in species_list:
        force_plot(
            path.with_name(path.name.replace("train", "filtered_in_force")),
            path.with_name(path.name.replace("train", "filtered_out_force")),
            ax_list[5],
            species,
            f"Force on filtered data - {pretty_species_list}",
            "force-filtered data, force for ",
        )

    fig.suptitle(title, fontsize=16)
    plt.savefig(train_name.replace("train", "energy_forces").replace(".extxyz", ".png"))

    with open("energy_train_rmse.txt", "a") as file:
        for entry in rmse:
            file.write(f"{entry}\n")


class ElementCollection:
    """
    A class to handle different species operations for a collection of atoms.

    The Species class provides methods to extract unique chemical elements (species),
    determine all possible pairs of these species, and retrieve their atomic numbers
    in a formatted string.
    """

    def __init__(self, atoms):
        self.atoms = atoms

    def get_species(self) -> list:
        """Extract a list of unique species (chemical elements) from the atoms."""
        sepcies_list = []

        for at in self.atoms:
            sym_all = at.get_chemical_symbols()
            syms = list(set(sym_all))
            for sym in syms:
                if sym in sepcies_list:
                    continue
                sepcies_list.append(sym)

        return sepcies_list

    def find_element_pairs(self, symb_list=None) -> list:
        """
        Generate a list of all possible unique pairs of species.

        It can operate on an optional list of symbols or default to
        using the species extracted from the atoms.

        Parameters
        ----------
        symb_list:
            List of chemical symbols.

        """
        species_list = self.get_species() if symb_list is None else symb_list

        pairs = []

        for i in range(len(species_list)):
            for j in range(i, len(species_list)):
                pair = (species_list[i], species_list[j])
                pairs.append(pair)

        return pairs

    def get_number_of_species(self) -> int:
        """Return the number of unique species present among the atoms."""
        return len(self.get_species())

    def get_species_Z(self) -> str:
        """Return a formatted string of atomic numbers of the unique species."""
        atom_numbers = []
        for atom_type in self.get_species():
            atom = Atoms(atom_type, [(0, 0, 0)])
            atom_numbers.append(int(atom.get_atomic_numbers()[0]))

        species_Z = "{"
        for i in range(len(atom_numbers) - 1):
            species_Z += str(atom_numbers[i]) + " "
        species_Z += str(atom_numbers[-1]) + "}"

        return species_Z


def parallel_calc_descriptor_vec(atom: Atoms, selected_descriptor: str) -> Atoms:
    """
    Calculate the SOAP descriptor vector for a given atom and hypers in parallel.

    Parameters
    ----------
    atom : ase.Atoms
        The atom for which to calculate the descriptor vector.
    selected_descriptor : str
        The quip descriptor string to use for the calculation.

    Returns
    -------
    ase.Atom
        The input atoms, with the descriptor vector added to its info dictionary.

    Notes
    -----
    The descriptor vector is added to the atom's info dictionary with the key 'descriptor_vec'.
    """
    desc_object = descriptors.Descriptor(selected_descriptor)
    atom.info["descriptor_vec"] = desc_object.calc(atom)["data"]

    return atom


def cur_select(
    atoms,
    selected_descriptor,
    kernel_exp,
    select_nums,
    stochastic=True,
    random_seed=None,
) -> list[Atoms] | None:
    """
    Perform CUR selection on a set of atoms to get representative SOAP descriptors.

    Parameters
    ----------
    atoms: list of ase.Atoms
        The atoms for which to perform CUR selection.
    selected_descriptor: str
        The quip descriptor string to use for the calculation.
    kernel_exp: float
        The kernel exponent to use in the calculation.
    select_nums: int
        The number of atoms to select.
    stochastic: bool
        Whether to perform stochastic CUR selection.
    random_seed: int
        The seed for the random number generator.

    Returns
    -------
    list of ase.Atoms
        The selected atoms.

    Notes
    -----
    This function calculates the descriptor vector for each atom,
    then performs CUR selection on the resulting vectors.

    References
    ----------
    *    Title: Research data supporting "De novo exploration and self-guided learning of potential-energy surfaces"
    *    Script: select_by_descriptor.py
    *    Author: Noam Bernstein, Gábor Csányi and Volker L. Deringer
    *    Date 11/10/2019
    *    Availability: https://www.repository.cam.ac.uk/items/3aff252b-a583-4e7c-afc9-9dc1540cc37e
    *    License: Attribution 4.0 International (CC BY 4.0) license.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(atoms[0], list):
        print("flattening")
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms

    num_workers = min(len(fatoms), os.cpu_count() or 1)

    descriptor_worker = partial(
        parallel_calc_descriptor_vec, selected_descriptor=selected_descriptor
    )
    with threadpool_limits(limits=1), Pool(processes=num_workers) as pool:
        # TODO: implement argument for number of cores throughout
        ats = pool.map(descriptor_worker, fatoms)

    if isinstance(ats, list) & (len(ats) != 0):
        at_descs = np.array([at.info["descriptor_vec"] for at in ats]).T
        m = (
            np.matmul((np.squeeze(at_descs)).T, np.squeeze(at_descs)) ** kernel_exp
            if kernel_exp > 0.0
            else at_descs
        )

        def descriptor_svd(at_descs, num, do_vectors="vh"):
            def mv(v):
                return np.dot(at_descs, v)

            def rmv(v):
                return np.dot(at_descs.T, v)

            A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)
            return svds(A, k=num, return_singular_vectors=do_vectors)

        (_, _, vt) = descriptor_svd(
            m, min(max(1, int(select_nums / 2)), min(m.shape) - 1)
        )
        c_scores = np.sum(vt**2, axis=0) / vt.shape[0]
        if stochastic:
            selected = sorted(
                np.random.choice(
                    range(len(ats)), size=select_nums, replace=False, p=c_scores
                )
            )
        else:
            selected = sorted(np.argsort(c_scores)[-select_nums:])

        selected_atoms = [ats[i] for i in selected]

        for at in selected_atoms:
            del at.info["descriptor_vec"]

        return selected_atoms

    return None


def boltzhist_cur_one_shot(
    atoms: list[Atoms] | list[list[Atoms]],
    descriptor: str,
    isolated_atom_energies: dict,
    bolt_frac: float = 0.1,
    bolt_max_num: int = 3000,
    cur_num: int = 100,
    kernel_exp: float = 4,
    kt: float = 0.3,
    energy_label: str = "energy",
    pressures: list[float] | list[list[float]] | None = None,
    random_seed: int = None,
) -> list | None:
    """
    Sample atoms from a list according to boltzmann energy weighting and CUR diversity.

    Parameters
    ----------
    atoms: list[ase.Atoms] or list[list[ase.Atoms]]
        The atoms from which to select. If this is a list of lists, it is flattened.
    descriptor: str
        The quippy SOAP descriptor string for CUR.
    isolated_atom_energies: dict
        Dictionary of isolated energy values for species. Required for 'boltzhist_cur'
        selection method.
    bolt_frac: float
        The fraction to control the flat Boltzmann selection number.
    bolt_max_num: int
        The maximum number of atoms to select by Boltzmann-weighted flat histogram.
    cur_num: int
        The number of atoms to select by CUR.
    kernel_exp: float
        The exponent for the dot-product SOAP kernel.
    kt: float
        The product of the Boltzmann constant and the temperature, in eV.
    energy_label: str
        The label for the energy property in the atoms.
    pressures: list[float] or list[list[float]]
        The pressures at which the atoms have been optimized, in GPa.
    random_seed : int
        The seed for the random number generator.

    Returns
    -------
    list of ase.Atoms
        The selected atoms. These are copies of the atoms in the input list.

    Notes
    -----
    The algorithm uses a combination of CUR selection and Boltzmann weighting
    to select the atoms with diversity and low energy.

    References
    ----------
    *    Title: Research data supporting "De novo exploration and self-guided learning of potential-energy surfaces"
    *    Script: select_enthalpy_flat_histogram.py
    *    Author: Noam Bernstein, Gábor Csányi and Volker L. Deringer
    *    Date 11/10/2019
    *    Availability: https://www.repository.cam.ac.uk/items/3aff252b-a583-4e7c-afc9-9dc1540cc37e
    *    License: Attribution 4.0 International (CC BY 4.0) license.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(atoms[0], list):
        print("flattening")
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms

    if pressures is None:
        logging.info("Pressures not supplied, attempting to use pressure in atoms dict")

        try:
            ps = np.array([at.info["RSS_applied_pressure"] for at in fatoms])
        except RuntimeError:
            print("No pressures, so can't Boltzmann weight")

    else:
        ps = flatten_list(pressures)

    enthalpies = []

    at_ids = [atom.get_atomic_numbers() for atom in fatoms]

    if energy_label == "energy":
        formation_energies = []
        for ct, atom in enumerate(fatoms):
            if "energy" in atom.info:
                formation_energy = atom.info["energy"] - sum(
                    [isolated_atom_energies[j] for j in at_ids[ct]]
                )
            else:
                formation_energy = atom.get_potential_energy() - sum(
                    [isolated_atom_energies[j] for j in at_ids[ct]]
                )
            formation_energies.append(formation_energy)
        formation_energies = np.array(formation_energies)

    else:
        formation_energies = np.array(
            [
                atom.info[energy_label]
                - sum([isolated_atom_energies[j] for j in at_ids[ct]])
                for ct, atom in enumerate(fatoms)
            ]
        )

    for i, at in enumerate(fatoms):
        enthalpy = (formation_energies[i] + at.get_volume() * ps[i] * GPa) / len(at)
        enthalpies.append(enthalpy)

    enthalpies = np.array(enthalpies)
    min_H = np.min(enthalpies)
    config_prob = []
    histo = np.histogram(enthalpies)
    for H in enthalpies:
        bin_i = np.searchsorted(histo[1][1:], H, side="right")
        if bin_i == len(histo[1][1:]):
            bin_i = bin_i - 1
        p = 1.0 / histo[0][bin_i] if histo[0][bin_i] > 0.0 else 0.0
        if kt > 0.0:
            p *= np.exp(-(H - min_H) / kt)
        config_prob.append(p)

    select_num = round(bolt_frac * len(fatoms))

    select_num = select_num if select_num < bolt_max_num else bolt_max_num

    config_prob = np.array(config_prob)
    selected_bolt_ats = []
    for _ in range(select_num):
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)
        rv = np.random.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        selected_bolt_ats.append(fatoms[config_i])
        # remove from config_prob by converting to list
        config_prob = np.delete(config_prob, config_i)
        # remove from other lists
        del fatoms[config_i]
        enthalpies = np.delete(enthalpies, config_i)

    if cur_num < select_num:
        selected_atoms = cur_select(
            atoms=selected_bolt_ats,
            selected_descriptor=descriptor,
            kernel_exp=kernel_exp,
            select_nums=cur_num,
            stochastic=True,
            random_seed=random_seed,
        )
    else:
        selected_atoms = selected_bolt_ats

    return selected_atoms


def boltzhist_cur_dual_iter(
    atoms: list[Atoms] | list[list[Atoms]],
    descriptor: str,
    isolated_atom_energies: dict,
    bolt_frac: float = 0.1,
    bolt_max_num: int = 3000,
    cur_num: int = 100,
    kernel_exp: float = 4,
    kt: float = 0.3,
    energy_label: str = "energy",
    pressures: list[list[float]] | None = None,
    random_seed: int = None,
) -> list | None:
    """
    Execute sampling with two iterations.

    Each iteration includes a Boltzmann flat histogram in enthalpy
    followed by a CUR process.

    Parameters
    ----------
    atoms: list[ase.Atoms] or list[list[ase.Atoms]]
        The atoms from which to select. If this is a list of lists, it is flattened.
    descriptor: str
        The quippy SOAP descriptor string for CUR.
    isolated_atom_energies: dict
        Dictionary of isolated energy values for species. Required for 'boltzhist_cur'
        selection method. Default is None.
    bolt_frac: float
        The fraction to control the flat Boltzmann selection number.
    bolt_max_num: int
        The maximum number of atoms to select by Boltzmann-weighted flat histogram.
    cur_num: int
        The number of atoms to select by CUR.
    kernel_exp: float
        The exponent for the dot-product SOAP kernel.
    kt: float
        The product of the Boltzmann constant and the temperature, in eV.
    energy_label: str
        The label for the energy property in the atoms.
    pressures: list[float] or list[list[float]]
        The pressures at which the atoms have been optimized, in GPa.
    random_seed : int
        The seed for the random number generator.

    Returns
    -------
    list of ase.Atoms
        The selected atoms. These are copies of the atoms in the input list.

    Notes
    -----
    This function selects the most diverse atoms based on the chosen algorithm.
    The algorithm uses a combination of CUR selection and Boltzmann weighting to select the atoms.
    """
    atom_minima = [ats[-1] for ats in atoms]
    minima_indices = [ats[-1].info["unique_starting_index"] for ats in atoms]
    pressure_minima = None if pressures is None else [p[-1] for p in pressures]

    selected_minima = boltzhist_cur_one_shot(
        atoms=atom_minima,
        descriptor=descriptor,
        isolated_atom_energies=isolated_atom_energies,
        bolt_frac=bolt_frac,
        bolt_max_num=bolt_max_num,
        cur_num=cur_num,
        kernel_exp=kernel_exp,
        kt=kt,
        energy_label=energy_label,
        pressures=pressure_minima,
        random_seed=random_seed,
    )

    if selected_minima is None:
        raise ValueError(
            "The structures obtained from the first bcur sampling cannot be None."
        )

    selected_minima_indices = [
        at.info["unique_starting_index"] for at in selected_minima
    ]
    selected__indices = [minima_indices.index(item) for item in selected_minima_indices]
    selected_trajs = [atoms[i] for i in selected__indices]
    selected_trajs_pressure = (
        None if pressures is None else [pressures[j] for j in selected__indices]
    )

    return boltzhist_cur_one_shot(
        atoms=selected_trajs,
        descriptor=descriptor,
        isolated_atom_energies=isolated_atom_energies,
        bolt_frac=bolt_frac,
        bolt_max_num=bolt_max_num,
        cur_num=cur_num,
        kernel_exp=kernel_exp,
        kt=kt,
        energy_label=energy_label,
        pressures=selected_trajs_pressure,
        random_seed=random_seed,
    )


def convexhull_cur(
    atoms: list[Atoms],
    descriptor: str,
    bolt_frac: float = 0.1,
    bolt_max_num: int = 3000,
    cur_num: int = 100,
    kernel_exp: float = 4,
    kt: float = 0.5,
    energy_label: str = "REF_energy",
    isolated_atom_energies: dict = None,
    element_order: list | None = None,
    scheme: str = "linear-hull",
) -> list | None:
    """
    Sample atoms from a list according to Boltzmann energy weighting relative to convex hull and CUR diversity.

    Parameters
    ----------
    atoms: list of ase.Atoms
        The atoms for which to perform CUR selection.
    bolt_frac: float
        The fraction to control the proportion of atoms kept
        during the Boltzmann selection step.
    bolt_max_num: int
        The maximum number of atoms to select by Boltzmann flat
        histogram.
    cur_num: int
        The number of atoms to select by CUR.
    kernel_exp: float
        The kernel exponent to use in the calculation.
    kt: float
        The product of the Boltzmann constant and the temperature,
        in eV.
    energy_label: str
        The label for the energy property in the atoms.
    descriptor: str, optional
        The quip descriptor string to use for the calculation.
    isolated_atom_energies: dict, optional
        The isolated atom energies for each element in the system.
    element_order: list of str, optional
        The order of elements for the isolated atom energies.
    scheme: str, optional
        The scheme to use for the convex hull calculation.
        Default is 'linear-hull' (2D E,V hull).
        For 2-component systems with varying stoichiometry,
        use 'volume-stoichiometry' (>=3D E,V,mole-fraction hull).
        TODO: need to generalise this to ND hulls for mcp systems.
        GST good test case.

    Returns
    -------
    list of ase.Atoms
        The selected atoms.

    Notes
    -----
    This function calculates the descriptor vector for each atom,
    then performs CUR selection on the resulting vectors.
    The selection is based on the convex hull of the vectors.
    """
    if isinstance(atoms[0], list):
        print("flattening")
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms

    if isolated_atom_energies is None:
        raise KeyError("isolated_atom_energies must be supplied for convexhull_cur")

    if scheme == "linear-hull":
        hull, p = get_convex_hull(fatoms, energy_name=energy_label)
        des = np.array(
            [
                get_e_distance_to_hull(hull, at, energy_name=energy_label)
                for at in fatoms
            ]
        )

    elif scheme == "volume-stoichiometry":
        points = label_stoichiometry_volume(
            fatoms,
            isolated_atom_energies=isolated_atom_energies,
            energy_name=energy_label,
            element_order=element_order,
        )
        hull = calculate_hull_nd(points)

        des = np.array(
            [
                get_e_distance_to_hull_nd(
                    hull,
                    at,
                    isolated_atom_energies=isolated_atom_energies,
                    energy_name=energy_label,
                    element_order=element_order,
                )
                for at in fatoms
            ]
        )
        print("it will be coming soon!")

    else:
        raise ValueError(
            'scheme must be either "linear-hull" or "volume-stoichiometry"'
        )

    histo = np.histogram(des)
    config_prob = []
    min_ec = np.min(des)

    for ec in des:
        bin_i = np.searchsorted(histo[1][1:], ec, side="right")
        if bin_i == len(histo[1][1:]):
            bin_i = bin_i - 1
        p = 1.0 / histo[0][bin_i] if histo[0][bin_i] > 0.0 else 0.0
        if kt > 0.0:
            p *= np.exp(-(ec - min_ec) / kt)
        config_prob.append(p)

    select_num = round(bolt_frac * len(fatoms))

    select_num = select_num if select_num < bolt_max_num else bolt_max_num

    config_prob = np.array(config_prob)
    selected_bolt_ats = []
    for _ in range(select_num):
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)  # cumulate prob
        rv = np.random.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        selected_bolt_ats.append(fatoms[config_i])
        # remove from config_prob by converting to list
        config_prob = np.delete(config_prob, config_i)
        # remove from other lists
        del fatoms[config_i]
        des = np.delete(des, config_i)

    # implement CUR
    if cur_num < select_num:
        selected_atoms = cur_select(
            atoms=selected_bolt_ats,
            selected_descriptor=descriptor,
            kernel_exp=kernel_exp,
            select_nums=cur_num,
            stochastic=True,
        )
    else:
        selected_atoms = selected_bolt_ats

    return selected_atoms


def data_distillation(
    vasp_ref_dir: str, force_max: float, force_label: str
) -> list[Atom | Atoms]:
    """
    For data distillation.

    Parameters
    ----------
    vasp_ref_dir: str
        VASP reference data directory.
    force_max: float
        Maximally allowed force.
    force_label: str
        The label for the force property in the atoms.

    Returns
    -------
    atoms_distilled:
        List of distilled atoms.

    """
    atoms = ase.io.read(vasp_ref_dir, index=":")

    atoms_distilled = []
    for at in atoms:
        forces = np.abs(at.arrays[force_label])
        f_component_max = np.max(forces)

        if f_component_max < force_max:
            atoms_distilled.append(at)

    logging.warning(
        f"After distillation, there are still {len(atoms_distilled)} data points remaining."
    )

    return atoms_distilled


def stratified_dataset_split(
    atoms: Atoms, split_ratio: float, energy_label: str
) -> tuple[
    list[Atom | Atoms]
    | list[Atom | Atoms | list[Atom | Atoms] | list[Atom | Atoms | list]],
    list[Atom | Atoms | list[Atom | Atoms] | list[Atom | Atoms | list]],
]:
    """
    Split the dataset.

    Parameters
    ----------
    atoms: Atoms
        ASE Atoms object
    split_ratio: float
        Parameter to divide the training set and the test set.
    energy_label: str
        The label for the energy property in the atoms.

    Returns
    -------
    train_structures, test_structures:
        Split-up datasets of train structures and test structures.

    """
    atom_bulk = []
    atom_isolated_and_dimer = []
    for at in atoms:
        if (
            at.info["config_type"] != "dimer"
            and at.info["config_type"] != "IsolatedAtom"
        ):
            atom_bulk.append(at)
        else:
            atom_isolated_and_dimer.append(at)

    if len(atoms) != len(atom_bulk):
        atoms = atom_bulk

    # Need this try except block because the energy label is not present as info
    try:
        average_energies = np.array(
            [atom.info[energy_label] / len(atom) for atom in atoms]
        )
    except KeyError:
        average_energies = np.array(
            [atom.get_potential_energy() / len(atom) for atom in atoms]
        )
    # sort by energy
    sorted_indices = np.argsort(average_energies)
    atoms = [atoms[i] for i in sorted_indices]
    average_energies = average_energies[sorted_indices]

    stratified_average_energies = pd.qcut(average_energies, q=2, labels=False)
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    for train_index, test_index in split.split(atoms, stratified_average_energies):
        train_structures = [atoms[i] for i in train_index]
        test_structures = [atoms[i] for i in test_index]

    if atom_isolated_and_dimer:
        train_structures = atom_isolated_and_dimer + train_structures

    return train_structures, test_structures


def create_soap_descriptor(
    soap_paras: dict[str, int | float | str], n_species: int, species_Z: str
) -> str:
    """
    Generate a SOAP descriptor string based on the given parameters.

    Parameters
    ----------
    soap_paras:
        A dictionary containing SOAP parameters.
    n_species:
        The number of species.
    species_Z:
        A string representing species Z.
    """
    return (
        "soap l_max="
        + str(soap_paras["l_max"])
        + " n_max="
        + str(soap_paras["n_max"])
        + " atom_sigma="
        + str(soap_paras["atom_sigma"])
        + " cutoff="
        + str(soap_paras["cutoff"])
        + " n_species="
        + str(n_species)
        + " species_Z="
        + species_Z
        + " cutoff_transition_width="
        + str(soap_paras["cutoff_transition_width"])
        + " average="
        + str(soap_paras["average"])
    )


def flatten_list(input_list: list | list[list]) -> list:
    """Flatten a nested list into a single list if necessary."""
    if (
        isinstance(input_list, list)
        and len(input_list) > 0
        and isinstance(input_list[0], list)
    ):
        return list(chain.from_iterable(input_list))

    return input_list


def handle_rss_trajectory(
    traj_path, remove_traj_files
) -> tuple[list[list], list[list]]:
    """
    Handle trajectory and associated information.

    Parameters
    ----------
    traj_path: list | None
        List of dictionaries containing trajectory information.
        Each dictionary should have keys 'traj_path' and 'pressure'.
        If None, an empty list will be used.
    remove_traj_files: bool
        Whether to remove the directories containing trajectory files
        after processing them. Default is False.

    Returns
    -------
    tuple:
        atoms: list
            List of ASE Atoms objects read from the trajectory files.
        pressures: list
            List of pressure values corresponding to the atoms.
    """
    atoms = []
    pressures = []
    traj_path = [] if traj_path is None else flatten_list(traj_path)
    traj_dirs = []

    if all(i is None for i in traj_path):
        raise ValueError("No valid trajectory path was obtained!")

    for traj in traj_path:
        if traj is not None and Path(traj).exists():
            print("Processing trajectory:", traj)
            at = ase.io.read(traj, index=":")
            atoms.append(at)
            pressure = [i.info["RSS_applied_pressure"] for i in at]
            pressures.append(pressure)
            traj_dirs.append(os.path.dirname(traj))

    if remove_traj_files and traj_dirs:
        traj_dirs = list(set(traj_dirs))
        for dir_path in traj_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for root, _, files in os.walk(dir_path):
                    for name in files:
                        if "RSS_relax_results" in name:
                            file_path = os.path.join(root, name)
                            os.remove(file_path)

    return atoms, pressures
