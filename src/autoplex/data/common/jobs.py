"""Jobs to create training data for ML potentials."""

import logging
import os
import pickle
import shutil
import traceback
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from atomate2.utils.path import strip_hostname
from emmet.core.math import Matrix3D
from jobflow.core.job import job
from phonopy.structure.cells import get_supercell
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp.outputs import Vasprun

from autoplex.data.common.utils import (
    ElementCollection,
    boltzhist_cur_dual_iter,
    boltzhist_cur_one_shot,
    create_soap_descriptor,
    cur_select,
    data_distillation,
    flatten,
    handle_rss_trajectory,
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle,
    stratified_dataset_split,
    to_ase_trajectory,
)
from autoplex.fitting.common.regularization import set_custom_sigma

if TYPE_CHECKING:
    from ase import Atoms

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@job
def convert_to_extxyz(job_output, pkl_file, config_type, factor):
    """
    Convert data and write extxyt file.

    Parameters
    ----------
    job_output:
        The (static) job output object.
    pkl_file: str
        A pickle file.
    config_type: str
        Configuration type of the data.
    factor: str
        String of a factor to resize cell parameters.

    """
    with open(Path(job_output.dir_name) / Path(pkl_file), "rb") as file:
        traj_obj = pickle.load(file)
    data = to_ase_trajectory(traj_obj=traj_obj)
    data[-1].write("tmp.xyz")
    file = read("tmp.xyz", index=":")
    for i in file:
        # TODO: enable switching to stress
        virial_list = -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
        i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
        del i.calc.results["stress"]
        i.arrays["REF_forces"] = i.calc.results["forces"]
        del i.calc.results["forces"]
        i.info["REF_energy"] = i.calc.results["energy"]
        del i.calc.results["energy"]
        i.info["config_type"] = config_type
        i.pbc = True
    write("ref_" + factor + ".extxyz", file, append=True)

    return os.getcwd()


@job
def plot_force_distribution(
    cell_factor: float,
    path: str,
    x_min: int = 0,
    x_max: int = 5,
    bin_width: float = 0.125,
):
    """
    Plotter for the force distribution.

    Parameters
    ----------
    cell_factor: float
        Factor to resize cell parameters.
    path:
        Path to the ref_XYZ.extxyz file.
    x_min: int
        Minimum value for the plot x-axis.
    x_max: int
        Maximum value for the plot x-axis.
    bin_width: float
        Width of the plot bins.

    """
    plt.xlabel("Forces")
    plt.ylabel("Count")
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    plot_total = []

    # TODO split data collection and plotting

    plot_data = []
    with open(path + "/ref_" + str(cell_factor).replace(".", "") + ".extxyz") as file:
        for line in file:
            # Split the line into columns
            columns = line.split()

            # Check if the line has exactly 10 columns
            if len(columns) == 10:
                # Extract the last three columns
                data = columns[-3:]
                norm_data = np.linalg.norm(data, axis=-1)
                plot_data.append(norm_data)

        plt.hist(plot_data, bins=bins, edgecolor="black")
        plt.title(f"Data for factor {cell_factor}")

        plt.savefig("data_factor_" + str(cell_factor).replace(".", "") + ".png")

        plot_total += plot_data
    plt.hist(plot_total, bins=bins, edgecolor="black")
    plt.title("Data")

    plt.savefig("total_data.png")


@job
def get_supercell_job(structure: Structure, supercell_matrix: Matrix3D):
    """
    Create a job to get the supercell.

    Parameters
    ----------
    structure: Structure
        The pymatgen structure object.
    supercell_matrix: Matrix3D
        The matrix to generate the supercell.

    Returns
    -------
    supercell: Structure
        A pymatgen structure object.

    """
    supercell = get_supercell(
        unitcell=get_phonopy_structure(structure), supercell_matrix=supercell_matrix
    )
    return get_pmg_structure(supercell)


@job(data=[Structure])
def generate_randomized_structures(
    structure: Structure,
    supercell_matrix: Matrix3D | None = None,
    distort_type: int = 0,
    n_structures: int = 10,
    volume_scale_factor_range: list[float] | None = None,
    volume_custom_scale_factors: list[float] | None = None,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_std: float = 0.01,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
):
    """
    Take in a pymatgen Structure object and generates angle/volume distorted + rattled structures.

    Parameters
    ----------
    structure: Structure.
        Pymatgen structures object.
    supercell_matrix: Matrix3D.
        Matrix for obtaining the supercell.
    distort_type: int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures: int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    volume_scale_factor_range: list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume +-10%.
    volume_custom_scale_factors: list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
    min_distance: float
        Minimum separation allowed between any two atoms.
        Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    rattle_type: int.
        0- standard rattling, 1- Monte-Carlo rattling. Default=0.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
        Note that for MC rattling, displacements generated will roughly be
        rattle_mc_n_iter**0.5 * rattle_std for small values of n_iter.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
        Default=42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.

    Returns
    -------
    Response.output.
        Volume or angle-distorted structures with rattled atoms.
    """
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    supercell = get_supercell(
        unitcell=get_phonopy_structure(structure),
        supercell_matrix=supercell_matrix,
    )
    structure = get_pmg_structure(supercell)

    # distort cells by volume or angle
    if distort_type == 0:
        distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
    elif distort_type == 1:
        distorted_cells = random_vary_angle(
            structure=structure,
            min_distance=min_distance,
            angle_percentage_scale=angle_percentage_scale,
            w_angle=w_angle,
            n_structures=n_structures,
            angle_max_attempts=angle_max_attempts,
        )
    elif distort_type == 2:
        initial_distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
        distorted_cells = []
        for cell in initial_distorted_cells:
            distorted_cell = random_vary_angle(
                structure=cell,
                min_distance=min_distance,
                angle_percentage_scale=angle_percentage_scale,
                w_angle=w_angle,
                n_structures=1,
                angle_max_attempts=angle_max_attempts,
            )
            distorted_cells.append(distorted_cell)
        distorted_cells = list(chain.from_iterable(distorted_cells))
    else:
        raise TypeError("distort_type is not recognised")

    # distorted_cells=list(chain.from_iterable(distorted_cells))

    # rattle cells by standard or mc
    rattled_cells = (
        [
            std_rattle(
                structure=cell,
                n_structures=1,
                rattle_std=rattle_std,
                rattle_seed=rattle_seed + icell,
            )
            for icell, cell in enumerate(distorted_cells)
        ]
        if rattle_type == 0
        else (
            [
                mc_rattle(
                    structure=cell,
                    n_structures=1,
                    rattle_std=rattle_std,
                    min_distance=min_distance,
                    rattle_seed=rattle_seed + icell,
                    rattle_mc_n_iter=rattle_mc_n_iter,
                )
                for icell, cell in enumerate(distorted_cells)
            ]
            if rattle_type == 1
            else None
        )
    )

    if rattled_cells is None:
        raise TypeError("rattle_type is not recognized")

    return list(chain.from_iterable(rattled_cells))


@job
def sample_data(
    selection_method: Literal[
        "cur", "bcur1s", "bcur2i", "random", "uniform"
    ] = "random",
    num_of_selection: int = 5,
    bcur_params: dict | None = None,
    dir: list[str] | str | None = None,
    structure: list[Structure] | list[list[Structure]] | None = None,
    traj_path: list | None = None,
    isolated_atom_energies: dict | None = None,
    random_seed: int = None,
    remove_traj_files: bool = False,
) -> list[Structure]:
    """
    Job to sample training configurations from trajectories of MD/RSS.

    Parameters
    ----------
    selection_method : Literal['cur', 'bcur1s', 'bcur2s', 'random', 'uniform']
       Method for selecting samples. Options include:

        - 'cur': Pure CUR selection.

        - 'bcur': Boltzmann flat histogram in enthalpy, then CUR.

            - 'bcur1s': Execute bcur with one shot (1s)
            - 'bcur2i': Execute bcur with two iterations (2i)

        - 'random': Random selection.
        - 'uniform': Uniform selection.

    num_of_selection: int
       Number of structures to be sampled.
    bcur_params: dict
        Parameters for Boltzmann CUR selection. The default dictionary includes:

        - 'soap_paras': SOAP descriptor parameters

            - 'l_max': int, Maximum degree of spherical harmonics (default 12).
            - 'n_max': int, Maximum number of radial basis functions (default 12).
            - 'atom_sigma': float, Width of Gaussian smearing (default 0.0875).
            - 'cutoff': float, Radial cutoff distance (default 10.5).
            - 'cutoff_transition_width': float, Width of the transition region (default 1.0).
            - 'zeta': float, Exponent for dot-product SOAP kernel (default 4.0).
            - 'average': bool, Whether to average the SOAP vectors (default True).
            - 'species': bool, Whether to consider species information (default True).

        - 'kt': float, Temperature in eV for Boltzmann weighting (default 0.3).
        - 'frac_of_bcur': float, Fraction of Boltzmann CUR selections (default 0.8).
        - 'bolt_max_num': int, Maximum number of Boltzmann selections (default 3000).
        - 'kernel_exp': float, Exponent for the kernel (default 4.0).
        - 'energy_label': str, Label for the energy data (default 'energy').

    dir: str
        Directory containing trajectory files for MD/RSS simulations. Default is None.
    structure: list[Structure]
        List of structures for sampling. Default is None.
    traj_path: list[list[str]]
        List of lists containing trajectory paths. Default is None.
    isolated_atom_energies: dict
        Dictionary of isolated energy values for species. Required for 'boltzhist_cur'
        selection method. Default is None.
    random_seed: int, optional
        Seed for random number generation, ensuring reproducibility of sampling.
    remove_traj_files: bool
        Remove all trajectory files raised by RSS to save memory

    Returns
    -------
    list of ase.Atoms
        The selected atoms.
    """
    default_bcur_params = {
        "soap_paras": {
            "l_max": 12,
            "n_max": 12,
            "atom_sigma": 0.0875,
            "cutoff": 10.5,
            "cutoff_transition_width": 1.0,
            "zeta": 4.0,
            "average": True,
            "species": True,
        },
        "kt": 0.3,
        "frac_of_bcur": 0.8,
        "bolt_max_num": 3000,
        "kernel_exp": 4.0,
        "energy_label": "energy",
    }

    if bcur_params is not None:
        default_bcur_params.update(bcur_params)

    bcur_params = default_bcur_params
    pressures = None

    if dir is not None:
        if isinstance(dir, list):
            atoms = [read(i, index=":") for i in dir]
            atoms = flatten(atoms, recursive=True)
        else:
            atoms = read(dir, index=":")

    elif structure is not None:
        if isinstance(structure[0], list):
            structure = flatten(structure, recursive=False)
        atoms = [AseAtomsAdaptor().get_atoms(at) for at in structure]

    else:
        atoms, pressures = handle_rss_trajectory(traj_path, remove_traj_files)

    if selection_method in {"cur", "bcur1s", "bcur2i"}:
        n_species = ElementCollection(
            flatten(atoms, recursive=True)
        ).get_number_of_species()
        species_Z = ElementCollection(flatten(atoms, recursive=True)).get_species_Z()

        if not isinstance(bcur_params["soap_paras"], dict):
            raise TypeError("soap_paras must be a dictionary")
        if not isinstance(bcur_params["kt"], float):
            raise TypeError("kt must be a float")
        if not isinstance(bcur_params["frac_of_bcur"], float):
            raise TypeError("frac_of_bcur must be a float")
        if not isinstance(bcur_params["bolt_max_num"], int):
            raise TypeError("bolt_max_num must be an integer")
        if not isinstance(bcur_params["kernel_exp"], float):
            raise TypeError("kernel_exp must be a float")
        if not isinstance(bcur_params["energy_label"], str):
            raise TypeError("energy_label must be a string")

        soap_paras = bcur_params["soap_paras"]
        descriptor = create_soap_descriptor(soap_paras, n_species, species_Z)

        if selection_method == "cur":
            selected_atoms = cur_select(
                atoms=atoms,
                selected_descriptor=descriptor,
                kernel_exp=bcur_params["kernel_exp"],
                select_nums=num_of_selection,
                stochastic=True,
                random_seed=random_seed,
            )

        elif selection_method in {"bcur1s", "bcur2i"}:
            if isolated_atom_energies is not None:
                isolated_atom_energies = {
                    int(k): v for k, v in isolated_atom_energies.items()
                }
            else:
                raise ValueError("Please provide the energy of isolated atoms!")

            if selection_method == "bcur1s":
                selected_atoms = boltzhist_cur_one_shot(
                    atoms=atoms,
                    isolated_atom_energies=isolated_atom_energies,
                    bolt_frac=bcur_params["frac_of_bcur"],
                    bolt_max_num=bcur_params["bolt_max_num"],
                    cur_num=num_of_selection,
                    kernel_exp=bcur_params["kernel_exp"],
                    kt=bcur_params["kt"],
                    energy_label=bcur_params["energy_label"],
                    pressures=pressures,
                    descriptor=descriptor,
                    random_seed=random_seed,
                )
            else:
                selected_atoms = boltzhist_cur_dual_iter(
                    atoms=atoms,
                    isolated_atom_energies=isolated_atom_energies,
                    bolt_frac=bcur_params["frac_of_bcur"],
                    bolt_max_num=bcur_params["bolt_max_num"],
                    cur_num=num_of_selection,
                    kernel_exp=bcur_params["kernel_exp"],
                    kt=bcur_params["kt"],
                    energy_label=bcur_params["energy_label"],
                    pressures=pressures,
                    descriptor=descriptor,
                    random_seed=random_seed,
                )

        if selected_atoms is None:
            raise ValueError(
                "Unable to sample correctly. Please recheck the parameters!"
            )

        selected_atoms = [AseAtomsAdaptor().get_structure(at) for at in selected_atoms]

    elif selection_method == "random":
        if random_seed is not None:
            np.random.seed(random_seed)

        structure = [AseAtomsAdaptor().get_structure(at) for at in atoms]

        try:
            selection = np.random.choice(
                len(structure), num_of_selection, replace=False
            )
            selected_atoms = [at for i, at in enumerate(structure) if i in selection]

        except ValueError:
            logging.error(
                "The number of selected structures must be less than the total!"
            )
            traceback.print_exc()

    elif selection_method == "uniform":
        try:
            indices = np.linspace(0, len(atoms) - 1, num_of_selection, dtype=int)
            structure = [AseAtomsAdaptor().get_structure(at) for at in atoms]
            selected_atoms = [structure[idx] for idx in indices]

        except ValueError:
            logging.error(
                "The number of selected structures must be less than the total!"
            )
            traceback.print_exc()

    if selected_atoms is None:
        raise ValueError("Unable to sample correctly. Please recheck the parameters!")

    return selected_atoms


@job
def collect_dft_data(
    vasp_ref_file: str = "vasp_ref.extxyz",
    rss_group: str = "RSS",
    vasp_dirs: dict | None = None,
) -> dict:
    """
    Collect VASP data from specified directories.

    Parameters
    ----------
    vasp_ref_file : str
        Reference file for VASP data. Default is 'vasp_ref.extxyz'.
    rss_group : str
        Group name for GAP RSS. Default is 'RSS'.
    vasp_dirs : dict
        Dictionary containing VASP directories and configuration types. Should have keys:

        - 'dirs_of_vasp': list
            List of directories containing VASP data.
        - 'config_type': list
            List of configuration types corresponding to each directory.

    Returns
    -------
    dict:
        A dictionary containing

        - 'vasp_ref_dir': Directory of the VASP reference file.
        - 'isolated_atom_energies': Isolated energy values.
    """
    if vasp_dirs is None:
        raise ValueError(
            "vasp_dirs must be provided and should contain 'dirs_of_vasp' and 'config_type' keys."
        )

    if "dirs_of_vasp" not in vasp_dirs or "config_type" not in vasp_dirs:
        raise ValueError(
            "vasp_dirs must contain 'dirs_of_vasp' and 'config_type' keys."
        )

    dirs = [safe_strip_hostname(value) for value in vasp_dirs["dirs_of_vasp"]]
    config_types = vasp_dirs["config_type"]

    logging.info("Attempting collecting VASP...")

    if dirs is None:
        raise ValueError("dft_dir must be specified if collect_vasp is True")

    atoms = []
    isolated_atom_energies = {}

    for i, val in enumerate(dirs):
        if os.path.exists(os.path.join(val, "vasprun.xml.gz")):

            converged = check_convergence_vasp(os.path.join(val, "vasprun.xml.gz"))

            if converged:
                at = read(os.path.join(val, "vasprun.xml.gz"), index=":")
                for at_i in at:
                    virial_list = (
                        -voigt_6_to_full_3x3_stress(at_i.get_stress())
                        * at_i.get_volume()
                    )
                    at_i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
                    del at_i.calc.results["stress"]
                    at_i.arrays["REF_forces"] = at_i.calc.results["forces"]
                    del at_i.calc.results["forces"]
                    at_i.info["REF_energy"] = at_i.calc.results["free_energy"]
                    del at_i.calc.results["energy"]
                    del at_i.calc.results["free_energy"]
                    atoms.append(at_i)
                    at_i.info["config_type"] = config_types[i]
                    if (
                        at_i.info["config_type"] != "dimer"
                        and at_i.info["config_type"] != "IsolatedAtom"
                    ):
                        at_i.pbc = True
                        at_i.info["rss_group"] = rss_group
                    else:
                        at_i.info["rss_nonperiodic"] = "T"

                    if at_i.info["config_type"] == "IsolatedAtom":
                        at_ids = at_i.get_atomic_numbers()
                        # array_key = at_ids.tostring()
                        isolated_atom_energies[int(at_ids[0])] = at_i.info["REF_energy"]

            else:
                logging.warning(
                    f"Calculation did not converge for path: {os.path.join(val, 'vasprun.xml.gz')}"
                )

    logging.info(f"Total {len(atoms)} structures from VASP are exactly collected.")

    write(vasp_ref_file, atoms, format="extxyz", parallel=False)

    dir_path = Path.cwd()

    vasp_ref_dir = os.path.join(dir_path, vasp_ref_file)

    return {
        "vasp_ref_dir": vasp_ref_dir,
        "isolated_atom_energies": isolated_atom_energies,
    }


def check_convergence_vasp(path: str) -> bool:
    """
    Check if VASP calculation has converged.

    Parameters
    ----------
    path: Path
        Path to the vasp output file to check convergence.

    Return
    ------
    bool
        True if a run is converged both ionically and electronically.
    """
    vasprun = Vasprun(path)
    converged_e = vasprun.converged_electronic
    converged_i = vasprun.converged_ionic

    return converged_e and converged_i


def safe_strip_hostname(value):
    """
    Strip the hostname from a given path or URL.

    Parameters
    ----------
    value: str
        The path or URL from which to strip the hostname.

    Returns
    -------
    Optional[str]
        The path or URL without the hostname if the operation is successful,
        otherwise None.
    """
    try:
        return strip_hostname(value)
    except Exception as e:
        print(f"Error processing '{value}': {e}")
        return None


@job
def preprocess_data(
    vasp_ref_dir: str,
    test_ratio: float | None = None,
    regularization: bool = False,
    retain_existing_sigma: bool = False,
    scheme: str = "linear-hull",
    element_order: list | None = None,
    distillation: bool = False,
    force_max: float = 40,
    force_label: str = "REF_forces",
    energy_label: str = "REF_energy",
    pre_database_dir: str | None = None,
    reg_minmax: list[tuple] | None = None,
    isolated_atom_energies: dict | None = None,
) -> Path:
    """
    Preprocesse data to before fiting machine learning models.

    This function handles tasks such as splitting the dataset,
    applying regularization, accumulating database, and filtering
    structures based on maximum force values.

    Parameters
    ----------
    vasp_ref_dir: str
        Path to the directory containing the reference VASP calculation data.
    test_ratio: float
        The proportion of the test set after splitting the data.
        If None, no splitting will be performed.
    regularization: bool
        If true, apply regularization. This only works for GAP.
    retain_existing_sigma: bool
        Whether to keep the current sigma values for specific configuration types.
        If set to True, existing sigma values for specific configurations will remain unchanged.
    scheme: str
        Scheme to use for regularization.
    element_order:
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2).
        This value is useful when constructing high-dimensional convex hulls based on the
        "volume-stoichiometry" scheme. Specially, if the dataset contains compounds with
        different numbers of constituent elements (e.g., both binary and ternary structures),
        this value must be explicitly set to ensure the convex hull is constructed consistently.
    distillation: bool
        If True, apply data distillation.
    force_max: float
        Maximum force value to exclude structures.
    force_label: str
        The label of force values to use for distillation.
    energy_label: str
        The label of energy values to use for distillation.
    pre_database_dir : str
        Directory where the previous database was saved.
    reg_minmax: list[tuple]
        A list of tuples representing the minimum and maximum
        values for regularization.
    isolated_atom_energies: dict
        A dictionary containing isolated energy values for different species.

    Returns
    -------
    Path
        The current working directory.
    """
    atoms = (
        data_distillation(vasp_ref_dir, force_max, force_label)
        if distillation
        else read(vasp_ref_dir, index=":")
    )

    if test_ratio == 0 or test_ratio is None:
        train_structures, test_structures = atoms, atoms
    else:
        train_structures, test_structures = stratified_dataset_split(
            atoms, test_ratio, energy_label
        )

    if pre_database_dir and os.path.exists(pre_database_dir):
        files_to_copy = ["train.extxyz", "test.extxyz"]
        current_working_directory = os.getcwd()

        for file_name in files_to_copy:
            source_file_path = os.path.join(pre_database_dir, file_name)
            destination_file_path = os.path.join(current_working_directory, file_name)
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, destination_file_path)
                print(f"File {file_name} has been copied to {destination_file_path}")

    write("train.extxyz", train_structures, format="extxyz", append=True)
    write("test.extxyz", test_structures, format="extxyz", append=True)

    if regularization:
        atoms_reg: list[Atoms] = read("train.extxyz", index=":")

        if reg_minmax is None:
            reg_minmax = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]

        atom_with_sigma = set_custom_sigma(
            atoms=atoms_reg,
            reg_minmax=reg_minmax,
            isolated_atom_energies=isolated_atom_energies,
            scheme=scheme,
            element_order=element_order,
            retain_existing_sigma=retain_existing_sigma,
        )

        write("train.extxyz", atom_with_sigma, format="extxyz")

    return Path.cwd()
