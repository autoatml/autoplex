"""RSS Jobs include the generation of the initial potential model as well as iterative RSS exploration."""

import logging
from typing import Literal

from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from jobflow import Flow, Response, job

from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.data.common.jobs import (
    collect_dft_data,
    preprocess_data,
    sample_data,
)
from autoplex.data.rss.flows import BuildMultiRandomizedStructure
from autoplex.data.rss.jobs import do_rss_multi_node
from autoplex.fitting.common.flows import MLIPFitMaker
from autoplex.misc.castep.jobs import CastepStaticMaker

__all__ = ["do_rss_iterations", "initial_rss"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_DEFAULT_STATIC_ENERGY_MAKER = StaticMaker(
    input_set_generator=StaticSetGenerator(
        user_incar_settings={
            "ADDGRID": "True",
            "ENCUT": 520,
            "EDIFF": 1e-06,
            "ISMEAR": 0,
            "SIGMA": 0.01,
            "PREC": "Accurate",
            "ISYM": None,
            "KSPACING": 0.2,
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
            "NELM": 100,
            "LMAXMIX": None,
            "LASPH": None,
            "AMIN": None,
        }
    ),
    run_vasp_kwargs={"handlers": ()},
)


@job
def initial_rss(
    tag: str,
    generated_struct_numbers: list[int],
    num_of_initial_selected_structs: list[int] | None = None,
    cell_seed_paths: list[str] | None = None,
    buildcell_options: list[dict] | None = None,
    fragment_file: str | None = None,
    fragment_numbers: list[str] | None = None,
    num_processes_buildcell: int = 1,
    initial_selection_enabled: bool = False,
    bcur_params: dict | None = None,
    random_seed: int | None = None,
    include_isolated_atom: bool = False,
    isolatedatom_box: list[float] | None = None,
    e0_spin: bool = False,
    include_dimer: bool = False,
    dimer_box: list[float] | None = None,
    dimer_range: list | None = None,
    dimer_num: int = 21,
    custom_incar: dict | None = None,
    custom_potcar: dict | None = None,
    config_type: str | None = None,
    dft_ref_file: str = "dft_ref.extxyz",
    rss_group: str = "initial",
    test_ratio: float = 0.1,
    regularization: bool = False,
    retain_existing_sigma: bool = False,
    scheme: str | None = None,
    element_order: list | None = None,
    reg_minmax: list[tuple] | None = None,
    distillation: bool = False,
    force_max: float | None = None,
    force_label: str = "REF_forces",
    pre_database_dir: str | None = None,
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"] = "GAP",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    auto_delta: bool = False,
    num_processes_fit: int = 1,
    device_for_fitting: str = "cpu",
    static_energy_maker: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
    ) = _DEFAULT_STATIC_ENERGY_MAKER,
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker | None
    ) = None,
    **fit_kwargs,
):
    """
    Run initial Random Structure Searching (RSS) workflow from scratch.

    Parameters
    ----------
    tag: str
        Tag of systems. It can also be used for setting up elements and stoichiometry.
        For example, the tag of 'SiO2' will be recognized as a 1:2 ratio of Si to O and
        passed into the parameters of buildcell. However, note that this will be overwritten
        if the stoichiometric ratio of elements is defined in the 'cell_seed_paths' or 'buildcell_options'.
    generated_struct_numbers: list[int]
        Expected number of generated randomized unit cells.
    num_of_initial_selected_structs: list[int] | None
        Number of structures to be sampled. Default is None.
    cell_seed_paths: list[str]
        A list of paths to the custom buildcell control files, which ends with '.cell'. If these files exist,
        the buildcell_options argument will no longer take effect.
    buildcell_options: list[dict] | None
        Customized parameters for buildcell. Default is None.
    fragment_file: Atoms | list[Atoms] | None
        Fragment(s) for random structures, e.g. molecules, to be placed indivudally intact.
        atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
        atoms.cell must be defined (e.g. Atoms.cell = np.eye(3)*20).
    fragment_numbers: list[str] | None
        Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
    num_processes_buildcell: int
        Number of processes to use for parallel computation during buildcell generation. Default is 1.
    initial_selection_enabled: bool
        If true, sample structures using CUR. Default is False.
    bcur_params: dict | None
        Parameters for Boltzmann CUR selection. Default is None.
    random_seed: int | None
        A seed to ensure reproducibility of CUR selection. Default is None.
    include_isolated_atom: bool
        If true, perform single-point calculations for isolated atoms. Default is False.
    isolatedatom_box: list[float] | None
        List of the lattice constants for an isolated atom configuration. Default is None.
    e0_spin: bool
        If true, include spin polarization in isolated atom and dimer calculations. Default is False.
    include_dimer: bool
        If true, perform single-point calculations for dimers. Default is False.
    dimer_box: list[float] | None
        The lattice constants of a dimer box. Default is None.
    dimer_range: list[float] | None
        Range of distances for dimer calculations. Default is None.
    dimer_num: int
        Number of different distances to consider for dimer calculations. Default is 21.
    custom_incar: dict | None
        Dictionary of custom VASP input parameters. If provided, will update the
        default parameters. Default is None.
    custom_potcar: dict | None
        Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
        Default is None.
    config_type: str | None
        Configuration type for the DFT calculations. Default is None.
    dft_ref_file: str
        Reference file for DFT-labelled data. Default is 'dft_ref.extxyz'.
    rss_group: str
        Group name for GAP RSS. Default is 'initial'.
    test_ratio: float
        The proportion of the test set after splitting the data.
        If None, no splitting will be performed. Default is 0.1.
    regularization: bool
        If true, apply regularization. This only works for GAP. Default is False.
    retain_existing_sigma: bool
        Whether to keep the current sigma values for specific configuration types.
        If set to True, existing sigma values for specific configurations will remain unchanged.
    scheme: str | None
        Scheme to use for regularization. Default is None.
    element_order:
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2).
        This value is useful when constructing high-dimensional convex hulls based on the
        "volume-stoichiometry" scheme. Specially, if the dataset contains compounds with
        different numbers of constituent elements (e.g., both binary and ternary structures),
        this value must be explicitly set to ensure the convex hull is constructed consistently.
    reg_minmax: list[tuple] | None
        A list of tuples representing the minimum and maximum values for regularization.
    distillation: bool
        If true, apply data distillation. Default is False.
    force_max: float | None
        Maximum force value to exclude structures. Default is None.
    force_label: str
        The label of force values to use for distillation. Default is 'REF_forces'.
    pre_database_dir: str | None
        Directory where the previous database was saved. Default is None.
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted. Default is 'GAP'.
    ref_energy_name: str
        Reference energy name. Default is 'REF_energy'.
    ref_force_name: str
        Reference force name. Default is 'REF_forces'.
    ref_virial_name: str
        Reference virial name. Default is 'REF_virial'.
    auto_delta: bool
        If true, apply automatic determination of delta for GAP terms. Default is False.
    num_processes_fit: int
        Number of processes used for fitting. Default is 1.
    device_for_fitting: str
        Device to be used for model fitting, either "cpu" or "cuda".
    static_energy_maker: BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker for static energy jobs: either BaseVaspMaker (VASP-based) or CastepStaticMaker
        (CASTEP-based) or ForceFieldStaticMaker (force field-based). Defaults to StaticMaker (VASP-based).
    static_energy_maker_isolated_atoms: BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker for static energy jobs of isolated atoms: either BaseVaspMaker (VASP-based) or CastepStaticMaker
        (CASTEP-based) or ForceFieldStaticMaker (force field-based) or None. If set to `None`, the parameters
        from `static_energy_maker` will be used as the default for isolated atoms. In this case,
        if `static_energy_maker` is a `StaticMaker`, all major settings will be inherited,
        except that `kspacing` will be automatically set to 100 to enforce a Gamma-point-only calculation.
        This is typically suitable for single-atom systems. Default is None. If a non-`StaticMaker` maker
        is used here, its output must include a `dir_name` field to ensure compatibility with downstream workflows.
    fit_kwargs:
        Additional keyword arguments for the MLIP fitting process.

    Returns
    -------
    dict:
        A dictionary with following information

        - 'test_error': float, The test error of the fitted MLIP.
        - 'pre_database_dir': str, The directory of the preprocessed database.
        - 'mlip_path': Path to the fitted MLIP.
        - 'isolated_atom_energies': dict, The isolated energy values.
        - 'current_iter': int, The current iteration index, set to 0.
    """
    if isolatedatom_box is None:
        isolatedatom_box = [20.0, 20.0, 20.0]
    if dimer_box is None:
        dimer_box = [20.0, 20.0, 20.0]

    do_randomized_structure_generation = BuildMultiRandomizedStructure(
        generated_struct_numbers=generated_struct_numbers,
        cell_seed_paths=cell_seed_paths,
        buildcell_options=buildcell_options,
        fragment_file=fragment_file,
        fragment_numbers=fragment_numbers,
        selected_struct_numbers=num_of_initial_selected_structs,
        tag=tag,
        num_processes=num_processes_buildcell,
        initial_selection_enabled=initial_selection_enabled,
        bcur_params=bcur_params,
        random_seed=random_seed,
    ).make()

    # TODO: this needs to be generalized beyond VASP and instead be able to use a different dft calculator,
    # or a force field
    do_dft_static = DFTStaticLabelling(
        e0_spin=e0_spin,
        isolatedatom_box=isolatedatom_box,
        isolated_atom=include_isolated_atom,
        dimer=include_dimer,
        dimer_box=dimer_box,
        dimer_range=dimer_range,
        dimer_num=dimer_num,
        custom_incar=custom_incar,
        custom_potcar=custom_potcar,
        static_energy_maker=static_energy_maker,
        static_energy_maker_isolated_atoms=static_energy_maker_isolated_atoms,
    ).make(
        structures=do_randomized_structure_generation.output, config_type=config_type
    )
    do_data_collection = collect_dft_data(
        dft_ref_file=dft_ref_file, rss_group=rss_group, dft_dirs=do_dft_static.output
    )
    do_data_preprocessing = preprocess_data(
        test_ratio=test_ratio,
        regularization=regularization,
        retain_existing_sigma=retain_existing_sigma,
        scheme=scheme,
        element_order=element_order,
        distillation=distillation,
        force_max=force_max,
        force_label=force_label,
        dft_ref_dir=do_data_collection.output["dft_ref_dir"],
        pre_database_dir=pre_database_dir,
        reg_minmax=reg_minmax,
        isolated_atom_energies=do_data_collection.output["isolated_atom_energies"],
    )
    do_mlip_fit = MLIPFitMaker(
        mlip_type=mlip_type,
        ref_energy_name=ref_energy_name,
        ref_force_name=ref_force_name,
        ref_virial_name=ref_virial_name,
        num_processes_fit=num_processes_fit,
        apply_data_preprocessing=False,
        auto_delta=auto_delta,
        glue_xml=False,
    ).make(
        isolated_atom_energies=do_data_collection.output["isolated_atom_energies"],
        database_dir=do_data_preprocessing.output,
        device=device_for_fitting,
        **fit_kwargs,
    )

    job_list = [
        do_randomized_structure_generation,
        do_dft_static,
        do_data_collection,
        do_data_preprocessing,
        do_mlip_fit,
    ]

    return Response(
        replace=Flow(job_list),
        output={
            "test_error": do_mlip_fit.output["test_error"],
            "pre_database_dir": do_data_preprocessing.output,
            "mlip_path": do_mlip_fit.output["mlip_path"][0],
            "isolated_atom_energies": do_data_collection.output[
                "isolated_atom_energies"
            ],
        },
    )


@job
def do_rss_iterations(
    input: dict,
    tag: str,
    generated_struct_numbers: list[int],
    num_of_initial_selected_structs: list[int] | None = None,
    cell_seed_paths: list[str] | None = None,
    buildcell_options: list[dict] | None = None,
    fragment_file: str | None = None,
    fragment_numbers: list[str] | None = None,
    num_processes_buildcell: int = 1,
    initial_selection_enabled: bool = False,
    rss_selection_method: str = None,
    num_of_rss_selected_structs: int = 100,
    bcur_params: dict | None = None,
    random_seed: int | None = None,
    include_isolated_atom: bool = False,
    isolatedatom_box: list[float] | None = None,
    e0_spin: bool = False,
    include_dimer: bool = False,
    dimer_box: list[float] | None = None,
    dimer_range: list | None = None,
    dimer_num: int = 21,
    custom_incar: dict | None = None,
    custom_potcar: dict | None = None,
    config_types: list[str] | None = None,
    dft_ref_file: str = "dft_ref.extxyz",
    rss_group: str = "rss",
    test_ratio: float = 0.1,
    regularization: bool = False,
    retain_existing_sigma: bool = False,
    scheme: str | None = None,
    element_order: list | None = None,
    reg_minmax: list[tuple] | None = None,
    distillation: bool = True,
    force_max: float = 200,
    force_label: str = "REF_forces",
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"] = "GAP",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    auto_delta: bool = False,
    num_processes_fit: int = 1,
    device_for_fitting: str = "cpu",
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 200,
    force_tol: float = 0.05,
    stress_tol: float = 0.05,
    hookean_repul: bool = False,
    hookean_paras: dict[tuple[int, int], tuple[float, float]] | None = None,
    keep_symmetry: bool = False,
    remove_traj_files: bool = False,
    num_processes_rss: int = 1,
    device_for_rss: str = "cpu",
    stop_criterion: float = 0.01,
    max_iteration_number: int = 5,
    num_groups: int = 1,
    initial_kt: float = 0.3,
    current_iter_index: int = 1,
    static_energy_maker: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
    ) = _DEFAULT_STATIC_ENERGY_MAKER,
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker | None
    ) = None,
    **fit_kwargs,
):
    """
    Perform iterative RSS to improve the accuracy of a MLIP.

    Each iteration involves generating new structures, sampling, running
    DFT calculations, collecting data, preprocessing data, and fitting a new MLIP.

    Parameters
    ----------
    input : dict
        A dictionary parameter used to pass specific input data required during the RSS iterations.
        The keys in this dictionary should be one of the following valid keys:

            test_error: float
                The test error of the fitted MLIP.
            pre_database_dir: str
                The directory of the preprocessed database.
            mlip_path: str | path
                Path to the fitted MLIP.
            isolated_atom_energies: dict
                The isolated energy values.
            current_iter: int
                The current iteration index.
            kt: float
                The value of kt.

    tag: str
        Tag of systems. It can also be used for setting up elements and stoichiometry.
        For example, the tag of 'SiO2' will be recognized as a 1:2 ratio of Si to O and
        passed into the parameters of buildcell. However, note that this will be overwritten
        if the stoichiometric ratio of elements is defined in the 'cell_seed_paths' or 'buildcell_options'.
    generated_struct_numbers: list[int]
        Expected number of generated randomized unit cells.
    num_of_initial_selected_structs: list[int] | None
        Number of structures to be sampled. Default is None.
    cell_seed_paths: list[str]
        A list of paths to the custom buildcell control files, which ends with '.cell'. If these files exist,
        the buildcell_options argument will no longer take effect.
    buildcell_options: list[dict] | None
        Customized parameters for buildcell. Default is None.
    fragment_file: Atoms | list[Atoms] | None
        Fragment(s) for random structures, e.g. molecules, to be placed indivudally intact.
        atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
        atoms.cell must be defined (e.g. Atoms.cell = np.eye(3)*20).
    fragment_numbers: list[str] | None
        Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
    num_processes_buildcell: int
        Number of processes to use for parallel computation during buildcell generation. Default is 1.
    initial_selection_enabled: bool
        If true, sample structures using CUR. Default is False.
    rss_selection_method: str
        Method for selecting samples from the generated structures. Default is None.
    num_of_rss_selected_structs: int
        Number of structures to be selected.
    bcur_params: dict | None
        Parameters for Boltzmann CUR selection. Default is None.
    random_seed: int | None
        A seed to ensure reproducibility of CUR selection. Default is None.
    include_isolated_atom: bool
        If true, perform single-point calculations for isolated atoms. Default is False.
    isolatedatom_box: list[float] | None
        List of the lattice constants for an isolated atom configuration. Default is None.
    e0_spin: bool
        If true, include spin polarization in isolated atom and dimer calculations. Default is False.
    include_dimer: bool
        If true, perform single-point calculations for dimers only once. Default is False.
    dimer_box: list[float] | None
        The lattice constants of a dimer box. Default is None.
    dimer_range: list[float] | None
        Range of distances for dimer calculations. Default is None.
    dimer_num: int
        Number of different distances to consider for dimer calculations. Default is 21.
    custom_incar: dict | None
        Dictionary of custom VASP input parameters. If provided, will update the
        default parameters. Default is None.
    custom_potcar: dict | None
        Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
        Default is None.
    config_types: list[str] | None
        Configuration types for the DFT calculations. Default is None.
    dft_ref_file: str
        Reference file for DFT-labelled data. Default is 'dft_ref.extxyz'.
    rss_group: str
        Group name for GAP RSS. Default is 'rss'.
    test_ratio: float
        The proportion of the test set after splitting the data. Default is 0.1.
    regularization: bool
        If true, apply regularization. This only works for GAP. Default is False.
    retain_existing_sigma: bool
        Whether to keep the current sigma values for specific configuration types.
        If set to True, existing sigma values for specific configurations will remain unchanged.
    scheme: str | None
        Scheme to use for regularization. Default is None.
    element_order:
        List of atomic numbers in order of choice (e.g. [42, 16] for MoS2).
        This value is useful when constructing high-dimensional convex hulls based on the
        "volume-stoichiometry" scheme. Specially, if the dataset contains compounds with
        different numbers of constituent elements (e.g., both binary and ternary structures),
        this value must be explicitly set to ensure the convex hull is constructed consistently.
    reg_minmax: list[tuple] | None
        A list of tuples representing the minimum and maximum values for regularization.
    distillation: bool
        If true, apply data distillation. Default is True.
    force_max: float
        Maximum force value to exclude structures. Default is 200.
    force_label: str
        The label of force values to use for distillation. Default is 'REF_forces'.
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted. Default is 'GAP'.
    ref_energy_name: str
        Reference energy name. Default is 'REF_energy'.
    ref_force_name: str
        Reference force name. Default is 'REF_forces'.
    ref_virial_name: str
        Reference virial name. Default is 'REF_virial'.
    auto_delta: bool
        If true, apply automatic determination of delta for GAP terms. Default is False.
    num_processes_fit: int
        Number of processes used for fitting. Default is 1.
    device_for_fitting: str
        Device to be used for model fitting, either "cpu" or "cuda".
    scalar_pressure_method: str
        Method for adding external pressures. Default is 'exp'.
    scalar_exp_pressure: float
        Scalar exponential pressure. Default is 100.
    scalar_pressure_exponential_width: float
        Width for scalar pressure exponential. Default is 0.2.
    scalar_pressure_low: float
        Low limit for scalar pressure. Default is 0.
    scalar_pressure_high: float
        High limit for scalar pressure. Default is 50.
    max_steps: int
        Maximum number of steps for relaxation. Default is 200.
    force_tol: float
        Force residual tolerance for relaxation. Default is 0.05.
    stress_tol: float
        Stress residual tolerance for relaxation. Default is 0.05.
    hookean_repul: bool
        If true, apply Hookean repulsion. Default is False.
    hookean_paras: dict[tuple[int, int], tuple[float, float]] | None
        Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
    keep_symmetry: bool
        If true, preserve symmetry during relaxation. Default is False.
    remove_traj_files: bool
            If true, remove all trajectory files raised by RSS to save memory. Default is False.
    num_processes_rss: int
        Number of processes used for running RSS. Default is 1.
    device_for_rss: str
        Specify device to use "cuda" or "cpu" for running RSS. Default is "cpu".
    stop_criterion: float
        Convergence criterion for stopping RSS iterations. Default is 0.01.
    max_iteration_number: int
        Maximum number of RSS iterations to perform. Default is 5.
    num_groups: int
        Number of structure groups, used for assigning tasks across multiple nodes. Default is 1.
    initial_kt: float
        Initial temperature (in eV) for Boltzmann sampling. Default is 0.3.
    current_iter_index: int
        Index for the current RSS iteration. Default is 1.
    static_energy_maker: BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker for static energy jobs: either BaseVaspMaker (VASP-based) or CastepStaticMaker
        (CASTEP-based) or ForceFieldStaticMaker (force field-based). Defaults to StaticMaker (VASP-based).
    static_energy_maker_isolated_atoms: BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker for static energy jobs of isolated atoms: either BaseVaspMaker (VASP-based) or CastepStaticMaker
        (CASTEP-based) or ForceFieldStaticMaker (force field-based) or None. If set to `None`, the parameters
        from `static_energy_maker` will be used as the default for isolated atoms. In this case,
        if `static_energy_maker` is a `StaticMaker`, all major settings will be inherited,
        except that `kspacing` will be automatically set to 100 to enforce a Gamma-point-only calculation.
        This is typically suitable for single-atom systems. Default is None. If a non-`StaticMaker` maker
        is used here, its output must include a `dir_name` field to ensure compatibility with downstream workflows.
    fit_kwargs:
        Additional keyword arguments for the MLIP fitting process.

    Returns
    -------
    dict:
        A dictionary with following information

        - 'test_error': float, The test error of the fitted MLIP.
        - 'pre_database_dir': str, The directory of the preprocessed database.
        - 'mlip_path': Path to the fitted MLIP.
        - 'isolated_atom_energies': dict, The isolated energy values.
        - 'current_iter': int, The current iteration index.
        - 'kt': float, The temperature (in eV) for Boltzmann sampling.
    """
    test_error = input.get("test_error")
    current_iter = input.get("current_iter", current_iter_index)
    current_kt = input.get("kt", initial_kt)

    config_type = (
        (config_types[0] if current_kt > 0.1 else config_types[-1])
        if config_types
        else None
    )

    if isolatedatom_box is None:
        isolatedatom_box = [20.0, 20.0, 20.0]
    if dimer_box is None:
        dimer_box = [20.0, 20.0, 20.0]

    logging.info(
        f"The configuration type of structures generated in the current iteration will be {config_type}!"
    )

    if (
        test_error is not None
        and test_error > stop_criterion
        and current_iter is not None
        and current_iter < max_iteration_number
    ):
        logging.info(f"Current kt: {current_kt}")
        logging.info(f"Current iter index: {current_iter}")
        logging.info(f"The error of {current_iter}th iteration: {test_error}")

        if bcur_params is None:
            bcur_params = {}
        bcur_params["kt"] = current_kt

        do_randomized_structure_generation = BuildMultiRandomizedStructure(
            generated_struct_numbers=generated_struct_numbers,
            cell_seed_paths=cell_seed_paths,
            buildcell_options=buildcell_options,
            fragment_file=fragment_file,
            fragment_numbers=fragment_numbers,
            selected_struct_numbers=num_of_initial_selected_structs,
            tag=tag,
            num_processes=num_processes_buildcell,
            initial_selection_enabled=initial_selection_enabled,
            bcur_params=bcur_params,
            random_seed=random_seed,
        ).make()
        do_rss = do_rss_multi_node(
            mlip_type=mlip_type,
            iteration_index=f"{current_iter}th",
            mlip_path=input["mlip_path"],
            structure_paths=do_randomized_structure_generation.output,
            scalar_pressure_method=scalar_pressure_method,
            scalar_exp_pressure=scalar_exp_pressure,
            scalar_pressure_exponential_width=scalar_pressure_exponential_width,
            scalar_pressure_low=scalar_pressure_low,
            scalar_pressure_high=scalar_pressure_high,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            hookean_repul=hookean_repul,
            hookean_paras=hookean_paras,
            keep_symmetry=keep_symmetry,
            num_processes_rss=num_processes_rss,
            device=device_for_rss,
            num_groups=num_groups,
            config_type=config_type,
        )
        do_data_sampling = sample_data(
            selection_method=rss_selection_method,
            num_of_selection=num_of_rss_selected_structs,
            bcur_params=bcur_params,
            traj_path=do_rss.output,
            random_seed=random_seed,
            isolated_atom_energies=input["isolated_atom_energies"],
            remove_traj_files=remove_traj_files,
        )
        do_dft_static = DFTStaticLabelling(
            e0_spin=e0_spin,
            isolatedatom_box=isolatedatom_box,
            isolated_atom=include_isolated_atom,
            dimer=include_dimer,
            dimer_box=dimer_box,
            dimer_range=dimer_range,
            dimer_num=dimer_num,
            custom_incar=custom_incar,
            custom_potcar=custom_potcar,
            static_energy_maker=static_energy_maker,
            static_energy_maker_isolated_atoms=static_energy_maker_isolated_atoms,
        ).make(structures=do_data_sampling.output, config_type=config_type)
        do_data_collection = collect_dft_data(
            dft_ref_file=dft_ref_file,
            rss_group=rss_group,
            dft_dirs=do_dft_static.output,
        )
        do_data_preprocessing = preprocess_data(
            test_ratio=test_ratio,
            regularization=regularization,
            retain_existing_sigma=retain_existing_sigma,
            scheme=scheme,
            element_order=element_order,
            distillation=distillation,
            force_max=force_max,
            force_label=force_label,
            dft_ref_dir=do_data_collection.output["dft_ref_dir"],
            pre_database_dir=input["pre_database_dir"],
            reg_minmax=reg_minmax,
            isolated_atom_energies=input["isolated_atom_energies"],
        )
        do_mlip_fit = MLIPFitMaker(
            mlip_type=mlip_type,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            num_processes_fit=num_processes_fit,
            apply_data_preprocessing=False,
            auto_delta=auto_delta,
            glue_xml=False,
        ).make(
            database_dir=do_data_preprocessing.output,
            isolated_atom_energies=input["isolated_atom_energies"],
            device=device_for_fitting,
            **fit_kwargs,
        )

        kt = current_kt - 0.1 if (current_kt - 0.1) > 0.1 else 0.1
        current_iter += 1
        if include_isolated_atom:
            include_isolated_atom = False
        if include_dimer:
            include_dimer = False

        do_iteration = do_rss_iterations(
            input={
                "test_error": do_mlip_fit.output["test_error"],
                "pre_database_dir": do_data_preprocessing.output,
                "mlip_path": do_mlip_fit.output["mlip_path"][0],
                "isolated_atom_energies": input["isolated_atom_energies"],
                "current_iter": current_iter,
                "kt": kt,
            },
            generated_struct_numbers=generated_struct_numbers,
            num_of_initial_selected_structs=num_of_initial_selected_structs,
            tag=tag,
            cell_seed_paths=cell_seed_paths,
            buildcell_options=buildcell_options,
            fragment_file=fragment_file,
            fragment_numbers=fragment_numbers,
            num_processes_buildcell=num_processes_buildcell,
            initial_selection_enabled=initial_selection_enabled,
            rss_selection_method=rss_selection_method,
            num_of_rss_selected_structs=num_of_rss_selected_structs,
            bcur_params=bcur_params,
            random_seed=random_seed,
            e0_spin=e0_spin,
            isolatedatom_box=isolatedatom_box,
            include_isolated_atom=include_isolated_atom,
            include_dimer=include_dimer,
            dimer_box=dimer_box,
            dimer_range=dimer_range,
            dimer_num=dimer_num,
            custom_incar=custom_incar,
            custom_potcar=custom_potcar,
            config_types=config_types,
            dft_ref_file=dft_ref_file,
            rss_group=rss_group,
            test_ratio=test_ratio,
            regularization=regularization,
            retain_existing_sigma=retain_existing_sigma,
            scheme=scheme,
            element_order=element_order,
            reg_minmax=reg_minmax,
            distillation=distillation,
            force_max=force_max,
            force_label=force_label,
            mlip_type=mlip_type,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            auto_delta=auto_delta,
            num_processes_fit=num_processes_fit,
            scalar_pressure_method=scalar_pressure_method,
            scalar_exp_pressure=scalar_exp_pressure,
            scalar_pressure_exponential_width=scalar_pressure_exponential_width,
            scalar_pressure_low=scalar_pressure_low,
            scalar_pressure_high=scalar_pressure_high,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            hookean_repul=hookean_repul,
            hookean_paras=hookean_paras,
            keep_symmetry=keep_symmetry,
            remove_traj_files=remove_traj_files,
            num_processes_rss=num_processes_rss,
            device_for_rss=device_for_rss,
            stop_criterion=stop_criterion,
            max_iteration_number=max_iteration_number,
            num_groups=num_groups,
            initial_kt=initial_kt,
            current_iter_index=current_iter_index,
            static_energy_maker=static_energy_maker,
            static_energy_maker_isolated_atoms=static_energy_maker_isolated_atoms,
            **fit_kwargs,
        )

        job_list = [
            do_randomized_structure_generation,
            do_rss,
            do_data_sampling,
            do_dft_static,
            do_data_collection,
            do_data_preprocessing,
            do_mlip_fit,
            do_iteration,
        ]

        return Response(detour=job_list, output=do_iteration.output)

    return input
