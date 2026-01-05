"""MD workflow jobs."""

import logging
from dataclasses import field

from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure

from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.data.common.jobs import (
    collect_dft_data,
    preprocess_data,
    sample_data,
)

try:
    from emmet.core.types.enums import StoreTrajectoryOption
except ImportError:
    from emmet.core.vasp.calculation import StoreTrajectoryOption
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from ase.md.md import MolecularDynamics
from atomate2.ase.md import MDEnsemble
from atomate2.forcefields.utils import MLFF
from emmet.core.math import Matrix3D

from autoplex.data.common.flows import _DEFAULT_STATIC_ENERGY_MAKER
from autoplex.data.md.flows import MDAseMaker
from autoplex.fitting.common.flows import MLIPFitMaker
from autoplex.misc.castep.jobs import CastepStaticMaker

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@job
def do_md_single_step(
    structure: Structure | list[Structure],
    md_solver: Literal["ase", "lammps"] = "ase",  # TODO MDLammpsMaker
    starting_mlip: str | MLFF = MLFF.Forcefield,
    iter_mlip: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"] = "GAP",
    calculator_kwargs: dict | None = None,
    time_step: float | None = None,
    traj_interval: int = 1,
    ensemble: MDEnsemble = MDEnsemble.nvt,
    dynamics: str | MolecularDynamics | None = None,
    pressure: float | Sequence | np.ndarray | None = None,
    temperature_list: list[float] | None = None,
    eqm_step_list: list[int] | None = None,
    rate_list: list[int] | None = None,
    volume_custom_scale_factors: list[float] | None = None,
    supercell_matrix: Matrix3D | None = None,
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.PARTIAL,
    traj_file: str | Path | None = None,
    traj_file_fmt: Literal["pmg", "ase", "xdatcar"] = "ase",
    ionic_step_data: tuple[str, ...] | None = field(
        default=("energy", "forces", "stress", "struct_or_mol")
    ),
    dft_ref_file: str = "dft_md_ref.extxyz",
    include_isolated_atom: bool = False,
    isolatedatom_box: list[float] | None = None,
    e0_spin: bool = False,
    include_dimer: bool = False,
    dimer_box: list[float] | None = None,
    dimer_range: list | None = None,
    dimer_num: int = 21,
    custom_incar: dict | None = None,
    custom_potcar: dict | None = None,
    config_type: str = "md",
    static_energy_maker: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
    ) = _DEFAULT_STATIC_ENERGY_MAKER,
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker | None
    ) = None,
    selection_method: Literal["cur", "random", "uniform"] = "uniform",
    random_seed: int = 42,
    num_of_selection: int = 5,
    remove_traj_files: bool = False,
    isolated_atom_energies: dict | None = None,
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
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    auto_delta: bool = False,
    num_processes_fit: int = 1,
    device_for_fitting: str = "cpu",
    **fit_kwargs,
):
    """
    Run a single MD-driven data-generation step and label selected frames with DFT.

    Parameters
    ----------
    structure : Structure | list[Structure]
        Input pymatgen Structure or list of Structures used as starting geometry.
    md_solver : Literal["ase", "lammps"]
        MD engine to use.
    starting_mlip : str | MLFF
        Initial MLIP or force field identifier or object.
    iter_mlip : Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        MLIP type to fit after sampling when fitting is performed.
    calculator_kwargs : dict
        Keyword arguments passed to the MD calculator, for example MLFF param files.
    time_step : float | None
        MD timestep.
    traj_interval : int
        Save trajectory frames every this many MD steps.
    ensemble : MDEnsemble
        Thermodynamic ensemble, for example MDEnsemble.nvt.
    dynamics : str | MolecularDynamics | None
        Thermostat or ASE MolecularDynamics instance to use.
    pressure : float | Sequence | np.ndarray | None
        External pressure value or schedule. If a sequence, it is interpolated.
    temperature_list : list[float]
        List of temperatures in Kelvin that define a multi-stage schedule.
    eqm_step_list : list[int] | None
        Number of MD steps to hold each temperature stage. Length must match temperature_list.
    rate_list : list[float] | None
        Relative rates controlling interpolation between temperature stages.
    volume_custom_scale_factors : list[float] | None
        Explicit list of volume scale factors to sample.
    supercell_matrix : Matrix3D | None
        3x3 integer matrix to expand the input cell, for example [[2,0,0],[0,2,0],[0,0,2]].
    store_trajectory : StoreTrajectoryOption
        Option controlling trajectory storage.
    traj_file : str | Path | None
        Filename or Path for the trajectory output. If None, no file is written.
    traj_file_fmt : Literal["pmg", "ase", "xdatcar"]
        Trajectory file format.
    ionic_step_data : tuple[str, ...] | None
        Tuple of ionic-step data fields to record per step, such as "energy" or "forces".
    dft_ref_file : str
        Filename for the DFT-labelled extxyz reference file.
    include_isolated_atom : bool
        Whether to include isolated-atom single points in labelling.
    isolatedatom_box : list[float] | None
        Box dimensions for isolated-atom calculations when include_isolated_atom is True.
    e0_spin : bool
        Whether to include spin polarization in isolated atom and dimer calculations.
    include_dimer : bool
        Whether to include dimer single points in labelling.
    dimer_box : list[float] | None
        Box dimensions for dimer calculations.
    dimer_range : list | None
        Distance range for dimer calculations.
    dimer_num : int
        Number of distances to evaluate for dimers.
    custom_incar : dict | None
        Custom DFT INCAR-like settings.
    custom_potcar : dict | None
        Custom POTCAR-like mapping for element pseudopotentials.
    config_type : str
        Tag attached to generated configurations.
    static_energy_maker : BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker used to run static single-point calculations for labelling.
    static_energy_maker_isolated_atoms : BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker | None
        Maker used for isolated-atom single points. If None, static_energy_maker is used.
    selection_method : Literal["cur", "random", "uniform"]
        Method to select structures from trajectories.
    random_seed : int
        RNG seed for reproducible sampling.
    num_of_selection : int
        Number of structures to sample for labelling.
    remove_traj_files : bool
        Whether to remove intermediate trajectory files after sampling.
    isolated_atom_energies : dict | None
        Mapping of element symbols to isolated-atom reference energies to use instead of computing.
    test_ratio : float
        Fraction of data held out for testing during preprocessing or fitting.
    regularization : bool
        Whether to apply regularization during preprocessing.
    retain_existing_sigma : bool
        Preserve existing sigma values when applying regularization.
    scheme : str | None
        Regularization scheme identifier.
    element_order : list | None
        Optional element ordering to enforce for preprocessing.
    reg_minmax : list[tuple] | None
        List of (min, max) tuples for sigma regularization bounds.
    distillation : bool
        Whether to apply data distillation prior to fitting.
    force_max : float | None
        Maximum force threshold to exclude high-force structures.
    force_label : str
        Label key used for forces in the DFT data.
    pre_database_dir : str | None
        Path to an existing preprocessed database to resume from.
    ref_energy_name : str
        Field name used for reference energies in the dataset.
    ref_force_name : str
        Field name used for reference forces in the dataset.
    ref_virial_name : str
        Field name used for reference virials in the dataset.
    auto_delta : bool
        Whether to auto-determine GAP delta hyperparameters where applicable.
    num_processes_fit : int
        Number of processes to use for MLIP fitting.
    device_for_fitting : str
        Device string for fitting, for example "cpu" or "cuda".
    **fit_kwargs : dict
        Additional keyword arguments forwarded to the MLIP fitting maker. These typically include
        model hyperparameters and model-specific options.

    Returns
    -------
    dict
        A dictionary containing:
        - pre_database_dir: Path or output reference to the directory with collected and
          preprocessed DFT data.
        - mlip_path: Path or output reference to the fitted MLIP model (list or single item)
          when fitting is performed; absent or None if no fitting was done.
        - isolated_atom_energies: Mapping of element symbols to their isolated-atom reference
          energies.
    """
    if md_solver == "ase":
        md_maker = MDAseMaker
    # else:
    #     md_maker = MDLammpsMaker  # TODO MDLammpsMaker

    md_job = md_maker(
        force_field_name=starting_mlip,
        calculator_kwargs=calculator_kwargs,
        time_step=time_step,
        traj_interval=traj_interval,
        ensemble=ensemble,
        dynamics=dynamics,
        pressure=pressure,
        temperature_list=temperature_list,
        eqm_step_list=eqm_step_list,
        rate_list=rate_list,
        volume_custom_scale_factors=volume_custom_scale_factors,
        supercell_matrix=supercell_matrix,
        store_trajectory=store_trajectory,
        traj_file=traj_file,
        traj_file_fmt=traj_file_fmt,
        ionic_step_data=ionic_step_data,
    ).make(structure=structure)
    do_data_sampling = sample_data(
        selection_method=selection_method,
        num_of_selection=num_of_selection,
        traj_path=md_job.output,
        traj_type="md",
        random_seed=random_seed,
        remove_traj_files=remove_traj_files,
    )
    do_dft_static = DFTStaticLabelling(
        e0_spin=e0_spin,
        isolatedatom_box=isolatedatom_box,
        include_isolated_atom=include_isolated_atom,
        include_dimer=include_dimer,
        dimer_box=dimer_box,
        dimer_range=dimer_range,
        dimer_num=dimer_num,
        custom_incar=custom_incar,
        custom_potcar=custom_potcar,
        static_energy_maker=static_energy_maker,
        static_energy_maker_isolated_atoms=static_energy_maker_isolated_atoms,
        config_type=config_type,
    ).make(
        structures=do_data_sampling.output,
    )
    do_data_collection = collect_dft_data(
        dft_ref_file=dft_ref_file,
        dft_dirs=do_dft_static.output,
    )
    if isolated_atom_energies is None:
        isolated_atom_energies = do_data_collection.output["isolated_atom_energies"]
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
        isolated_atom_energies=isolated_atom_energies,
    )
    do_mlip_fit = MLIPFitMaker(
        mlip_type=iter_mlip,
        ref_energy_name=ref_energy_name,
        ref_force_name=ref_force_name,
        ref_virial_name=ref_virial_name,
        num_processes_fit=num_processes_fit,
        apply_data_preprocessing=False,
        auto_delta=auto_delta,
        glue_xml=False,
    ).make(
        database_dir=do_data_preprocessing.output,
        isolated_atom_energies=isolated_atom_energies,
        device=device_for_fitting,
        **fit_kwargs,
    )
    job_list = [
        md_job,
        do_data_sampling,
        do_dft_static,
        do_data_collection,
        do_data_preprocessing,
        do_mlip_fit,
    ]

    return Response(
        replace=Flow(job_list),
        output={
            "pre_database_dir": do_data_collection.output,
            "mlip_path": do_mlip_fit.output["mlip_path"][0],
            "isolated_atom_energies": do_data_collection.output[
                "isolated_atom_energies"
            ],
        },
    )
