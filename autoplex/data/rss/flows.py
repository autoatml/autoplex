"""Flows for running RSS."""

from __future__ import annotations

from jobflow import Flow, Response, job

from autoplex.data.common.flows import DFTStaticMaker
from autoplex.data.common.jobs import (
    Data_preprocessing,
    Sampling,
    VASP_collect_data,
)
from autoplex.data.rss.jobs import RandomizedStructure, do_rss_multi_node
from autoplex.fitting.common.flows import MLIPFitMaker


@job
def initial_RSS(
    struct_number: int = 10000,
    tag: str = "GeSb2Te4",
    selection_method: str = "cur",
    num_of_selection: int = 3,
    bcur_params: dict | None = None,
    random_seed: int = None,
    e0_spin: bool = False,
    isolated_atom: bool = True,
    dimer: bool = True,
    dimer_range: list = None,
    dimer_num: int = 10,
    custom_set: dict | None = None,
    config_types: list[str] | None = None,
    vasp_ref_file: str = "vasp_ref.extxyz",
    rss_group: str = "initial",
    test_ratio: float = 0.1,
    regularization: bool = True,
    distillation: bool = True,
    f_max: float = 200,
    pre_database_dir: str | None = None,
    mlip_type: str = "GAP",
    mlip_hyper: dict | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    num_processes_fit: int | None = None,
    kt: float = None,
    **fit_kwargs,
):
    """
    Run initial Random Structure Searching (RSS) workflow from scratch.

    The workflow consists of the following jobs:
    job1 - RandomizedStructure: Generates randomized structures
    job2 - Sampling: Samples a subset of the generated structures using CUR
    job3 - DFTStaticMaker: Runs single-point calculations on the sampled structures
    job4 - VASP_collect_data: Collects VASP calculation data
    job5 - Data_preprocessing: Preprocesses the data for fitting ML models
    job6 - MLIPFitMaker: Fits a ML interatomic potential (MLIP)

    Parameters
    ----------
    struct_number : int, optional
        Number of structures to generate. Default is 10000.
    tag : str, optional
        Tag for the generated structures. Default is 'GeSb2Te4'.
    selection_method : str, optional
        Method for selecting structures. Default is 'cur'.
    num_of_selection : int, optional
        Number of structures to select. Default is 3.
    bcur_params : str, optional
        Parameters for the CUR method. Default is None.
    random_seed : int, optional
        Seed for random number generator. Default is None.
    e0_spin : bool, optional
        Whether to include spin polarization in the static calculations of isolated atoms and dimers. Default is False.
    isolated_atom : bool, optional
        Whether to include isolated atom calculations. Default is True.
    dimer : bool, optional
        Whether to include dimer calculations. Default is True.
    dimer_range : list, optional
        Distance range for dimer calculations. Default is None.
    dimer_num : int, optional
        Number of dimers generated for calculations. Default is None.
    custom_set : dict, optional
        Custom set of parameters for VASP. Default is None.
    config_types : list[str], optional
        List of configuration types corresponding to the structures. If provided,
        should have the same length as the 'structures' list. If None, defaults
        to 'bulk'. Default is None.
    vasp_ref_file : str, optional
        File name of collected VASP data. Default is 'vasp_ref.extxyz'.
    rss_group : str, optional
        Group name of structures for RSS. Default is 'initial'.
    test_ratio : float, optional
        The proportion of the test set after splitting the data. Default is 0.1.
    regularization : bool, optional
        Whether to apply regularization. This only works for GAP. Default is True.
    distillation : bool, optional
        Whether to apply distillation of structures. Default is True.
    f_max : float, optional
        Maximum force value to exclude structures. Default is 200.
    pre_database_dir : str, optional
        Directory for the preprocessed database. Default is None.
    mlip_type : str, optional
        Type of MLIP to fit. Default is 'GAP'.
    mlip_hyper : str, optional
        Hyperparameters for the MLIP. Default is None.
    ref_energy_name : str, optional
        Reference energy name. Default is "REF_energy".
    ref_force_name : str, optional
        Reference force name. Default is "REF_forces".
    ref_virial_name : str, optional
        Reference virial name. Default is "REF_virial".
    num_processes_fit : int, optional
        Number of processes for fitting. Default is None.
    kt : float, optional
        Value of kT. Default is None.
    fit_kwargs : dict, optional
        Additional arguments for the machine learning fit. Default is None.

    Output
    ------
    - test_error: float
        The test error of the fitted MLIP.
    - pre_database_dir: str
        The directory of the preprocessed database.
    - mlip_path: str
        The path to the fitted MLIP.
    - isol_es: dict
        The isolated energy values.
    - current_iter: int
        The current iteration index, set to 0.
    - kt: float
        The value of kT.
    """
    job1 = RandomizedStructure(struct_number=struct_number, tag=tag).make()
    job2 = Sampling(
        selection_method=selection_method,
        num_of_selection=num_of_selection,
        bcur_params=bcur_params,
        dir=job1.output,
        random_seed=random_seed,
    )
    job3 = DFTStaticMaker(
        e0_spin=e0_spin,
        isolated_atom=isolated_atom,
        dimer=dimer,
        dimer_range=dimer_range,
        dimer_num=dimer_num,
        custom_set=custom_set,
    ).make(structures=job2.output, config_types=config_types)
    job4 = VASP_collect_data(
        vasp_ref_file=vasp_ref_file, rss_group=rss_group, vasp_dirs=job3.output
    )
    job5 = Data_preprocessing(
        test_ratio=test_ratio,
        regularization=regularization,
        distillation=distillation,
        f_max=f_max,
        vasp_ref_dir=job4.output["vasp_ref_dir"],
        pre_database_dir=pre_database_dir,
    )
    job6 = MLIPFitMaker(
        mlip_type=mlip_type,
        mlip_hyper=mlip_hyper,
        ref_energy_name=ref_energy_name,
        ref_force_name=ref_force_name,
        ref_virial_name=ref_virial_name,
    ).make(
        database_dir=job5.output,
        isol_es=job4.output["isol_es"],
        num_processes_fit=num_processes_fit,
        preprocessing_data=False,
        **fit_kwargs,
    )

    job_list = [job1, job2, job3, job4, job5, job6]

    return Response(
        replace=Flow(job_list),
        output={
            "test_error": job6.output["test_error"],
            "pre_database_dir": job5.output,
            "mlip_path": job6.output["mlip_path"],
            "isol_es": job4.output["isol_es"],
            "current_iter": 0,
            "kt": 0.6,
        },
    )


@job
def do_RSS_iterations(
    input: dict,
    struct_number: int = 10000,
    tag: str = "GeSb2Te4",
    selection_method1: str = "cur",
    selection_method2: str = "bcur",
    num_of_selection1: int = 3,
    num_of_selection2: int = 5,
    bcur_params: dict | None = None,
    random_seed: int = None,
    e0_spin: bool = False,
    isolated_atom: bool = True,
    dimer: bool = True,
    dimer_range: list = None,
    dimer_num: int = 10,
    custom_set: dict | None = None,
    config_types: list[str] | None = None,
    vasp_ref_file: str = "vasp_ref.extxyz",
    rss_group: str = "initial",
    test_ratio: float = 0.1,
    regularization: bool = True,
    distillation: bool = True,
    f_max: float = 200,
    mlip_type: str = "GAP",
    mlip_hyper: dict | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    num_processes_fit: int = None,
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 10,
    force_tol: float = 0.1,
    stress_tol: float = 0.1,
    Hookean_repul: bool = False,
    write_traj: bool = True,
    num_processes_rss: int = 4,
    device: str = "cpu",
    stop_criterion: float = 0.01,
    max_iteration_number: int = 9,
    num_groups: int = 5,
    config_type: str = "traj",
    **fit_kwargs,
):
    """
    Perform iterative RSS to improve the accuracy of a MLIP.

    Each iteration involves generating new structures, sampling, running
    VASP calculations, collecting data, preprocessing data, and fitting a new MLIP.
    """
    valid_keys = {
        "test_error",
        "pre_database_dir",
        "mlip_path",
        "isol_es",
        "current_iter",
        "kt",
    }

    for key in input:
        if key not in valid_keys:
            raise ValueError(
                f"Invalid key '{key}' in input dictionary. Allowed keys are: {valid_keys}"
            )

    test_error = input.get("test_error")
    current_iter = input.get("current_iter")

    if (
        test_error is not None
        and test_error > stop_criterion
        and current_iter is not None
        and current_iter < max_iteration_number
    ):
        kt = input["kt"] - 0.1 if input["kt"] > 0.15 else 0.1
        print("kt:", kt)
        current_iter += 1
        print("Current iter index:", current_iter)
        print(f"The error of {current_iter}th iteration:", test_error)

        if bcur_params is None:
            bcur_params = {}
        bcur_params["kT"] = kt

        job1 = RandomizedStructure(struct_number=struct_number, tag=tag).make()
        job2 = Sampling(
            selection_method=selection_method1,
            num_of_selection=num_of_selection1,
            bcur_params=bcur_params,
            dir=job1.output,
            random_seed=random_seed,
        )
        job3 = do_rss_multi_node(
            mlip_type=mlip_type,
            iteration_index=f"{current_iter}th",
            mlip_path=input["mlip_path"],
            structure=job2.output,
            scalar_pressure_method=scalar_pressure_method,
            scalar_exp_pressure=scalar_exp_pressure,
            scalar_pressure_exponential_width=scalar_pressure_exponential_width,
            scalar_pressure_low=scalar_pressure_low,
            scalar_pressure_high=scalar_pressure_high,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            Hookean_repul=Hookean_repul,
            write_traj=write_traj,
            num_processes_rss=num_processes_rss,
            device=device,
            num_groups=num_groups,
            config_type=config_type,
        )
        job4 = Sampling(
            selection_method=selection_method2,
            num_of_selection=num_of_selection2,
            bcur_params=bcur_params,
            traj_path=job3.output,
            random_seed=random_seed,
            isol_es=input["isol_es"],
        )
        job5 = DFTStaticMaker(
            e0_spin=e0_spin,
            isolated_atom=isolated_atom,
            dimer=dimer,
            dimer_range=dimer_range,
            dimer_num=dimer_num,
            custom_set=custom_set,
        ).make(structures=job4.output, config_types=config_types)
        job6 = VASP_collect_data(
            vasp_ref_file=vasp_ref_file, rss_group=rss_group, vasp_dirs=job5.output
        )
        job7 = Data_preprocessing(
            test_ratio=test_ratio,
            regularization=regularization,
            distillation=distillation,
            f_max=f_max,
            vasp_ref_dir=job6.output["vasp_ref_dir"],
            pre_database_dir=input["pre_database_dir"],
        )
        job8 = MLIPFitMaker(
            mlip_type=mlip_type,
            mlip_hyper=mlip_hyper,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
        ).make(
            database_dir=job7.output,
            isol_es=input["isol_es"],
            num_processes_fit=num_processes_fit,
            preprocessing_data=False,
            **fit_kwargs,
        )

        job9 = do_RSS_iterations(
            input={
                "test_error": job8.output["test_error"],
                "pre_database_dir": job7.output,
                "mlip_path": job8.output["mlip_path"],
                "isol_es": input["isol_es"],
                "current_iter": current_iter,
                "kt": kt,
            },
        )

        job_list = [job1, job2, job3, job4, job5, job6, job7, job8, job9]

        return Response(detour=job_list, output=job9.output)

    return Response(output=input)
