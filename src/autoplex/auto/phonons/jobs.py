"""General AutoPLEX automation jobs."""

from collections.abc import Iterable
from dataclasses import field
from pathlib import Path

import numpy as np
from atomate2.common.schemas.phonons import ForceConstants, PhononBSDOSDoc
from atomate2.vasp.jobs.base import BaseVaspMaker
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructure
from pymatgen.phonon.dos import PhononDos

from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from autoplex.data.phonons.flows import (
    DFTPhononMaker,
    IsoAtomMaker,
    IsoAtomStaticMaker,
    MLPhononMaker,
    RandomStructuresDataGenerator,
    TightDFTStaticMaker,
)
from autoplex.data.phonons.jobs import reduce_supercell_size


@job(
    data=[
        PhononBSDOSDoc,
        "dft_references",
        "metrics",
        "benchmark_structures",
        PhononDos,
        PhononBandStructure,
        ForceConstants,
        Structure,
    ]
)
def do_iterative_rattled_structures(
    workflow_maker_gen_0,
    workflow_maker_gen_1,
    structure_list: list[Structure],
    mp_ids: list[str],
    dft_references: list[PhononBSDOSDoc] | None = None,
    benchmark_structures: list[Structure] | None = None,
    benchmark_mp_ids: list[str] | None = None,
    pre_xyz_files: list[str] | None = None,
    pre_database_dir: str | None = None,
    rattle_seed: int | None = None,
    fit_kwargs_list: list | None = None,
    number_of_iteration=0,
    rms=0.2,
    max_iteration=5,
    rms_max=0.2,
    previous_output=None,
):
    """
    Job to run CompleteDFTvsMLBenchmarkWorkflow in an iterative manner.

    Parameters
    ----------
    workflow_maker_gen_0: CompleteDFTvsMLBenchmarkWorkflow.
        First Iteration will be performed with this flow.
    workflow_maker_gen_1: CompleteDFTvsMLBenchmarkWorkflow.
        All Iterations after the first one will be performed with this flow.
    structure_list:
            List of pymatgen structures.
    mp_ids:
        Materials Project IDs.
    dft_references: list[PhononBSDOSDoc] | None
        List of DFT reference files containing the PhononBSDOCDoc object.
        Reference files have to refer to a finite displacement of 0.01.
        For benchmarking, only 0.01 is supported
    benchmark_structures: list[Structure] | None
        The pymatgen structure for benchmarking.
    benchmark_mp_ids: list[str] | None
        Materials Project ID of the benchmarking structure.
    pre_xyz_files: list[str] or None
        Names of the pre-database train xyz file and test xyz file.
    pre_database_dir: str or None
        The pre-database directory.
    rattle_seed: int | None
        Random seed.
    fit_kwargs_list : list[dict].
        Dict including MLIP fit keyword args.
    number_of_iteration: int
        Number of iterations.
    rms: float
        current maximum rms value
    max_iteration: int.
        Maximum number of iterations to run.
    rms_max: float.
        Will stop once the best potential has a max rmse below this value.
    previous_output: dict | None.
        Dict including the output of the previous flow.
    """
    if rms is None or (number_of_iteration < max_iteration and rms > rms_max):
        jobs = []

        if number_of_iteration == 0:
            workflow_maker = workflow_maker_gen_0
            job1 = workflow_maker_gen_0.make(
                structure_list=structure_list,
                mp_ids=mp_ids,
                dft_references=dft_references,
                benchmark_structures=benchmark_structures,
                benchmark_mp_ids=benchmark_mp_ids,
                pre_xyz_files=pre_xyz_files,
                pre_database_dir=pre_database_dir,
                rattle_seed=rattle_seed,
                fit_kwargs_list=fit_kwargs_list,
            )
        else:
            workflow_maker = workflow_maker_gen_1
            job1 = workflow_maker_gen_1.make(
                structure_list=structure_list,
                mp_ids=mp_ids,
                dft_references=dft_references,
                benchmark_structures=benchmark_structures,
                benchmark_mp_ids=benchmark_mp_ids,
                pre_xyz_files=pre_xyz_files,
                pre_database_dir=pre_database_dir,
                rattle_seed=rattle_seed,
                fit_kwargs_list=fit_kwargs_list,
            )

        # rms needs to be computed somehow
        job1.append_name("_" + str(number_of_iteration))
        jobs.append(job1)
        # order is the same as in the scaling "scale_cells"
        if workflow_maker.volume_custom_scale_factors is not None:
            rattle_seed = rattle_seed + (
                len(workflow_maker.volume_custom_scale_factors)
                * len(workflow_maker.structure_list)
            )
        elif workflow_maker.n_structures is not None:
            rattle_seed = rattle_seed + (workflow_maker.n_structures) * len(
                workflow_maker.structure_list
            )

        job2 = do_iterative_rattled_structures(
            workflow_maker_gen_0=workflow_maker_gen_0,
            workflow_maker_gen_1=workflow_maker_gen_1,
            structure_list=structure_list,
            mp_ids=mp_ids,
            dft_references=job1.output["dft_references"],
            benchmark_structures=job1.output["benchmark_structures"],
            benchmark_mp_ids=job1.output["benchmark_mp_ids"],
            pre_xyz_files=job1.output["pre_xyz_files"],
            pre_database_dir=job1.output["pre_database_dir"],
            rattle_seed=rattle_seed,
            fit_kwargs_list=fit_kwargs_list,
            number_of_iteration=number_of_iteration + 1,
            rms=job1.output["rms"],
            max_iteration=max_iteration,
            rms_max=rms_max,
            previous_output=job1.output,
        )
        jobs.append(job2)
        # recreate the output to make sure all is correctly put into data:
        output_dict = {
            "dft_references": job2.output["dft_references"],
            "benchmark_structures": job2.output["benchmark_structures"],
            "benchmark_mp_ids": job2.output["benchmark_mp_ids"],
            "pre_xyz_files": job2.output["pre_xyz_files"],
            "pre_database_dir": job2.output["pre_database_dir"],
            "rms": job2.output["rms"],
            "metrics": job2.output["metrics"],
            "fit_kwargs_list": job2.output["fit_kwargs_list"],
        }

        return Response(replace=Flow(jobs), output=output_dict)

    return {
        "dft_references": previous_output["dft_references"],
        "benchmark_structures": previous_output["benchmark_structures"],
        "benchmark_mp_ids": previous_output["benchmark_mp_ids"],
        "pre_xyz_files": previous_output["pre_xyz_files"],
        "pre_database_dir": previous_output["pre_database_dir"],
        "rms": previous_output["rms"],
        "metrics": previous_output["metrics"],
        "fit_kwargs_list": previous_output["fit_kwargs_list"],
    }


@job(data=[PhononBSDOSDoc])
def complete_benchmark(  # this function was put here to prevent circular import
    ml_path: list,
    ml_model: str,
    ibenchmark_structure: int,
    benchmark_structure: Structure,
    mp_ids,
    benchmark_mp_ids,  # list[str] mypy: Value of type "list[Any] | None" is not indexable
    add_dft_phonon_struct: bool,
    fit_input,
    symprec,
    phonon_bulk_relax_maker: BaseVaspMaker,
    phonon_static_energy_maker: BaseVaspMaker,
    phonon_displacement_maker: BaseVaspMaker,
    atomwise_regularization_parameter: float,
    dft_references=None,
    soap_dict=None,
    displacement: float = 0.01,
    supercell_settings: dict | None = None,
    relax_maker_kwargs: dict | None = None,
    static_maker_kwargs: dict | None = None,
    **ml_phonon_maker_kwargs,
):
    """
    Construct a complete flow for benchmarking the MLIP fit quality using a DFT based phonon structure.

    The complete benchmark flow starts by calculating the MLIP based phonon structure for each structure that has to be
    benchmarked.
    Then, depending on if the user provided a DFT reference dataset or the DFT reference structure is
    already given from a previous loop, the existing or to-be-calculated DFT reference is used to generate the phonon
    bandstructure comparison plots, the q-point wise RMSE plots and to calculate the overall RMSE.
    This process is repeated with the default ML potential as well as the potentials from the different active user
    settings like sigma regularization, separated datasets or looping through several sets of hyperparameters.

    Parameters
    ----------
    ml_path: str
        Path to MLIP file.
        Default is path to gap_file.xml
    ml_model: str
        ML model to be used.
        Default is GAP.
    ibenchmark_structure: int
        The ith benchmark structure.
    benchmark_structure: Structure
        The pymatgen structure for benchmarking.
    benchmark_mp_ids: list[str]
        Materials Project ID of the benchmarking structure.
    mp_ids:
        Materials Project IDs.
    add_dft_phonon_struct: bool.
        If True, will add displaced supercells via phonopy for DFT calculation.
    fit_input : dict.
        CompletePhononDFTMLDataGenerationFlow output.
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    phonon_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    phonon_static_energy_maker: BaseVaspMaker
        Maker used for the static energy unit cell calculation.
    phonon_displacement_maker: BaseVaspMaker
        Maker used to compute the forces for a supercell.
    dft_references:
        List of DFT reference files containing the PhononBSDOCDoc object.
        Default None.
    supercell_settings: dict
        Settings for supercell generation
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    ml_phonon_maker_kwargs: dict
        Keyword arguments that can be passed to the MLPhononMaker.
    displacement: float
        Displacement used in the finite displacement method.
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    soap_dict: dict
        Dictionary containing SOAP parameters.

    """
    jobs = []
    collect_output = []

    for path in ml_path:
        suffix = Path(path).name
        if suffix == "without_regularization":
            suffix = "without_reg"
        if suffix not in ["phonon", "rattled"]:
            suffix = ""

        if phonon_displacement_maker is None:
            phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")

        if ml_model == "GAP":
            ml_potential = (
                Path(path) / "gap_file.xml"
            )  # TODO account for user-specific gap file name?
        elif ml_model == "J-ACE":
            raise UserWarning("No atomate2 ACE.jl PhononMaker implemented.")
        elif ml_model in ["M3GNET"]:
            ml_potential = Path(
                path
            )  # M3GNet requires path and fit already returns the path
            # also need to find a different solution for separated fit then (name to path could be modified)
        elif ml_model in ["NEP"]:
            ml_potential = Path(path) / "nep.txt"
        elif ml_model in ["NEQUIP"]:
            ml_potential = Path(path) / "deployed_nequip_model.pth"
        else:  # MACE
            # treat finetuned potentials
            # TODO: fix this naming issue (depends on input)
            ml_potential_fine = Path(path) / "MACE_final.model"
            ml_potential = (
                ml_potential_fine
                if ml_potential_fine.exists()
                else Path(path) / "MACE_model.model"
            )
        if Path(ml_potential).exists():
            add_data_ml_phonon = MLPhononMaker(
                relax_maker_kwargs=relax_maker_kwargs,
                static_maker_kwargs=static_maker_kwargs,
                displacement=displacement,
            ).make_from_ml_model(
                structure=benchmark_structure,
                ml_model=ml_model,
                potential_file=ml_potential,
                supercell_settings=supercell_settings,
                **ml_phonon_maker_kwargs,
            )
            jobs.append(add_data_ml_phonon)

            # DFT benchmark reference preparations
            if dft_references is None and benchmark_mp_ids is not None:
                # runs only the first time, then dft_references is added
                if (
                    benchmark_mp_ids[ibenchmark_structure] in mp_ids
                ) and add_dft_phonon_struct:

                    dft_references = fit_input[benchmark_mp_ids[ibenchmark_structure]][
                        "phonon_data"
                    ][f"{int(displacement * 100):03d}"]
                else:
                    dft_phonons = dft_phonopy_gen_data(
                        structure=benchmark_structure,
                        mp_id=benchmark_mp_ids[ibenchmark_structure],
                        displacements=[displacement],
                        symprec=symprec,
                        phonon_bulk_relax_maker=phonon_bulk_relax_maker,
                        phonon_static_energy_maker=phonon_static_energy_maker,
                        phonon_displacement_maker=phonon_displacement_maker,
                        supercell_settings=supercell_settings,
                    )
                    jobs.append(dft_phonons)
                    dft_references = dft_phonons.output["phonon_data"][
                        f"{int(displacement * 100):03d}"
                    ]

                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    ml_model=ml_model,
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references,
                    displacement=displacement,
                    atomwise_regularization_parameter=atomwise_regularization_parameter,
                    soap_dict=soap_dict,
                    suffix=suffix,
                )
            elif (
                dft_references is not None
                and not isinstance(dft_references, list)
                and benchmark_mp_ids is not None
            ):
                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    # this is important for reusing the same internally calculated DFT reference
                    # for looping through several settings
                    ml_model=ml_model,
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references,
                    displacement=displacement,
                    atomwise_regularization_parameter=atomwise_regularization_parameter,
                    soap_dict=soap_dict,
                    suffix=suffix,
                )
            else:
                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    # this is important for using a provided DFT reference
                    ml_model=ml_model,
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references[ibenchmark_structure],
                    displacement=displacement,
                    atomwise_regularization_parameter=atomwise_regularization_parameter,
                    soap_dict=soap_dict,
                    suffix=suffix,
                )
            jobs.append(add_data_bm)
            collect_output.append(add_data_bm.output)

    if isinstance(dft_references, list):
        return Response(
            replace=Flow(jobs),
            output={
                "bm_output": collect_output,
                "dft_references": dft_references[ibenchmark_structure],
            },
        )
    return Response(
        replace=Flow(jobs),
        output={"bm_output": collect_output, "dft_references": dft_references},
    )


@job
def generate_supercells(
    structures: list[Structure],
    supercell_settings: dict,
) -> list[Iterable]:
    """
    Run phonon displacements.

    Note, this job will replace itself with N displacement calculations,
    or a single socket calculation for all displacements.

    Parameters
    ----------
    structures : list[Structure]
        List of supercells.
    supercell_settings: dict
        Settings for supercells.

    """
    return [
        reduce_supercell_size(structure, **supercell_settings)
        for structure in structures
    ]


@job
def run_supercells(
    structures: list[Structure],
    supercell_matrices: list[int],
    mp_ids: list[str],
    dft_maker: BaseVaspMaker = None,
) -> Response:
    """
    Run supercell calculations.

    Note, this job will replace itself with supercell calculations.

    Parameters
    ----------
    structures: list[Structure]
        List of supercells.
    supercell_matrices: list[int]
        List of supercell matrices.
    mp_ids: list[str]
        List of Materials Project IDs.
    dft_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        Maker to use to generate dispacement calculations
    """
    if dft_maker is None:
        dft_maker = field(default_factory=TightDFTStaticMaker)
    dft_jobs = []
    outputs: dict[str, list] = {
        "uuids": [],
        "dirs": [],
        "mp-id": [],
    }

    for structure, supercell_matrix, mp_id in zip(
        structures, supercell_matrices, mp_ids
    ):
        structure.make_supercell(np.array(supercell_matrix).transpose())
        dft_job = dft_maker.make(structure=structure)
        dft_jobs.append(dft_job)
        outputs["uuids"].append(dft_job.output.uuid)
        outputs["dirs"].append(dft_job.output.dir_name)
        outputs["mp-id"].append(mp_id)

    displacement_flow = Flow(dft_jobs, outputs)
    return Response(replace=displacement_flow)


@job(data=["phonon_data"])
def dft_phonopy_gen_data(
    structure: Structure,
    mp_id: str,
    displacements,
    symprec,
    phonon_bulk_relax_maker,
    phonon_static_energy_maker,
    phonon_displacement_maker,
    supercell_settings,
):
    """
    Job to generate DFT reference database using phonopy to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        The pymatgen Structure object.
    mp_id: str
        Materials Project ID.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    phonon_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    phonon_static_energy_maker: BaseVaspMaker
        Maker used for the static energy unit cell calculation.
    displacements: list[float]
        List of phonon displacement.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    supercell_settings:
        Settings for supercell generation.
    """
    jobs = []
    dft_phonons_output = {}
    dft_phonons_dir_output = []
    if supercell_settings is None:
        supercell_settings = field(
            default_factory=lambda: {"min_length": 15, "max_length": 20}
        )
    supercell_matrix = supercell_settings.get(mp_id, {}).get("supercell_matrix")
    if not supercell_matrix:
        filtered_settings = {  # mismatching mp_ids would lead to a key error
            key: value
            for key, value in supercell_settings.items()
            if key
            in [
                "min_length",
                "max_length",
                "fallback_min_length",
                "max_atoms",
                "min_atoms",
                "step_size",
            ]
        }
        supercell_matrix = reduce_supercell_size(structure, **filtered_settings)

    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")

    # always set autoplex default as job name
    phonon_displacement_maker.name = "dft phonon static"
    if phonon_static_energy_maker is not None:
        phonon_static_energy_maker.name = "dft static"
    if phonon_bulk_relax_maker is not None:
        try:
            phonon_bulk_relax_maker.relax_maker1.name = "dft tight relax"
            phonon_bulk_relax_maker.relax_maker2.name = "dft tight relax"
        except AttributeError:
            phonon_bulk_relax_maker.name = "dft tight relax"

    for displacement in displacements:
        dft_phonons = DFTPhononMaker(
            symprec=symprec,
            static_energy_maker=phonon_static_energy_maker,
            bulk_relax_maker=phonon_bulk_relax_maker,
            phonon_displacement_maker=phonon_displacement_maker,
            born_maker=None,
            displacement=displacement,
        ).make(structure=structure, supercell_matrix=supercell_matrix)
        jobs.append(dft_phonons)
        dft_phonons_output[
            f"{displacement}".replace(".", "")  # key must not contain '.'
        ] = dft_phonons.output
        dft_phonons_dir_output.append(dft_phonons.output.jobdirs.displacements_job_dirs)

    flow = Flow(
        jobs,
        {"phonon_dir": dft_phonons_dir_output, "phonon_data": dft_phonons_output},
        name="dft_phononpy_gen_data",
    )
    return Response(replace=flow)


@job
def dft_random_gen_data(
    structure: Structure,
    mp_id,
    rattled_bulk_relax_maker,
    displacement_maker,
    uc: bool = False,
    volume_custom_scale_factors: list[float] | None = None,
    volume_scale_factor_range: list[float] | None = None,
    rattle_std: float = 0.01,
    distort_type: int = 0,
    n_structures: int = 10,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
    supercell_settings: dict | None = None,
):
    """
    Job to generate random structured DFT reference database to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        The pymatgen Structure object
    displacement_maker : .BaseVaspMaker or None
        Maker used for a static calculation for a supercell.
    rattled_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    mp_id:
        Materials Project ID.
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume +-10%.
    volume_custom_scale_factors : list[float]
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
    supercell_settings: dict
        Settings for supercells.
    """
    jobs = []

    if displacement_maker is None:
        displacement_maker = TightDFTStaticMaker(name="dft rattle static")

    # always set autoplex default as job name
    displacement_maker.name = "dft rattle static"
    if rattled_bulk_relax_maker is not None:
        rattled_bulk_relax_maker.name = "dft tight relax"

    # TODO: decide if we should remove the additional response here as well
    # looks like only the output is changing
    random_datagen = RandomStructuresDataGenerator(
        name="RandomDataGen",
        bulk_relax_maker=rattled_bulk_relax_maker,
        displacement_maker=displacement_maker,
        n_structures=n_structures,
        uc=uc,
        rattle_std=rattle_std,
        distort_type=distort_type,
        min_distance=min_distance,
        rattle_seed=rattle_seed,
        rattle_type=rattle_type,
        angle_max_attempts=angle_max_attempts,
        angle_percentage_scale=angle_percentage_scale,
        rattle_mc_n_iter=rattle_mc_n_iter,
        w_angle=w_angle,
        supercell_settings=supercell_settings,
    ).make(
        structure=structure,
        mp_id=mp_id,
        volume_custom_scale_factors=volume_custom_scale_factors,
        volume_scale_factor_range=volume_scale_factor_range,
    )
    jobs.append(random_datagen)

    flow = Flow(jobs, {"rattled_dir": random_datagen.output})
    return Response(replace=flow)


@job
def get_iso_atom(
    structure_list: list[Structure],
    isolated_atom_maker: IsoAtomStaticMaker,
):
    """
    Job to collect all atomic species of the structures and starting VASP calculation of isolated atoms.

    Parameters
    ----------
    structure_list: list[Structure]
        List of pymatgen Structure objects
    isolated_atom_maker: IsoAtomStaticMaker
        VASP maker for the isolated atom calculation.
    """
    jobs = []
    iso_atoms_dict = {}
    all_species = list(
        {specie for s in structure_list for specie in s.types_of_species}
    )

    isoatoms = IsoAtomMaker().make(
        all_species=all_species,
        isolated_atom_maker=isolated_atom_maker,
    )
    jobs.append(isoatoms)

    for i, species in enumerate(all_species):
        iso_atoms_dict.update({species.number: isoatoms.output["energies"][i]})

    flow = Flow(
        jobs,
        {
            "species": all_species,
            "energies": iso_atoms_dict,
            "dirs": isoatoms.output["dirs"],
        },
    )
    return Response(replace=flow)


@job(data=[PhononBSDOSDoc, "dft_references"])
def get_phonon_output(
    metrics: list,
    benchmark_structures: list[Structure] | None = None,
    benchmark_mp_ids: list[str] | None = None,
    dft_references: list[PhononBSDOSDoc] | None = None,
    pre_xyz_files: list[str] | None = None,
    pre_database_dir: str | None = None,
    fit_kwargs_list: list | None = None,
):
    """
    Job to collect and process all phonon-related output information for a potential restart of the flow.

    This function aggregates benchmark results, DFT reference data, and other input parameters
    to determine the best fit RMSE from phonon calculations across all benchmark fits.

    Parameters
    ----------
    metrics: list[dict]
        List of metric dictionaries from complete_benchmark jobs.
    dft_references: list[PhononBSDOSDoc] | None
        List of DFT reference files containing the PhononBSDOCDoc object.
        Reference files have to refer to a finite displacement of 0.01.
        For benchmarking, only 0.01 is supported
    benchmark_structures: list[Structure] | None
        The pymatgen structure for benchmarking.
    benchmark_mp_ids: list[str] | None
        Materials Project ID of the benchmarking structure.
    pre_xyz_files: list[str] or None
        Names of the pre-database train xyz file and test xyz file.
    pre_database_dir: str or None
        The pre-database directory.
    fit_kwargs_list : list[dict].
        Dict including MLIP fit keyword args.
    """
    # TODO: potentially evaluation of imaginary modes
    try:
        rms_max_values = []  # get the largest rms in each fit

        for i in range(len(metrics[0])):
            rms_max_value = max(
                sublist[i]["benchmark_phonon_rmse"] for sublist in metrics
            )
            rms_max_values.append(rms_max_value)
        rms = min(rms_max_values)
    except TypeError:
        # Set a large value as a fall back if None is discovered
        rms = 1000.0
    return {
        "metrics": metrics,
        "rms": rms,  # get the best fit RMSE
        "benchmark_structures": benchmark_structures,
        "benchmark_mp_ids": benchmark_mp_ids,
        "dft_references": dft_references,
        "pre_xyz_files": pre_xyz_files,
        "pre_database_dir": pre_database_dir,
        "fit_kwargs_list": fit_kwargs_list,
    }
