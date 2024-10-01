"""General AutoPLEX automation jobs."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from autoplex.data.phonons.flows import (
    DFTPhononMaker,
    IsoAtomMaker,
    MLPhononMaker,
    RandomStructuresDataGenerator,
    TightDFTStaticMaker,
    TightDFTStaticMakerBigSupercells,
)
from autoplex.data.phonons.jobs import reduce_supercell_size
from autoplex.data.phonons.utils import update_phonon_displacement_maker


@job
def complete_benchmark(  # this function was put here to prevent circular import
    ml_path: str,
    ml_model: str,
    ibenchmark_structure: int,
    benchmark_structure: Structure,
    mp_ids,
    benchmark_mp_ids,  # list[str] mypy: Value of type "list[Any] | None" is not indexable
    add_dft_phonon_struct: bool,
    min_length: float,
    fit_input,
    symprec,
    phonon_displacement_maker: BaseVaspMaker,
    dft_references=None,
    adaptive_phonopy_supercell_settings: bool = True,
    relax_maker_kwargs: dict | None = None,
    static_maker_kwargs: dict | None = None,
    **ml_phonon_maker_kwargs,
):
    """
    Construct a complete flow for benchmarking the MLIP fit quality using a DFT based phonon structure.

    The complete benchmark flow starts by calculating the MLIP based phonon structure for each structure that has to be
    benchmarked. Then depending on if the user provided a DFT reference dataset or the DFT reference structure is
    already given from a previous loop, the existing or to-be-calculated DFT reference is used to generate the phonon
    bandstructure comparison plots, the q-point wise RMSE plots and to calculate the overall RMSE.
    This process is repeated with the default ML potential as well as the potentials from the different active user
    settings like sigma regularization, separated datasets or looping through several sets of hyperparameters.

    Parameters
    ----------
    ml_path: str
        Path to MLIP file. Default is path to gap_file.xml
    ml_model: str
        ML model to be used. Default is GAP.
    ibenchmark_structure: int
        ith benchmark structure.
    benchmark_structure: Structure
        pymatgen structure for benchmarking.
    benchmark_mp_ids: list[str]
        Materials Project ID of the benchmarking structure.
    mp_ids:
        materials project IDs.
    add_dft_phonon_struct: bool.
        If True, will add displaced supercells via phonopy for DFT calculation.
    min_length: float
        min length of the supercell that will be built
    fit_input : dict.
        CompletePhononDFTMLDataGenerationFlow output.
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    phonon_displacement_maker: BaseVaspMaker
        Maker used to compute the forces for a supercell.
    dft_references:
        a list of DFT reference files containing the PhononBSDOCDoc object. Default None.
    adaptive_phonopy_supercell_settings: bool
        prevent too tight phonopy supercell settings.
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    ml_phonon_maker_kwargs: dict
        Keyword arguments that can be passed to the MLPhononMaker.
    """
    jobs = []
    collect_output = []
    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")
    if min_length >= 18:
        phonon_displacement_maker = TightDFTStaticMakerBigSupercells()
    for suffix in ["", "_wo_sigma", "_phonon", "_rand_struc"]:
        # _wo_sigma", "_phonon", "_rand_struc" only available for GAP at the moment
        if ml_model == "GAP":
            ml_potential = Path(ml_path) / f"gap_file{suffix}.xml"
        elif ml_model == "J-ACE":
            raise UserWarning("No atomate2 ACE.jl PhononMaker implemented.")
        elif ml_model in ["M3GNET"]:
            ml_potential = (
                Path(ml_path) / suffix
            )  # M3GNet requires path and fit already returns the path
            # also need to find a different solution for separated fit then (name to path could be modified)
        elif ml_model in ["NEQUIP"]:
            ml_potential = Path(ml_path) / f"deployed_nequip_model{suffix}.pth"
        else:  # MACE
            ml_potential = Path(ml_path) / f"MACE_model{suffix}.model"

        if Path(ml_potential).exists():
            add_data_ml_phonon = MLPhononMaker(
                min_length=min_length,
                relax_maker_kwargs=relax_maker_kwargs,
                static_maker_kwargs=static_maker_kwargs,
            ).make_from_ml_model(
                structure=benchmark_structure,
                ml_model=ml_model,
                potential_file=ml_potential,
                **ml_phonon_maker_kwargs,
            )
            jobs.append(add_data_ml_phonon)

            # DFT benchmark reference preparations
            if dft_references is None and benchmark_mp_ids is not None:
                if (
                    benchmark_mp_ids[ibenchmark_structure] in mp_ids
                ) and add_dft_phonon_struct:
                    dft_references = fit_input[benchmark_mp_ids[ibenchmark_structure]][
                        "phonon_data"
                    ]["001"]
                elif (
                    benchmark_mp_ids[ibenchmark_structure] not in mp_ids
                ) or (  # else?
                    add_dft_phonon_struct is False
                ):
                    dft_phonons = dft_phonopy_gen_data(
                        structure=benchmark_structure,
                        displacements=[0.01],
                        symprec=symprec,
                        phonon_displacement_maker=phonon_displacement_maker,
                        min_length=min_length,
                        adaptive_phonopy_supercell_settings=adaptive_phonopy_supercell_settings,
                    )
                    jobs.append(dft_phonons)
                    dft_references = dft_phonons.output["data"]["001"]

                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    ml_model=ml_model,
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references,
                )
            elif (
                dft_references is not None
                and not isinstance(dft_references, list)
                and benchmark_mp_ids is not None
            ):
                if benchmark_mp_ids[ibenchmark_structure] not in mp_ids:
                    add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                        # this is important for re-using the same internally calculated DFT reference
                        # for looping through several settings
                        ml_model=ml_model,
                        structure=benchmark_structure,
                        benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                        ml_phonon_task_doc=add_data_ml_phonon.output,
                        dft_phonon_task_doc=dft_references,
                    )
            else:
                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    # this is important for using a provided DFT reference
                    ml_model=ml_model,
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references[ibenchmark_structure],
                )
            jobs.append(add_data_bm)
            collect_output.append(add_data_bm.output)

    return Response(replace=jobs, output=collect_output)


@job(data=["data"])
def dft_phonopy_gen_data(
    structure: Structure,
    displacements,
    symprec,
    phonon_displacement_maker,
    min_length,
    adaptive_phonopy_supercell_settings: bool = True,
):
    """
    Job to generate DFT reference database using phonopy to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        pymatgen Structure object.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    displacements: list[float]
        list of phonon displacement.
    min_length: float
        min length of the supercell that will be built.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    adaptive_phonopy_supercell_settings: bool
        prevent too tight phonopy supercell settings.
    """
    jobs = []
    dft_phonons_output = {}
    dft_phonons_dir_output = []
    supercell_matrix = None

    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")
    if min_length >= 15:
        phonon_displacement_maker = TightDFTStaticMakerBigSupercells()
    if adaptive_phonopy_supercell_settings:
        lattice_avg = sum(structure.lattice.abc) / 3
        if lattice_avg > 10.5:
            supercell_matrix_job = reduce_supercell_size(
                structure=structure,
                min_length=min_length,
                max_length=25,
                fallback_min_length=15,
                max_atoms=500,
                min_atoms=300,
                step_size=1.0,
            )
            jobs.append(supercell_matrix_job)
            supercell_matrix = supercell_matrix_job.output
            # in case everything fails, and a fitting supercell matrix cannot be found, reduce the
            # reciprocal k-point density and search for a supercell within the atomate2 phonon wf:
            if supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                supercell_matrix = None
                phonon_displacement_maker = update_phonon_displacement_maker(
                    lattice_avg, TightDFTStaticMakerBigSupercells()
                )

    for displacement in displacements:
        dft_phonons = DFTPhononMaker(
            symprec=symprec,
            phonon_displacement_maker=phonon_displacement_maker,
            born_maker=None,
            displacement=displacement,
            min_length=min_length,
            max_length=35,
        ).make(structure=structure, supercell_matrix=supercell_matrix)
        jobs.append(dft_phonons)
        dft_phonons_output[
            f"{displacement}".replace(".", "")  # key must not contain '.'
        ] = dft_phonons.output
        dft_phonons_dir_output.append(dft_phonons.output.jobdirs.displacements_job_dirs)

    flow = Flow(jobs, {"dirs": dft_phonons_dir_output, "data": dft_phonons_output})
    return Response(replace=flow)


@job
def dft_random_gen_data(
    structure: Structure,
    mp_id,
    phonon_displacement_maker,
    uc: bool = False,
    volume_custom_scale_factors: list[float] | None = None,
    volume_scale_factor_range: list[float] | None = None,
    rattle_std: float = 0.01,
    supercell_matrix: Matrix3D | None = None,
    distort_type: int = 0,
    n_structures: int = 10,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
    adaptive_rattled_supercell_settings: bool = True,
):
    """
    Job to generate random structured DFT reference database to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        pymatgen Structure object
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    mp_id:
        materials project id
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
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
    adaptive_rattled_supercell_settings: bool
        prevent too big rattled supercells
    """
    jobs = []

    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft rattle static")

    random_datagen = RandomStructuresDataGenerator(
        name="RandomDataGen",
        phonon_displacement_maker=phonon_displacement_maker,
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
        adaptive_rattled_supercell_settings=adaptive_rattled_supercell_settings,
    ).make(
        structure=structure,
        mp_id=mp_id,
        supercell_matrix=supercell_matrix,
        volume_custom_scale_factors=volume_custom_scale_factors,
        volume_scale_factor_range=volume_scale_factor_range,
    )
    jobs.append(random_datagen)

    flow = Flow(jobs, random_datagen.output)
    return Response(replace=flow)


@job
def get_iso_atom(structure_list: list[Structure]):
    """
    Job to collect all atomic species of the structures and starting VASP calculation of isolated atoms.

    Parameters
    ----------
    structure_list: list[Structure]
        list of pymatgen Structure objects
    """
    jobs = []
    iso_atoms_dict = {}
    all_species = list(
        {specie for s in structure_list for specie in s.types_of_species}
    )

    isoatoms = IsoAtomMaker().make(all_species=all_species)
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
