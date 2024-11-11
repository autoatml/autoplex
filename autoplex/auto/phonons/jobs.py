"""General AutoPLEX automation jobs."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job

if TYPE_CHECKING:
    from collections.abc import Iterable

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from pymatgen.core.structure import Structure

    from autoplex.data.phonons.flows import IsoAtomStaticMaker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator

from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from autoplex.data.phonons.flows import (
    DFTPhononMaker,
    IsoAtomMaker,
    MLPhononMaker,
    RandomStructuresDataGenerator,
    TightDFTStaticMaker,
)
from autoplex.data.phonons.jobs import reduce_supercell_size


@job
def complete_benchmark(  # this function was put here to prevent circular import
    ml_path: str,
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
        ith benchmark structure.
    benchmark_structure: Structure
        pymatgen structure for benchmarking.
    benchmark_mp_ids: list[str]
        Materials Project ID of the benchmarking structure.
    mp_ids:
        materials project IDs.
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
        a list of DFT reference files containing the PhononBSDOCDoc object.
        Default None.
    supercell_settings: dict
        settings for supercell generation
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    ml_phonon_maker_kwargs: dict
        Keyword arguments that can be passed to the MLPhononMaker.
    displacement: float
        displacement used in the finite displacement method.
    atomwise_regularization_parameter: float
        regularization value for the atom-wise force components.
    soap_dict: dict
        dictionary containing SOAP parameters.

    """
    jobs = []
    collect_output = []
    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")

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
            # treat finetuned potentials
            ml_potential_fine = Path(ml_path) / f"MACE_final{suffix}.model"
            if ml_potential_fine.exists():
                ml_potential = ml_potential_fine
            else:
                ml_potential = Path(ml_path) / f"MACE_model{suffix}.model"
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
                    ]["001"]
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
                    dft_references = dft_phonons.output["phonon_data"]["001"]

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
                    # this is important for re-using the same internally calculated DFT reference
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

    return Response(replace=Flow(jobs), output=collect_output)


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
    supercell_settings: dict
        settings for supercells

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
        list of supercells
    supercell_matrices: list[int]
        list of supercell matrices
    mp_ids: list[str]
        list of material ids
    dft_maker : .BaseVaspMaker or .ForceFieldStaticMaker or .BaseAimsMaker
        A maker to use to generate dispacement calculations
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
        pymatgen Structure object.
    mp_id: str
        materials project id
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    phonon_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    phonon_static_energy_maker: BaseVaspMaker
        Maker used for the static energy unit cell calculation.
    displacements: list[float]
        list of phonon displacement.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    supercell_settings:
        settings for supercell generation
    """
    jobs = []
    dft_phonons_output = {}
    dft_phonons_dir_output = []
    supercell_matrix = supercell_settings.get(mp_id, {}).get("supercell_matrix")
    if not supercell_matrix:
        supercell_matrix = reduce_supercell_size(structure, **supercell_settings)

    if phonon_displacement_maker is None:
        phonon_displacement_maker = TightDFTStaticMaker(name="dft phonon static")
    if phonon_bulk_relax_maker is None:
        phonon_bulk_relax_maker = DoubleRelaxMaker.from_relax_maker(
            TightRelaxMaker(
                run_vasp_kwargs={"handlers": {}},
                input_set_generator=TightRelaxSetGenerator(
                    user_incar_settings={
                        "ALGO": "Normal",
                        "ISPIN": 1,
                        "LAECHG": False,
                        "ISMEAR": 0,
                        "ENCUT": 700,
                        "ISYM": 0,
                        "SIGMA": 0.05,
                        "LCHARG": False,  # Do not write the CHGCAR file
                        "LWAVE": False,  # Do not write the WAVECAR file
                        "LVTOT": False,  # Do not write LOCPOT file
                        "LORBIT": 0,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                        "LOPTICS": False,  # No PCDAT file
                        "NSW": 200,
                        "NELM": 500,
                        # to be removed
                        "NPAR": 4,
                    }
                ),
            )
        )

    if phonon_static_energy_maker is None:
        phonon_static_energy_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                auto_ispin=False,
                user_incar_settings={
                    "ALGO": "Normal",
                    "ISPIN": 1,
                    "LAECHG": False,
                    "ISMEAR": 0,
                    "ENCUT": 700,
                    "SIGMA": 0.05,
                    "LCHARG": False,  # Do not write the CHGCAR file
                    "LWAVE": False,  # Do not write the WAVECAR file
                    "LVTOT": False,  # Do not write LOCPOT file
                    "LORBIT": 0,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    # to be removed
                    "NPAR": 4,
                },
            )
        )

    # always set autoplex default as job name
    phonon_displacement_maker.name = "dft phonon static"
    phonon_bulk_relax_maker.name = "tight relax"
    phonon_static_energy_maker.name = "static"

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
        pymatgen Structure object
    displacement_maker : .BaseVaspMaker or None
        Maker used for a static calculation for a supercell.
    rattled_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    mp_id:
        materials project id
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
        settings for supercells
    """
    jobs = []

    if displacement_maker is None:
        displacement_maker = TightDFTStaticMaker(name="dft rattle static")
    if rattled_bulk_relax_maker is None:
        rattled_bulk_relax_maker = TightRelaxMaker(
            run_vasp_kwargs={"handlers": {}},
            input_set_generator=TightRelaxSetGenerator(
                user_incar_settings={
                    "ALGO": "Normal",
                    "ISPIN": 1,
                    "LAECHG": False,
                    "ISYM": 0,  # to be changed
                    "ISMEAR": 0,
                    "SIGMA": 0.05,  # to be changed back
                    "LCHARG": False,  # Do not write the CHGCAR file
                    "LWAVE": False,  # Do not write the WAVECAR file
                    "LVTOT": False,  # Do not write LOCPOT file
                    "LORBIT": 0,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    "NSW": 200,
                    "NELM": 500,
                    # to be removed
                    "NPAR": 4,
                }
            ),
        )

    # always set autoplex default as job name
    displacement_maker.name = "dft rattle static"
    rattled_bulk_relax_maker.name = "tight relax"

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

    flow = Flow(jobs, {"rand_struc_dir": random_datagen.output})
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
        list of pymatgen Structure objects
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
