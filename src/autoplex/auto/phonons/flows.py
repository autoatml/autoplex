"""Flows to perform automatic data generation, fitting, and benchmarking of ML potentials."""

import logging
import warnings
from dataclasses import dataclass, field

from atomate2.ase.jobs import AseMaker
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.mp import (
    MPGGADoubleRelaxMaker,
    MPGGARelaxMaker,
    MPGGAStaticMaker,
)
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.sets.core import TightRelaxSetGenerator
from jobflow import Flow, Maker
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPStaticSet,
)

from autoplex import MLIP_HYPERS
from autoplex.auto.phonons.jobs import (
    complete_benchmark,
    dft_phonopy_gen_data,
    dft_random_gen_data,
    do_iterative_rattled_structures,
    generate_supercells,
    get_iso_atom,
    get_phonon_output,
    run_supercells,
)
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from autoplex.data.phonons.flows import IsoAtomStaticMaker, TightDFTStaticMaker
from autoplex.data.phonons.jobs import reduce_supercell_size_job
from autoplex.fitting.common.flows import MLIPFitMaker

__all__ = [
    "CompleteDFTvsMLBenchmarkWorkflow",
    "CompleteDFTvsMLBenchmarkWorkflowMPSettings",
    "DFTSupercellSettingsMaker",
    "IterativeCompleteDFTvsMLBenchmarkWorkflow",
]


@dataclass
class CompleteDFTvsMLBenchmarkWorkflow(Maker):
    """
    Maker to construct a DFT (VASP) based dataset, composed of the following two configuration types.

    (1) single atom displaced supercells (based on the atomate2 PhononMaker subroutines)
    (2) supercells with randomly displaced atoms (based on the ase rattled function).

    Machine-learned interatomic potential(s) are then fitted on the dataset, followed by
    benchmarking the resulting potential(s) to DFT (VASP) level using the provided benchmark
    structure(s) and comparing the respective DFT and MLIP-based Phonon calculations.
    The benchmark metrics are provided in form of a phonon band structure comparison and
    q-point-wise phonons RMSE plots, as well as a summary text file.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    add_dft_phonon_struct: bool.
        If True, will add displaced supercells via phonopy for DFT calculation.
    add_dft_rattled_struct: bool.
        If True, will add rattled structures for DFT calculation.
    add_rss_struct: bool.
        If True, will add RSS generated structures for DFT calculation.
    n_structures: int.
        The total number of randomly displaced structures to be generated.
    displacement_maker: BaseVaspMaker|AseMaker
        Maker used for a static calculation for a supercell.
    phonon_bulk_relax_maker: BaseVaspMaker|AseMaker
        Maker used for the bulk relax unit cell calculation.
    rattled_bulk_relax_maker: BaseVaspMaker|AseMaker
        Maker used for the bulk relax unit cell calculation.
    phonon_static_energy_maker: BaseVaspMaker|AseMaker
        Maker used for the static energy unit cell calculation.
    isolated_atom_maker: IsoAtomStaticMaker|AseMaker
        VASP maker for the isolated atom calculation.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    displacements: list[float]
        Displacement distances for phonon data generation for the fiting.
        Only 0.01 is used for the benchmark at the moment.
        This value can currently not be changed.
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
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
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.
    ml_models: list[str]
        List of the ML models to be used. Default is GAP.
    force_max: float
        Maximum allowed force in the dataset.
    force_min: float
        Minimal force cutoff value for atom-wise regularization.
    split_ratio: float.
        Parameter to divide the training set and the test set.
        A value of 0.1 means that the ratio of the training set to the test set is 9:1.
    regularization: bool
        For using sigma regularization.
    distillation: bool
        For using data distillation.
    separated: bool
        Repeat the fit for each data_type available in the (combined) database.
    num_processes_fit: int
        Number of processes for fitting.
    apply_data_preprocessing: bool
        Apply data preprocessing.
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    atom_wise_regularization: bool
        For including atom-wise regularization.
    auto_delta: bool
        Automatically determines delta for 2b, 3b and soap terms.
    hyper_para_loop: bool
        Making it easier to loop through several hyperparameter sets.
    atomwise_regularization_list: list
        List of atom-wise regularization parameters that are checked.
    soap_delta_list: list
        List of SOAP delta values that are checked.
    n_sparse_list: list
        List of GAP n_sparse values that are checked.
    supercell_settings: dict
        Settings for supercell generation
    benchmark_kwargs: dict
        Keyword arguments for the benchmark flows
    summary_filename_prefix: str
        Prefix of the result summary file.
    glue_xml: bool
        Use the glue.xml core potential instead of fitting 2b terms.
    glue_file_path: str
        Name of the glue.xml file path.
    run_fits_on_different_cluster: bool
        Allows you to run fits on a different cluster than DFT (will transfer
        fit database via MongoDB, might be slow).
    """

    name: str = "add_data"
    add_dft_phonon_struct: bool = True
    add_dft_rattled_struct: bool = True
    add_rss_struct: bool = False
    displacement_maker: BaseVaspMaker | AseMaker = None
    phonon_bulk_relax_maker: BaseVaspMaker | AseMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(
            TightRelaxMaker(
                name="dft tight relax",
                run_vasp_kwargs={"handlers": {}},
                input_set_generator=TightRelaxSetGenerator(
                    user_incar_settings={
                        # TODO check if user_incar_settings has to be set like that or can be left out
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
                        "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                        "LOPTICS": False,  # No PCDAT file
                        "NSW": 200,
                        "NELM": 500,
                        # to be removed
                        "NPAR": 4,
                    }
                ),
            )
        )
    )
    phonon_static_energy_maker: BaseVaspMaker | AseMaker | None = None

    rattled_bulk_relax_maker: BaseVaspMaker | AseMaker | None = field(
        default_factory=lambda: TightRelaxMaker(
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
                    "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    "NSW": 200,
                    "NELM": 500,
                    # to be removed
                    "NPAR": 4,
                }
            ),
        )
    )
    isolated_atom_maker: IsoAtomStaticMaker | AseMaker | None = None
    n_structures: int = 10
    displacements: list[float] = field(default_factory=lambda: [0.01])
    symprec: float = 1e-4
    uc: bool = False
    volume_custom_scale_factors: list[float] | None = None
    volume_scale_factor_range: list[float] | None = None
    rattle_std: float = 0.01
    distort_type: int = 0
    min_distance: float = 1.5
    angle_percentage_scale: float = 10
    angle_max_attempts: int = 1000
    rattle_type: int = 0
    rattle_mc_n_iter: int = 10
    w_angle: list[float] | None = None
    ml_models: list[str] = field(default_factory=lambda: ["GAP"])
    atomwise_regularization_parameter: float = 0.1
    atom_wise_regularization: bool = True
    force_max: float = 40.0
    force_min: float = 0.01  # unit: eV Å-1
    split_ratio: float = 0.4
    regularization: bool = False
    separated: bool = False
    num_processes_fit: int | None = None
    distillation: bool = True
    apply_data_preprocessing: bool = True
    auto_delta: bool = False
    hyper_para_loop: bool = False
    atomwise_regularization_list: list | None = None
    soap_delta_list: list | None = None
    n_sparse_list: list | None = None
    supercell_settings: dict = field(
        default_factory=lambda: {"min_length": 15, "max_length": 20}
    )
    benchmark_kwargs: dict = field(default_factory=dict)
    summary_filename_prefix: str = "results_"
    glue_xml: bool = False
    glue_file_path: str = "glue.xml"
    run_fits_on_different_cluster: bool = False

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        hyperparameters: MLIP_HYPERS = MLIP_HYPERS,
        dft_references: list[PhononBSDOSDoc] | None = None,
        benchmark_structures: list[Structure] | None = None,
        benchmark_mp_ids: list[str] | None = None,
        pre_database_dir: str | None = None,
        pre_xyz_files: list[str] | None = None,
        rattle_seed: int | None = 42,
        fit_kwargs_list: list | None = None,
    ) -> Flow:
        """
        Make flow for constructing the dataset, fitting the potentials and performing the benchmarks.

        Parameters
        ----------
        structure_list:
            List of pymatgen structures.
        mp_ids:
            Materials Project IDs.
        hyperparameters: MLIP_HYPERS
            Hyperparameters for the MLIP models.
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
            Random seed for structure generation.
        fit_kwargs_list : list[dict].
            Dict including MLIP fit keyword args.

        """
        self.structure_list = structure_list
        self.mp_ids = mp_ids
        if rattle_seed is None:
            rattle_seed = 42

        flows = []
        fit_input = {}
        bm_outputs = []

        hyperparameters = hyperparameters.model_copy(deep=True)
        default_hyperparameters = hyperparameters.model_dump(by_alias=True)

        soap_default_dict = next(
            (
                {
                    key: value
                    for key, value in fit_kwargs["soap"].items()
                    if key in ["n_sparse", "delta"]
                }
                for fit_kwargs in (fit_kwargs_list or [])
                if "soap" in fit_kwargs
            ),
            default_hyperparameters["GAP"]["soap"],
        )
        # add an optional pre-optimization step here

        for structure, mp_id in zip(structure_list, mp_ids):
            self.supercell_settings.setdefault(mp_id, {})
            logging.warning(
                "Currently, "
                "the same supercell settings for single-atom displaced and rattled supercells are used."
            )
            supercell_matrix_job = reduce_supercell_size_job(
                structure=structure,
                min_length=self.supercell_settings.get("min_length", 15),
                max_length=self.supercell_settings.get("max_length", 20),
                fallback_min_length=self.supercell_settings.get(
                    "fallback_min_length", 12
                ),
                max_atoms=self.supercell_settings.get("max_atoms", 500),
                min_atoms=self.supercell_settings.get("min_atoms", 50),
                step_size=self.supercell_settings.get("step_size", 1.0),
            )
            flows.append(supercell_matrix_job)
            self.supercell_settings[mp_id][
                "supercell_matrix"
            ] = supercell_matrix_job.output

            # TODO: add a separate optimization here if add_dft_rattld_structu or add_dft_phonon_struct is activated
            # TODO: then forward the optimized structures to the next step

            if self.add_dft_rattled_struct:
                add_dft_ratt = self.add_dft_rattled(
                    structure=structure,
                    mp_id=mp_id,
                    displacement_maker=self.displacement_maker,
                    rattled_bulk_relax_maker=self.rattled_bulk_relax_maker,
                    n_structures=self.n_structures,
                    uc=self.uc,
                    volume_custom_scale_factors=self.volume_custom_scale_factors,
                    volume_scale_factor_range=self.volume_scale_factor_range,
                    rattle_std=self.rattle_std,
                    distort_type=self.distort_type,
                    min_distance=self.min_distance,
                    rattle_type=self.rattle_type,
                    rattle_seed=rattle_seed,
                    rattle_mc_n_iter=self.rattle_mc_n_iter,
                    angle_max_attempts=self.angle_max_attempts,
                    angle_percentage_scale=self.angle_percentage_scale,
                    w_angle=self.w_angle,
                    supercell_settings=self.supercell_settings,
                )
                add_dft_ratt.append_name(f"_{mp_id}")
                flows.append(add_dft_ratt)
                if self.volume_custom_scale_factors is not None:
                    rattle_seed = rattle_seed + len(self.volume_custom_scale_factors)
                elif self.n_structures is not None:
                    rattle_seed = rattle_seed + self.n_structures
                fit_input.update({mp_id: add_dft_ratt.output})
            if self.add_dft_phonon_struct:
                add_dft_phon = self.add_dft_phonons(
                    structure=structure,
                    mp_id=mp_id,
                    displacements=self.displacements,
                    symprec=self.symprec,
                    phonon_bulk_relax_maker=self.phonon_bulk_relax_maker,
                    phonon_static_energy_maker=self.phonon_static_energy_maker,
                    phonon_displacement_maker=self.displacement_maker,
                    supercell_settings=self.supercell_settings,
                )
                flows.append(add_dft_phon)
                add_dft_phon.append_name(f"_{mp_id}")
                fit_input.update({mp_id: add_dft_phon.output})
            if self.add_dft_rattled_struct and self.add_dft_phonon_struct:
                fit_input.update(
                    {
                        mp_id: {
                            "rattled_dir": add_dft_ratt.output["rattled_dir"],
                            "phonon_dir": add_dft_phon.output["phonon_dir"],
                            "phonon_data": add_dft_phon.output["phonon_data"],
                        }
                    }
                )
            if self.add_rss_struct:
                raise NotImplementedError

        isoatoms = get_iso_atom(structure_list, self.isolated_atom_maker)
        flows.append(isoatoms)

        # isolated atoms energy needs to be added
        if pre_xyz_files is None:
            fit_input.update(
                {"IsolatedAtom": {"iso_atoms_dir": [isoatoms.output["dirs"]]}}
            )

        for ml_model, fit_kwargs in zip(self.ml_models, fit_kwargs_list or [{}]):
            add_data_fit = MLIPFitMaker(
                mlip_type=ml_model,
                glue_xml=self.glue_xml,
                glue_file_path=self.glue_file_path,
                split_ratio=self.split_ratio,
                force_max=self.force_max,
                pre_xyz_files=pre_xyz_files,
                pre_database_dir=pre_database_dir,
                atomwise_regularization_parameter=self.atomwise_regularization_parameter,
                force_min=self.force_min,
                atom_wise_regularization=self.atom_wise_regularization,
                auto_delta=self.auto_delta,
                apply_data_preprocessing=self.apply_data_preprocessing,
                num_processes_fit=self.num_processes_fit,
                separated=self.separated,
                regularization=self.regularization,
                distillation=self.distillation,
                run_fits_on_different_cluster=self.run_fits_on_different_cluster,
            ).make(
                species_list=isoatoms.output["species"],
                isolated_atom_energies=isoatoms.output["energies"],
                hyperparameters=hyperparameters,
                fit_input=fit_input,
                **fit_kwargs,
            )
            flows.append(add_data_fit)

            if (benchmark_structures is not None) and (benchmark_mp_ids is not None):
                dft_new_references = []
                for ibenchmark_structure, benchmark_structure in enumerate(
                    benchmark_structures
                ):
                    # hard coded at the moment as other displacements
                    # are not treated correctly in benchmark part
                    complete_bm = complete_benchmark(
                        ibenchmark_structure=ibenchmark_structure,
                        benchmark_structure=benchmark_structure,
                        ml_model=ml_model,
                        ml_path=add_data_fit.output["mlip_path"],
                        mp_ids=mp_ids,
                        benchmark_mp_ids=benchmark_mp_ids,
                        add_dft_phonon_struct=self.add_dft_phonon_struct,
                        fit_input=fit_input,
                        symprec=self.symprec,
                        phonon_bulk_relax_maker=self.phonon_bulk_relax_maker,
                        phonon_static_energy_maker=self.phonon_static_energy_maker,
                        phonon_displacement_maker=self.displacement_maker,
                        dft_references=dft_references,
                        supercell_settings=self.supercell_settings,
                        displacement=0.01,
                        atomwise_regularization_parameter=self.atomwise_regularization_parameter,
                        soap_dict=soap_default_dict,
                        **self.benchmark_kwargs,
                    )
                    complete_bm.append_name(
                        f"_{benchmark_mp_ids[ibenchmark_structure]}"
                    )
                    flows.append(complete_bm)
                    bm_outputs.append(complete_bm.output["bm_output"])
                    dft_new_references.append(complete_bm.output["dft_references"])

            if self.hyper_para_loop:
                if self.atomwise_regularization_list is None:
                    self.atomwise_regularization_list = [0.1, 0.01, 0.001, 0.0001]
                if self.soap_delta_list is None:
                    self.soap_delta_list = [0.5, 1.0, 1.5]
                if self.n_sparse_list is None:
                    self.n_sparse_list = [
                        1000,
                        2000,
                        3000,
                        4000,
                        5000,
                        6000,
                        7000,
                        8000,
                        9000,
                    ]
                for atomwise_reg_parameter in self.atomwise_regularization_list:
                    for n_sparse in self.n_sparse_list:
                        for delta in self.soap_delta_list:
                            soap_dict = {
                                "n_sparse": n_sparse,
                                "delta": delta,
                            }
                            loop_data_fit = MLIPFitMaker(
                                mlip_type=ml_model,
                                glue_xml=self.glue_xml,
                                glue_file_path=self.glue_file_path,
                                split_ratio=self.split_ratio,
                                force_max=self.force_max,
                                pre_xyz_files=pre_xyz_files,
                                pre_database_dir=pre_database_dir,
                                atomwise_regularization_parameter=atomwise_reg_parameter,
                                force_min=self.force_min,
                                auto_delta=self.auto_delta,
                                num_processes_fit=self.num_processes_fit,
                                separated=self.separated,
                                regularization=self.regularization,
                                distillation=self.distillation,
                            ).make(
                                species_list=isoatoms.output["species"],
                                isolated_atom_energies=isoatoms.output["energies"],
                                hyperparameters=hyperparameters,
                                fit_input=fit_input,
                                soap=soap_dict,
                            )
                            flows.append(loop_data_fit)

                            if (benchmark_structures is not None) and (
                                benchmark_mp_ids is not None
                            ):
                                dft_new_references = []
                                for (
                                    ibenchmark_structure,
                                    benchmark_structure,
                                ) in enumerate(benchmark_structures):

                                    complete_bm = complete_benchmark(
                                        ibenchmark_structure=ibenchmark_structure,
                                        benchmark_structure=benchmark_structure,
                                        ml_model=ml_model,
                                        ml_path=loop_data_fit.output["mlip_path"],
                                        mp_ids=mp_ids,
                                        benchmark_mp_ids=benchmark_mp_ids,
                                        add_dft_phonon_struct=self.add_dft_phonon_struct,
                                        fit_input=fit_input,
                                        symprec=self.symprec,
                                        phonon_bulk_relax_maker=self.phonon_bulk_relax_maker,
                                        phonon_static_energy_maker=self.phonon_static_energy_maker,
                                        phonon_displacement_maker=self.displacement_maker,
                                        dft_references=dft_references,
                                        supercell_settings=self.supercell_settings,
                                        displacement=0.01,
                                        atomwise_regularization_parameter=atomwise_reg_parameter,
                                        soap_dict=soap_dict,
                                        **self.benchmark_kwargs,
                                    )
                                    complete_bm.append_name(
                                        f"_{benchmark_mp_ids[ibenchmark_structure]}"
                                    )
                                    flows.append(complete_bm)
                                    bm_outputs.append(complete_bm.output["bm_output"])
                                    # save the dft references okay

                                    dft_new_references.append(
                                        complete_bm.output["dft_references"]
                                    )

        collect_bm = write_benchmark_metrics(
            benchmark_structures=benchmark_structures,
            metrics=bm_outputs,
            filename_prefix=self.summary_filename_prefix,
        )
        # collect_bm must be extended in a way that we get access to all benchmark structures and the previous databases

        flows.append(collect_bm)

        output_flow = get_phonon_output(
            metrics=collect_bm.output,
            benchmark_structures=benchmark_structures,
            benchmark_mp_ids=benchmark_mp_ids,
            dft_references=dft_new_references,
            pre_xyz_files=pre_xyz_files,
            pre_database_dir=add_data_fit.output["database_dir"],
            fit_kwargs_list=fit_kwargs_list,
        )
        flows.append(output_flow)
        return Flow(jobs=flows, output=output_flow.output, name=self.name)

    @staticmethod
    def add_dft_phonons(
        structure: Structure,
        mp_id: str,
        displacements: list[float],
        symprec: float,
        phonon_bulk_relax_maker: BaseVaspMaker,
        phonon_static_energy_maker: BaseVaspMaker,
        phonon_displacement_maker: BaseVaspMaker,
        supercell_settings: dict,
    ):
        """Add DFT phonon runs for reference structures.

        Parameters
        ----------
        structure: Structure
            The pymatgen Structure object
        mp_id: str
            Materials Project ID
        displacements: list[float]
           Displacement distance for phonons
        symprec: float
            Symmetry precision to use in the
            reduction of symmetry to find the primitive/conventional cell
            (use_primitive_standard_structure, use_conventional_standard_structure)
            and to handle all symmetry-related tasks in phonopy
        phonon_displacement_maker: BaseVaspMaker
            Maker used to compute the forces for a supercell.
        phonon_bulk_relax_maker: BaseVaspMaker
            Maker used for the bulk relax unit cell calculation.
        phonon_static_energy_maker: BaseVaspMaker
            Maker used for the static energy unit cell calculation.
        supercell_settings: dict
            Supercell settings

        """
        dft_phonons = dft_phonopy_gen_data(
            structure=structure,
            mp_id=mp_id,
            displacements=displacements,
            symprec=symprec,
            phonon_bulk_relax_maker=phonon_bulk_relax_maker,
            phonon_static_energy_maker=phonon_static_energy_maker,
            phonon_displacement_maker=phonon_displacement_maker,
            supercell_settings=supercell_settings,
        )
        # let's append a name
        dft_phonons.name = "single-atom displaced supercells"
        return dft_phonons

    @staticmethod
    def add_dft_rattled(
        structure: Structure,
        mp_id: str,
        rattled_bulk_relax_maker: BaseVaspMaker,
        displacement_maker: BaseVaspMaker,
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
        """Add DFT static runs for randomly displaced structures.

        Parameters
        ----------
        structure: Structure
            The pymatgen Structure object
        mp_id: str
            Materials Project ID
        displacement_maker: BaseVaspMaker
            Maker used for a static calculation for a supercell.
        rattled_bulk_relax_maker: BaseVaspMaker
            Maker used for the bulk relax unit cell calculation.
        uc: bool.
            If True, will generate randomly distorted structures (unitcells)
            and add static computation jobs to the flow.
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
            Settings for supercells
        """
        additonal_dft_random = dft_random_gen_data(
            structure=structure,
            mp_id=mp_id,
            rattled_bulk_relax_maker=rattled_bulk_relax_maker,
            displacement_maker=displacement_maker,
            n_structures=n_structures,
            uc=uc,
            volume_custom_scale_factors=volume_custom_scale_factors,
            volume_scale_factor_range=volume_scale_factor_range,
            rattle_std=rattle_std,
            distort_type=distort_type,
            rattle_seed=rattle_seed,
            rattle_mc_n_iter=rattle_mc_n_iter,
            rattle_type=rattle_type,
            angle_max_attempts=angle_max_attempts,
            angle_percentage_scale=angle_percentage_scale,
            w_angle=w_angle,
            min_distance=min_distance,
            supercell_settings=supercell_settings,
        )
        additonal_dft_random.name = "rattled supercells"
        return additonal_dft_random


@dataclass
class CompleteDFTvsMLBenchmarkWorkflowMPSettings(CompleteDFTvsMLBenchmarkWorkflow):
    """
    Maker to construct a DFT (VASP) based dataset, composed of the following two configuration types.

    (1) single atom displaced supercells (based on the atomate2 PhononMaker subroutines)
    (2) supercells with randomly displaced atoms (based on the ase rattled function).

    Machine-learned interatomic potential(s) are then fitted on the dataset, followed by
    benchmarking the resulting potential(s) to DFT (VASP) level using the provided benchmark
    structure(s) and comparing the respective DFT and MLIP-based Phonon calculations.
    The benchmark metrics are provided in form of a phonon band structure comparison and
    q-point-wise phonons RMSE plots, as well as a summary text file.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    add_dft_phonon_struct: bool.
        If True, will add displaced supercells via phonopy for DFT calculation.
    add_dft_rattled_struct: bool.
        If True, will add randomly distorted structures for DFT calculation.
    add_rss_struct: bool.
        If True, will add RSS generated structures for DFT calculation.
        n_structures: int.
        The total number of randomly displaced structures to be generated.
    displacement_maker: BaseVaspMaker
        Maker used for a static calculation for a supercell.
    phonon_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    rattled_bulk_relax_maker: BaseVaspMaker
        Maker used for the bulk relax unit cell calculation.
    phonon_static_energy_maker: BaseVaspMaker
        Maker used for the static energy unit cell calculation.
    isolated_atom_maker: IsoAtomStaticMaker
        VASP maker for the isolated atom calculation.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    displacements: list[float]
        Displacement distances for phonons
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
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
    ml_models: list[str]
        List of the ML models to be used. Default is GAP.
    force_max: float
        Maximum allowed force in the dataset.
    force_min: float
        Minimal force cutoff value for atom-wise regularization.
    split_ratio: float.
        Parameter to divide the training set and the test set.
        A value of 0.1 means that the ratio of the training set to the test set is 9:1.
    apply_data_preprocessing: bool
        Apply data preprocessing.
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    atom_wise_regularization: bool
        For including atom-wise regularization.
    auto_delta: bool
        Automatically determines delta for 2b, 3b and soap terms.
    hyper_para_loop: bool
        If true, performs several fits using the provided hyperparameter sets.
    atomwise_regularization_list: list
        List of atom-wise regularization parameters that are checked.
    soap_delta_list: list
        List of SOAP delta values that are checked.
    n_sparse_list: list
        List of GAP n_sparse values that are checked.
    supercell_settings: dict
        Settings for supercell generation
    benchmark_kwargs: dict
        Keyword arguments for the benchmark flows
    summary_filename_prefix: str
        Prefix of the result summary file.
    glue_file_path: str
        Name of the glue.xml file path.
    """

    phonon_bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: MPGGADoubleRelaxMaker.from_relax_maker(
            MPGGARelaxMaker(
                name="dft tight relax",
                run_vasp_kwargs={"handlers": ()},
                input_set_generator=MPRelaxSet(
                    force_gamma=True,
                    auto_metal_kpoints=True,
                    inherit_incar=False,
                    user_incar_settings={
                        "NPAR": 4,
                        "EDIFF": 1e-7,
                        "EDIFFG": 1e-6,
                        "ALGO": "NORMAL",
                        "ISPIN": 1,
                        "LREAL": False,
                        "LCHARG": False,
                        "ISMEAR": 0,
                        "KSPACING": 0.2,
                    },
                ),
            )
        )
    )
    rattled_bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: MPGGADoubleRelaxMaker.from_relax_maker(
            MPGGARelaxMaker(
                name="dft tight relax",
                run_vasp_kwargs={"handlers": ()},
                input_set_generator=MPRelaxSet(
                    force_gamma=True,
                    auto_metal_kpoints=True,
                    inherit_incar=False,
                    user_incar_settings={
                        "NPAR": 4,
                        "EDIFF": 1e-7,
                        "EDIFFG": 1e-6,
                        "ALGO": "NORMAL",
                        "ISPIN": 1,
                        "LREAL": False,
                        "LCHARG": False,
                        "ISMEAR": 0,
                        "KSPACING": 0.2,
                    },
                ),
            )
        )
    )
    displacement_maker: BaseVaspMaker = field(
        default_factory=lambda: MPGGAStaticMaker(
            run_vasp_kwargs={"handlers": ()},
            name="dft phonon static",
            input_set_generator=MPStaticSet(
                force_gamma=True,
                auto_metal_kpoints=True,
                inherit_incar=False,
                user_incar_settings={
                    "NPAR": 4,
                    "EDIFF": 1e-7,
                    "EDIFFG": 1e-6,
                    "ALGO": "NORMAL",
                    "ISPIN": 1,
                    "LREAL": False,
                    "LCHARG": False,
                    "ISMEAR": 0,
                    "KSPACING": 0.2,
                },
            ),
        )
    )
    isolated_atom_maker: BaseVaspMaker = field(
        default_factory=lambda: MPGGAStaticMaker(
            run_vasp_kwargs={"handlers": ()},
            input_set_generator=MPStaticSet(
                user_kpoints_settings={"reciprocal_density": 1},
                force_gamma=True,
                auto_metal_kpoints=True,
                inherit_incar=False,
                user_incar_settings={
                    "NPAR": 4,
                    "EDIFF": 1e-7,
                    "EDIFFG": 1e-6,
                    "ALGO": "NORMAL",
                    "ISPIN": 1,
                    "LREAL": False,
                    "LCHARG": False,
                    "ISMEAR": 0,
                },
            ),
        )
    )

    phonon_static_energy_maker: BaseVaspMaker = None


@dataclass
class DFTSupercellSettingsMaker(Maker):
    """
    Maker to test the DFT and supercell settings.

    This maker is used to test your queue settings for the rattled and phonon supercells.
    Although the cells are not displaced, it provides an impression of the required memory
    and other resources as the process runs without symmetry considerations.

    Parameters
    ----------
    name (str): The name of the maker. Default is "test dft and supercell settings".
    supercell_settings (dict): Settings for the supercells. Default is {"min_length": 15}.
    DFT_Maker (BaseVaspMaker): The DFT maker to be used. Default is TightDFTStaticMaker.

    """

    name: str = "test dft and supercell settings"
    supercell_settings: dict = field(default_factory=lambda: {"min_length": 15})
    DFT_Maker: BaseVaspMaker = field(default_factory=TightDFTStaticMaker)

    def make(self, structure_list: list[Structure], mp_ids: list[str]) -> Flow:
        """
        Generate and runs supercell jobs for the given list of structures.

        Args:
            structure_list (list[Structure]): List of structures to process.
            mp_ids (list[str]): List of MP IDs.

        Returns
        -------
            Flow: A Flow object containing the jobs and their output.
        """
        job_list = []

        # Modify to run for more than one cell
        supercell_job = generate_supercells(structure_list, self.supercell_settings)
        job_list.append(supercell_job)

        supercell_job = run_supercells(
            structure_list, supercell_job.output, mp_ids, self.DFT_Maker
        )
        job_list.append(supercell_job)

        return Flow(jobs=job_list, output=supercell_job.output, name=self.name)


@dataclass
class IterativeCompleteDFTvsMLBenchmarkWorkflow:
    """
    Iterative Version of CompleteDFTvsMLBenchmarkWorkflow.

    Maker to run CompleteDFTvsMLBenchmarkWorkflow in an iterative
    fashion to ensure convergence of the potentials.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    max_iterations: int.
        Maximum number of iterations to run.
    rms_max: float.
        Will stop once the best potential has a max rmse below this value.
        The RMSE value is in THz.
    complete_dft_vs_ml_benchmark_workflow_0: CompleteDFTvsMLBenchmarkWorkflow.
        First Iteration will be performed with this flow.
    complete_dft_vs_ml_benchmark_workflow_1: CompleteDFTvsMLBenchmarkWorkflow.
        All Iterations after the first one will be performed with this flow.
    """

    name: str = "IterativeCompleteDFTvsMLBenchmarkWorkflow"
    max_iterations: int = 10
    rms_max: float = 0.2
    complete_dft_vs_ml_benchmark_workflow_0: CompleteDFTvsMLBenchmarkWorkflow | None = (
        field(default_factory=CompleteDFTvsMLBenchmarkWorkflow)
    )
    complete_dft_vs_ml_benchmark_workflow_1: CompleteDFTvsMLBenchmarkWorkflow | None = (
        field(
            default_factory=lambda: CompleteDFTvsMLBenchmarkWorkflow(
                add_dft_phonon_struct=False
            )
        )
    )

    def make(
        self,
        structure_list: list[Structure],
        mp_ids: list[str] | None = None,
        dft_references: list[PhononBSDOSDoc] | None = None,
        benchmark_structures: list[Structure] | None = None,
        benchmark_mp_ids: list[str] | None = None,
        pre_database_dir: str | None = None,
        pre_xyz_files: list[str] | None = None,
        rattle_seed: int = 0,
        fit_kwargs_list: list | None = None,
    ) -> Flow:
        """Make flow for constructing the dataset, fitting the potentials and performing the benchmarks.

        Parameters
        ----------
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

        """
        flow = do_iterative_rattled_structures(
            workflow_maker_gen_0=self.complete_dft_vs_ml_benchmark_workflow_0,
            workflow_maker_gen_1=self.complete_dft_vs_ml_benchmark_workflow_1,
            structure_list=structure_list,
            mp_ids=mp_ids,
            benchmark_structures=benchmark_structures,
            benchmark_mp_ids=benchmark_mp_ids,
            number_of_iteration=0,
            rms=None,
            max_iteration=self.max_iterations,
            rms_max=self.rms_max,
            rattle_seed=rattle_seed,
            dft_references=dft_references,
            fit_kwargs_list=fit_kwargs_list,
            pre_database_dir=pre_database_dir,
            pre_xyz_files=pre_xyz_files,
            previous_output=None,
        )
        return Flow(flow, flow.output)

    def __post_init__(self) -> None:
        """Test settings during the initialisation."""
        if self.complete_dft_vs_ml_benchmark_workflow_1.add_dft_phonon_struct:
            warnings.warn(
                "Phonon Data is generated in the second iteration. This will likely"
                "waste resources. It is recommended to switch this off.",
                stacklevel=2,
            )
