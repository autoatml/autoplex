"""Flows to perform automatic data generation, fitting, and benchmarking of ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from jobflow import Flow, Maker

from autoplex.auto.phonons.jobs import (
    complete_benchmark,
    dft_phonopy_gen_data,
    dft_random_gen_data,
    get_iso_atom,
)
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from autoplex.fitting.common.flows import MLIPFitMaker

__all__ = ["CompleteDFTvsMLBenchmarkWorkflow"]


# Volker's idea: provide several default flows with different setting/setups


@dataclass
class CompleteDFTvsMLBenchmarkWorkflow(Maker):
    """
    Maker to construct a DFT (VASP) based dataset, composed of the following two configuration types.

    1) single atom displaced supercells (based on the atomate2 PhononMaker subroutines)
    2) supercells with randomly displaced atoms (based on the ase rattled function).
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
    add_dft_random_struct: bool.
        If True, will add randomly distorted structures for DFT calculation.
    add_rss_struct: bool.
        If True, will add RSS generated structures for DFT calculation.
        n_structures: int.
        The total number of randomly displaced structures to be generated.
    phonon_displacement_maker: BaseVaspMaker
        Maker used to compute the forces for a supercell.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    displacements: list[float]
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy.
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
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
        list of the ML models to be used. Default is GAP.
    hyper_para_loop: bool
        making it easier to loop through several hyperparameter sets.
    atomwise_regularization_list: list
        List of atom-wise regularization parameters that are checked.
    soap_delta_list: list
        List of SOAP delta values that are checked.
    n_sparse_list: list
        List of GAP n_sparse values that are checked.
    adaptive_supercell_settings: bool
        prevent too big rattled supercells or too tight phonopy supercell settings.
    benchmark_kwargs: dict
        kwargs for the benchmark flows
    """

    name: str = "add_data"
    add_dft_phonon_struct: bool = True
    add_dft_random_struct: bool = True
    add_rss_struct: bool = False
    phonon_displacement_maker: BaseVaspMaker = None
    n_structures: int = 10
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    uc: bool = False
    volume_custom_scale_factors: list[float] | None = None
    volume_scale_factor_range: list[float] | None = None
    rattle_std: float = 0.01
    supercell_matrix: Matrix3D | None = None
    distort_type: int = 0
    min_distance: float = 1.5
    angle_percentage_scale: float = 10
    angle_max_attempts: int = 1000
    rattle_type: int = 0
    rattle_seed: int = 42
    rattle_mc_n_iter: int = 10
    w_angle: list[float] | None = None
    ml_models: list[str] = field(default_factory=lambda: ["GAP"])
    hyper_para_loop: bool = False
    atomwise_regularization_list: list | None = None
    soap_delta_list: list | None = None
    n_sparse_list: list | None = None
    adaptive_supercell_settings: bool = True
    benchmark_kwargs: dict = field(default_factory=dict)

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        split_ratio: float = 0.4,
        f_max: float = 40.0,
        pre_xyz_files: list[str] | None = None,
        pre_database_dir: str | None = None,
        preprocessing_data: bool = True,
        atomwise_regularization_parameter: float = 0.1,
        f_min: float = 0.01,  # unit: eV Å-1
        atom_wise_regularization: bool = True,
        auto_delta: bool = False,
        dft_references: list[PhononBSDOSDoc] | None = None,
        benchmark_structures: list[Structure] | None = None,
        benchmark_mp_ids: list[str] | None = None,
        **fit_kwargs,
    ):
        """
        Make flow for constructing the dataset, fitting the potentials and performing the benchmarks.

        Parameters
        ----------
        structure_list:
            list of pymatgen structures.
        mp_ids:
            materials project IDs.
        split_ratio: float.
            Parameter to divide the training set and the test set.
            A value of 0.1 means that the ratio of the training set to the test set is 9:1.
        f_max: float
            Maximally allowed force in the data set.
        pre_xyz_files: list[str] or None
            names of the pre-database train xyz file and test xyz file.
        pre_database_dir: str or None
            the pre-database directory.
        preprocessing_data: bool
            preprocessing the data.
        atomwise_regularization_parameter: float
            regularization value for the atom-wise force components.
        f_min: float
            minimal force cutoff value for atom-wise regularization.
        atom_wise_regularization: bool
            for including atom-wise regularization.
        auto_delta: bool
            automatically determine delta for 2b, 3b and soap terms.
        dft_references: list[PhononBSDOSDoc] | None
            a list of DFT reference files containing the PhononBSDOCDoc object.
        benchmark_structures: list[Structure] | None
            pymatgen structure for benchmarking.
        benchmark_mp_ids: list[str] | None
            Materials Project ID of the benchmarking structure.
        fit_kwargs : dict.
            dict including MLIP fit keyword args.

        """
        flows = []
        fit_input = {}
        hyper_list: list[dict[Any, Any]] = []
        bm_outputs = []

        for structure, mp_id in zip(structure_list, mp_ids):
            if self.add_dft_random_struct:
                addDFTrand = self.add_dft_random(
                    structure=structure,
                    mp_id=mp_id,
                    phonon_displacement_maker=self.phonon_displacement_maker,
                    n_structures=self.n_structures,
                    uc=self.uc,
                    volume_custom_scale_factors=self.volume_custom_scale_factors,
                    volume_scale_factor_range=self.volume_scale_factor_range,
                    rattle_std=self.rattle_std,
                    supercell_matrix=self.supercell_matrix,
                    distort_type=self.distort_type,
                    min_distance=self.min_distance,
                    rattle_type=self.rattle_type,
                    rattle_seed=self.rattle_seed,
                    rattle_mc_n_iter=self.rattle_mc_n_iter,
                    angle_max_attempts=self.angle_max_attempts,
                    angle_percentage_scale=self.angle_percentage_scale,
                    w_angle=self.w_angle,
                    adaptive_rattled_supercell_settings=self.adaptive_supercell_settings,
                )
                flows.append(addDFTrand)
                fit_input.update({mp_id: addDFTrand.output})
            if self.add_dft_phonon_struct:
                addDFTphon = self.add_dft_phonons(
                    structure,
                    self.displacements,
                    self.symprec,
                    self.phonon_displacement_maker,
                    self.min_length,
                    self.adaptive_supercell_settings,
                )
                flows.append(addDFTphon)
                fit_input.update({mp_id: addDFTphon.output})
            if self.add_dft_random_struct and self.add_dft_phonon_struct:
                fit_input.update({mp_id: {**addDFTrand.output, **addDFTphon.output}})
            if self.add_rss_struct:
                raise NotImplementedError

        isoatoms = get_iso_atom(structure_list)
        flows.append(isoatoms)

        if pre_xyz_files is None:
            fit_input.update(
                {"IsolatedAtom": {"iso_atoms_dir": [isoatoms.output["dirs"]]}}
            )

        for ml_model in self.ml_models:
            add_data_fit = MLIPFitMaker(
                mlip_type=ml_model,
            ).make(
                species_list=isoatoms.output["species"],
                isolated_atom_energies=isoatoms.output["energies"],
                fit_input=fit_input,
                split_ratio=split_ratio,
                force_max=f_max,
                pre_xyz_files=pre_xyz_files,
                pre_database_dir=pre_database_dir,
                atomwise_regularization_param=atomwise_regularization_parameter,
                force_min=f_min,
                atom_wise_regularization=atom_wise_regularization,
                auto_delta=auto_delta,
                apply_data_preprocessing=preprocessing_data,
                **fit_kwargs,
            )
            flows.append(add_data_fit)
            if "regularization" in fit_kwargs and fit_kwargs["regularization"]:
                hyper_list.append(
                    {
                        "f="
                        + str(atomwise_regularization_parameter): "default with sigma"
                    }
                )
            hyper_list.append(
                {"f=" + str(atomwise_regularization_parameter): "default"}
            )
            if "separated" in fit_kwargs and fit_kwargs["separated"]:
                hyper_list.append(
                    {"f=" + str(atomwise_regularization_parameter): "default phonon"}
                )
                hyper_list.append(
                    {"f=" + str(atomwise_regularization_parameter): "default randstruc"}
                )

            if (benchmark_structures is not None) and (benchmark_mp_ids is not None):
                for ibenchmark_structure, benchmark_structure in enumerate(
                    benchmark_structures
                ):
                    complete_bm = complete_benchmark(
                        ibenchmark_structure=ibenchmark_structure,
                        benchmark_structure=benchmark_structure,
                        min_length=self.min_length,
                        ml_model=ml_model,
                        ml_path=add_data_fit.output["mlip_path"],
                        mp_ids=mp_ids,
                        benchmark_mp_ids=benchmark_mp_ids,
                        add_dft_phonon_struct=self.add_dft_phonon_struct,
                        fit_input=fit_input,
                        symprec=self.symprec,
                        phonon_displacement_maker=self.phonon_displacement_maker,
                        dft_references=dft_references,
                        **self.benchmark_kwargs,
                    )
                    flows.append(complete_bm)
                    bm_outputs.append(complete_bm.output)

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
                for atomwise_regularization in self.atomwise_regularization_list:
                    for n_sparse in self.n_sparse_list:
                        for delta in self.soap_delta_list:
                            soap_dict = {
                                "n_sparse": n_sparse,
                                "delta": delta,
                            }
                            loop_data_fit = MLIPFitMaker(mlip_type=ml_model).make(
                                species_list=isoatoms.output["species"],
                                isolated_atom_energies=isoatoms.output["energies"],
                                fit_input=fit_input,
                                split_ratio=split_ratio,
                                force_max=f_max,
                                pre_xyz_files=pre_xyz_files,
                                pre_database_dir=pre_database_dir,
                                atomwise_regularization_param=atomwise_regularization,
                                force_min=f_min,
                                atom_wise_regularization=atom_wise_regularization,
                                auto_delta=auto_delta,
                                soap=soap_dict,
                            )
                            flows.append(loop_data_fit)
                            hyper_list.append(
                                {"f=" + str(atomwise_regularization): soap_dict}
                            )
                            if (benchmark_structures is not None) and (
                                benchmark_mp_ids is not None
                            ):
                                for (
                                    ibenchmark_structure,
                                    benchmark_structure,
                                ) in enumerate(benchmark_structures):
                                    complete_bm = complete_benchmark(
                                        ibenchmark_structure=ibenchmark_structure,
                                        benchmark_structure=benchmark_structure,
                                        min_length=self.min_length,
                                        ml_model=ml_model,
                                        ml_path=loop_data_fit.output["mlip_path"],
                                        mp_ids=mp_ids,
                                        benchmark_mp_ids=benchmark_mp_ids,
                                        add_dft_phonon_struct=self.add_dft_phonon_struct,
                                        fit_input=fit_input,
                                        symprec=self.symprec,
                                        phonon_displacement_maker=self.phonon_displacement_maker,
                                        dft_references=dft_references,
                                        **self.benchmark_kwargs,
                                    )
                                    flows.append(complete_bm)
                                    bm_outputs.append(complete_bm.output)
        collect_bm = write_benchmark_metrics(
            ml_models=self.ml_models,
            benchmark_structures=benchmark_structures,
            benchmark_mp_ids=benchmark_mp_ids,
            metrics=bm_outputs,
            displacements=self.displacements,
            hyper_list=hyper_list,
        )
        flows.append(collect_bm)

        return Flow(jobs=flows, output=collect_bm, name=self.name)

    def add_dft_phonons(
        self,
        structure: Structure,
        displacements: list[float],
        symprec: float,
        phonon_displacement_maker: BaseVaspMaker,
        min_length: float,
        adaptive_phonopy_supercell_settings: bool = True,
    ):
        """Add DFT phonon runs for reference structures.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        displacements: list[float]
           displacement distance for phonons
        symprec: float
            Symmetry precision to use in the
            reduction of symmetry to find the primitive/conventional cell
            (use_primitive_standard_structure, use_conventional_standard_structure)
            and to handle all symmetry-related tasks in phonopy
        phonon_displacement_maker: BaseVaspMaker
            Maker used to compute the forces for a supercell.
        min_length: float
             min length of the supercell that will be built
        adaptive_phonopy_supercell_settings: bool
            prevent too tight phonopy supercell settings
        """
        additonal_dft_phonon = dft_phonopy_gen_data(
            structure,
            displacements,
            symprec,
            phonon_displacement_maker,
            min_length,
            adaptive_phonopy_supercell_settings,
        )

        return Flow(
            jobs=additonal_dft_phonon,  # flows
            output={
                "phonon_dir": additonal_dft_phonon.output["dirs"],
                "phonon_data": additonal_dft_phonon.output["data"],
            },
            name=self.name,
        )

    def add_dft_random(
        self,
        structure: Structure,
        mp_id: str,
        phonon_displacement_maker: BaseVaspMaker,
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
        """Add DFT phonon runs for randomly displaced structures.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        mp_id: str
            materials project id
        phonon_displacement_maker: BaseVaspMaker
            Maker used to compute the forces for a supercell.
        uc: bool.
            If True, will generate randomly distorted structures (unitcells)
            and add static computation jobs to the flow.
        supercell_matrix: Matrix3D or None
            The matrix to construct the supercell.
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
        additonal_dft_random = dft_random_gen_data(
            structure=structure,
            mp_id=mp_id,
            phonon_displacement_maker=phonon_displacement_maker,
            n_structures=n_structures,
            uc=uc,
            volume_custom_scale_factors=volume_custom_scale_factors,
            volume_scale_factor_range=volume_scale_factor_range,
            rattle_std=rattle_std,
            supercell_matrix=supercell_matrix,
            distort_type=distort_type,
            rattle_seed=rattle_seed,
            rattle_mc_n_iter=rattle_mc_n_iter,
            rattle_type=rattle_type,
            angle_max_attempts=angle_max_attempts,
            angle_percentage_scale=angle_percentage_scale,
            w_angle=w_angle,
            min_distance=min_distance,
            adaptive_rattled_supercell_settings=adaptive_rattled_supercell_settings,
        )
        return Flow(
            jobs=additonal_dft_random,
            output={"rand_struc_dir": additonal_dft_random.output},
            name=self.name,
        )
