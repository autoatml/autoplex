"""Flows consisting of jobs to fit ML potentials."""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ase.io
from jobflow import Flow, Maker, job

from autoplex import MLIP_HYPERS
from autoplex.fitting.common.jobs import machine_learning_fit
from autoplex.fitting.common.regularization import set_custom_sigma
from autoplex.fitting.common.utils import (
    get_list_of_vasp_calc_dirs,
    vaspoutput_2_extended_xyz,
    write_after_distillation_data_split,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

__all__ = [
    "DataPreprocessing",
    "MLIPFitMaker",
]


@dataclass
class MLIPFitMaker(Maker):
    """
    Maker to fit ML potentials based on DFT labelled reference data.

    This Maker will filter the provided dataset in a data preprocessing step and then proceed
    with the MLIP fit (default is GAP).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted.
    hyperpara_opt: bool
        Perform hyperparameter optimization using XPOT
        (XPOT: https://pubs.aip.org/aip/jcp/article/159/2/024803/2901815)
    ref_energy_name : str
        Reference energy name.
    ref_force_name : str
        Reference force name.
    ref_virial_name : str
        Reference virial name.
    glue_file_path: str
        Name of the glue.xml file path.
    split_ratio: float
        Ratio to divide the dataset into training and test sets.
        A value of 0.1 means 90% training data and 10% test data
    force_max: float
        Maximum allowed force in the dataset.
    force_min: float
        Minimal force cutoff value for atom-wise regularization.
    regularization: bool
        For using sigma regularization.
    distillation: bool
        For using data distillation.
    separated: bool
        Repeat the fit for each data_type available in the (combined) database.
    pre_xyz_files: list[str] or None
        Names of the pre-database train xyz file and test xyz file.
    pre_database_dir: str or None
        The pre-database directory.
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    atom_wise_regularization: bool
        For including atom-wise regularization.
    auto_delta: bool
        Automatically determine delta for 2b, 3b and soap terms.
    glue_xml: bool
        Use the glue.xml core potential instead of fitting 2b terms.
    num_processes_fit: int
        Number of processes for fitting.
    apply_data_preprocessing: bool
        Determine whether to preprocess the data.
    run_fits_on_different_cluster: bool
        If true, run fits on different clusters.
    """

    name: str = "MLpotentialFit"
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"] = "GAP"
    hyperpara_opt: bool = False
    ref_energy_name: str = "REF_energy"
    ref_force_name: str = "REF_forces"
    ref_virial_name: str = "REF_virial"
    glue_file_path: str = "glue.xml"
    split_ratio: float = 0.4
    force_max: float = 40.0
    force_min: float = 0.01  # unit: eV Å-1
    distillation: bool = True
    separated: bool = False
    pre_xyz_files: list[str] | None = None
    pre_database_dir: str | None = None
    regularization: bool = False  # This is only used for GAP.
    atomwise_regularization_parameter: float = 0.1  # This is only used for GAP.
    atom_wise_regularization: bool = True  # This is only used for GAP.
    auto_delta: bool = False  # This is only used for GAP.
    glue_xml: bool = False  # This is only used for GAP.
    num_processes_fit: int | None = None
    apply_data_preprocessing: bool = True
    run_fits_on_different_cluster: bool = False

    def make(
        self,
        database_dir: Path | str | None = None,
        fit_input: dict | None = None,  # This is specific to phonon workflow
        hyperparameters: MLIP_HYPERS = MLIP_HYPERS,
        species_list: list | None = None,
        isolated_atom_energies: dict | None = None,
        device: str = "cpu",
        **fit_kwargs,
    ):
        """
        Make a flow for fitting MLIP models.

        Parameters
        ----------
        database_dir: Path | str
            Path to the directory containing the database.
        fit_input: dict
            Output from the CompletePhononDFTMLDataGenerationFlow process.
        database_dir: Path | str
            Path to the directory containing the database.
        hyperparameters: MLIP_HYPERS
            Hyperparameters for the MLIP.
        species_list: list
            List of element names (strings) involved in the training dataset
        isolated_atom_energies: dict
            Dictionary of isolated atoms energies.
        device: str
            Device to be used for model fitting, either "cpu" or "cuda".
        fit_kwargs: dict
            Additional keyword arguments for MLIP fitting.
        """
        if self.mlip_type not in ["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]:
            raise ValueError(
                "Please correct the MLIP name!"
                "The current version ONLY supports the following models: GAP, J-ACE, NEP, NEQUIP, M3GNET, and MACE."
            )

        if self.apply_data_preprocessing:
            jobs = []
            data_prep_job = DataPreprocessing(
                split_ratio=self.split_ratio,
                regularization=self.regularization,
                separated=self.separated,
                distillation=self.distillation,
                force_max=self.force_max,
                pre_xyz_files=self.pre_xyz_files,
                pre_database_dir=self.pre_database_dir,
                force_min=self.force_min,
                ref_virial_name=self.ref_virial_name,
                ref_force_name=self.ref_force_name,
                ref_energy_name=self.ref_energy_name,
                atomwise_regularization_parameter=self.atomwise_regularization_parameter,
                atom_wise_regularization=self.atom_wise_regularization,
                run_fits_on_different_cluster=self.run_fits_on_different_cluster,
            ).make(
                fit_input=fit_input,
            )
            jobs.append(data_prep_job)

            mlip_fit_job = machine_learning_fit(
                database_dir=data_prep_job.output["database_dir"],
                run_fits_on_different_cluster=self.run_fits_on_different_cluster,
                isolated_atom_energies=isolated_atom_energies,
                num_processes_fit=self.num_processes_fit,
                auto_delta=self.auto_delta,
                glue_xml=self.glue_xml,
                glue_file_path=self.glue_file_path,
                mlip_type=self.mlip_type,
                hyperpara_opt=self.hyperpara_opt,
                hyperparameters=hyperparameters,
                ref_energy_name=self.ref_energy_name,
                ref_force_name=self.ref_force_name,
                ref_virial_name=self.ref_virial_name,
                device=device,
                species_list=species_list,
                database_dict=data_prep_job.output["database_dict"],
                **fit_kwargs,
            )
            jobs.append(mlip_fit_job)
            output = {
                "mlip_path": mlip_fit_job.output["mlip_path"],
                "train_error": mlip_fit_job.output["train_error"],
                "test_error": mlip_fit_job.output["test_error"],
                "convergence": mlip_fit_job.output["convergence"],
                "database_dir": data_prep_job.output["database_dir"],
            }
            return Flow(jobs=jobs, output=output, name=self.name)

        # this will only run if train.extxyz and test.extxyz files are present in the database_dir
        # TODO: shouldn't this be the exception rather then the default run?!
        # TODO: I assume we always want to use data from before?
        if isinstance(database_dir, str):
            database_dir = Path(database_dir)

        mlip_fit_job = machine_learning_fit(
            database_dir=database_dir,
            isolated_atom_energies=isolated_atom_energies,
            num_processes_fit=self.num_processes_fit,
            auto_delta=self.auto_delta,
            glue_xml=self.glue_xml,
            glue_file_path=self.glue_file_path,
            mlip_type=self.mlip_type,
            hyperpara_opt=self.hyperpara_opt,
            hyperparameters=hyperparameters,
            ref_energy_name=self.ref_energy_name,
            ref_force_name=self.ref_force_name,
            ref_virial_name=self.ref_virial_name,
            device=device,
            species_list=species_list,
            **fit_kwargs,
        )

        output = {
            "mlip_path": mlip_fit_job.output["mlip_path"],
            "train_error": mlip_fit_job.output["train_error"],
            "test_error": mlip_fit_job.output["test_error"],
            "convergence": mlip_fit_job.output["convergence"],
            "database_dir": self.pre_database_dir,
        }

        return Flow(jobs=mlip_fit_job, output=output, name=self.name)


@dataclass
class DataPreprocessing(Maker):
    """
    Data preprocessing of the provided dataset.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    split_ratio: float
        Parameter to divide the training set and the test set.
        A value of 0.1 means that the ratio of the training set to the test set is 9:1
    regularization: bool
        For using sigma regularization.
    separated: bool
        Repeat the fit for each data_type available in the (combined) database.
    distillation: bool
        For using data distillation.
    ref_energy_name : str
        Reference energy name in xyz file.
    ref_force_name : str
        Reference force name in xyz file.
    ref_virial_name : str
        Reference virial name in xyz file.
    force_max: float
        Maximally allowed force in the data set.
    force_min: float
        Minimal force cutoff value for atom-wise regularization.
    pre_database_dir: str or None
        The pre-database directory.
    pre_xyz_files: list[str] or None
        Names of the pre-database train xyz file and test xyz file labelled by VASP.
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    atom_wise_regularization: bool
        If True, includes atom-wise regularization.
    train_data_file: str
        Name of the training xyz data file.
    test_data_file: str
        Name of the test xyz data file.
    run_fits_on_different_cluster: bool
        If True, will copy the fitting database to the MongoDB

    """

    name: str = "data_preprocessing_for_fitting"
    split_ratio: float = 0.5
    regularization: bool = False
    separated: bool = False
    distillation: bool = False
    ref_energy_name: str = "REF_energy"
    ref_force_name: str = "REF_forces"
    ref_virial_name: str = "REF_virial"
    force_max: float = 40.0
    force_min: float = 0.01  # unit: eV Å-1
    pre_database_dir: str | None = None
    pre_xyz_files: list[str] | None = None
    atomwise_regularization_parameter: float = 0.1
    atom_wise_regularization: bool = True
    train_data_file: str = "train.extxyz"
    test_data_file: str = "test.extxyz"
    run_fits_on_different_cluster: bool = False

    @job(data=["database_dict"])
    def make(
        self,
        fit_input: dict,
    ):
        """
        Maker for data preprocessing.

        Parameters
        ----------
        fit_input:
            Mixed list of dictionary and lists of the fit input data.
        """
        if self.pre_xyz_files is None:
            self.pre_xyz_files = ["train.extxyz", "test.extxyz"]

        list_of_vasp_calc_dirs = get_list_of_vasp_calc_dirs(flow_output=fit_input)
        print(fit_input)
        config_types = [
            key
            for key, value in fit_input.items()
            for key2, value2 in value.items()
            if key2 != "phonon_data"
            for _ in value2[0]
        ]

        data_types = [
            key2
            for key, value in fit_input.items()
            for key2, value2 in value.items()
            if key2 != "phonon_data"
            for _ in value2[0]
        ]

        if self.pre_database_dir and os.path.exists(self.pre_database_dir):
            current_working_directory = os.getcwd()

            if len(self.pre_xyz_files) == 1:
                for file_name in self.pre_xyz_files:
                    source_file_path = os.path.join(self.pre_database_dir, file_name)
                    destination_file_path = os.path.join(
                        current_working_directory, "vasp_ref.extxyz"
                    )
                    shutil.copy(source_file_path, destination_file_path)
                    logging.info(
                        f"File {file_name} has been copied to {destination_file_path}"
                    )
            if len(self.pre_xyz_files) == 2:
                # join to one file and then split again afterwards
                # otherwise, split percentage will not be true
                destination_file_path = os.path.join(
                    current_working_directory, "vasp_ref.extxyz"
                )
                for file_name in self.pre_xyz_files:
                    # TODO: if it makes sense to remove isolated atoms from other files as well
                    atoms_list = ase.io.read(
                        os.path.join(self.pre_database_dir, file_name), index=":"
                    )
                    new_atoms_list = [
                        atoms
                        for atoms in atoms_list
                        if atoms.info["config_type"] != "IsolatedAtom"
                    ]

                    ase.io.write(destination_file_path, new_atoms_list, append=True)

                    logging.info(
                        f"File {self.pre_xyz_files[0]} has been copied to {destination_file_path}"
                    )

            elif len(self.pre_xyz_files) > 2:
                raise ValueError(
                    "Please provide a train and a test extxyz file (two files in total) for the pre_xyz_files."
                )

        vaspoutput_2_extended_xyz(
            path_to_vasp_static_calcs=list_of_vasp_calc_dirs,
            config_types=config_types,
            data_types=data_types,
            f_min=self.force_min,
            regularization=self.atomwise_regularization_parameter,
            atom_wise_regularization=self.atom_wise_regularization,
            ref_force_name=self.ref_force_name,
            ref_energy_name=self.ref_energy_name,
            ref_virial_name=self.ref_virial_name,
        )

        write_after_distillation_data_split(
            distillation=self.distillation,
            force_max=self.force_max,
            split_ratio=self.split_ratio,
            force_label=self.ref_force_name,
            energy_label=self.ref_energy_name,
            train_name=self.train_data_file,
            test_name=self.test_data_file,
        )

        # Merging database
        # TODO: does a merge happen here?
        if self.regularization:
            base_dir = os.getcwd()
            folder_name = os.path.join(base_dir, "without_regularization")
            try:
                os.makedirs(folder_name, exist_ok=True)
                logging.info(f"Created/verified folder: {folder_name}")
            except Exception as e:
                logging.warning(f"Error creating folder {folder_name}: {e}")
            train_path = os.path.join(folder_name, self.train_data_file)
            test_path = os.path.join(folder_name, self.test_data_file)
            atoms = ase.io.read(self.train_data_file, index=":")
            ase.io.write(train_path, atoms, format="extxyz")
            logging.info(f"Written train file without regularization to: {train_path}")
            try:
                shutil.copy(self.test_data_file, test_path)
                logging.info(f"Copied test file to: {test_path}")
            except FileNotFoundError:
                logging.warning("test.extxyz not found. Skipping copy.")
            except Exception as e:
                logging.warning(f"Error copying test.extxyz: {e}")
            atoms_with_sigma = set_custom_sigma(
                atoms,
                reg_minmax=[(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)],
            )
            ase.io.write(self.train_data_file, atoms_with_sigma, format="extxyz")
        if self.separated:
            base_dir = os.getcwd()
            atoms_train = ase.io.read(self.train_data_file, index=":")
            atoms_test = ase.io.read(self.test_data_file, index=":")
            for dt in set(data_types):
                data_type = dt.removesuffix("_dir")
                if data_type != "iso_atoms":
                    folder_name = os.path.join(base_dir, data_type)
                    try:
                        os.makedirs(folder_name, exist_ok=True)
                        logging.info(f"Created/verified folder: {folder_name}")
                    except Exception as e:
                        logging.warning(
                            f"Error creating folder {folder_name}: {e}. "
                            f"\nProceeding without separated dataset"
                        )
                        continue
                    vasp_ref_path = os.path.join(folder_name, "vasp_ref.extxyz")
                    train_path = os.path.join(folder_name, self.train_data_file)
                    test_path = os.path.join(folder_name, self.test_data_file)

                    for atoms in atoms_train + atoms_test:
                        if atoms.info["data_type"] == "iso_atoms":
                            ase.io.write(
                                vasp_ref_path,
                                atoms,
                                format="extxyz",
                                append=True,
                            )
                        if atoms.info["data_type"] == data_type:
                            ase.io.write(
                                vasp_ref_path,
                                atoms,
                                format="extxyz",
                                append=True,
                            )
                    try:
                        write_after_distillation_data_split(
                            distillation=self.distillation,
                            force_max=self.force_max,
                            split_ratio=self.split_ratio,
                            vasp_ref_name=vasp_ref_path,
                            train_name=train_path,
                            test_name=test_path,
                            force_label=self.ref_force_name,
                        )
                        logging.info(f"Data split written: {train_path}, {test_path}")
                    except Exception as e:
                        logging.warning(
                            f"Error in write_after_distillation_data_split: {e}"
                        )

        # TODO: add a database to MongoDB besides just the path
        if self.run_fits_on_different_cluster:
            from pymatgen.io.ase import AseAtomsAdaptor

            adapter = AseAtomsAdaptor()

            # must always exist
            required_paths = ["train.extxyz", "test.extxyz"]

            optional_paths = [
                "phonon/train.extxyz",
                "phonon/test.extxyz",
                "rattled/train.extxyz",
                "rattled/test.extxyz",
                "without_regularization/train.extxyz",
                "without_regularization/test.extxyz",
            ]

            database_dict = {
                path: [
                    adapter.get_structure(atoms)
                    for atoms in ase.io.read(Path.cwd() / path, ":")
                ]
                for path in required_paths
            }

            database_dict.update(
                {
                    path: (
                        [
                            adapter.get_structure(atoms)
                            for atoms in ase.io.read(Path.cwd() / path, ":")
                        ]
                        if (Path.cwd() / path).exists()
                        else None
                    )
                    for path in optional_paths
                }
            )

            return {"database_dir": Path.cwd(), "database_dict": database_dict}

        return {"database_dir": Path.cwd(), "database_dict": None}
