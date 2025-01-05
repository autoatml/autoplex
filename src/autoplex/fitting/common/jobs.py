"""General fitting jobs using several MLIPs available."""

from pathlib import Path

import numpy as np
from jobflow import job

from autoplex.fitting.common.utils import (
    check_convergence,
    gap_fitting,
    jace_fitting,
    m3gnet_fitting,
    mace_fitting,
    nequip_fitting,
)

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "mlip-phonon-defaults.json"


@job
def machine_learning_fit(
    database_dir: str | Path,
    species_list: list,
    run_fits_on_different_cluster: bool = False,
    path_to_hyperparameters: Path | str | None = None,
    isolated_atom_energies: dict | None = None,
    num_processes_fit: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    glue_file_path: str = "glue.xml",
    mlip_type: str | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    use_defaults: bool = True,
    device: str = "cuda",
    database_dict: dict | None = None,
    hyperpara_opt: bool = False,
    **fit_kwargs,
):
    """
    Job for fitting potential(s).

    Parameters
    ----------
    database_dir: Str | Path
        Path to the directory containing the database.
    species_list: list
        List of element names (strings) involved in the training dataset
    run_fit_on_different_cluster: bool
        Whether to run fitting on different clusters.
    path_to_hyperparameters : str or Path.
        Path to JSON file containing the MLIP hyperparameters.
    isolated_atom_energies: dict
        Dictionary of isolated atoms energies.
    num_processes_fit: int
        Number of processes for fitting.
    auto_delta: bool
        Automatically determine delta for 2b, 3b and soap terms.
    glue_xml: bool
        Use the glue.xml core potential instead of fitting 2b terms.
    glue_file_path: str
        Name of the glue.xml file path.
    mlip_type: str
        Choose one specific MLIP type to be fitted:
        'GAP' | 'J-ACE' | 'NEQUIP' | 'M3GNET' | 'MACE'
    ref_energy_name: str
        Reference energy name.
    ref_force_name: str
        Reference force name.
    ref_virial_name: str
        Reference virial name.
    use_defaults: bool
        If True, use default fitting parameters
    device: str
        Device to be used for model fitting, either "cpu" or "cuda".
    database_dict: dict
        Dict including all training and test databases.
    hyperpara_opt: bool
        Perform hyperparameter optimization using XPOT
        (XPOT: https://pubs.aip.org/aip/jcp/article/159/2/024803/2901815)
    fit_kwargs: dict
        Additional keyword arguments for MLIP fitting.
    """
    if run_fits_on_different_cluster:
        from ase.io import write
        from pymatgen.io.ase import AseAtomsAdaptor

        adapter = AseAtomsAdaptor()
        for key, values in database_dict.items():
            if values is not None:
                if not Path(key).parent.exists():
                    Path(key).parent.mkdir(parents=True, exist_ok=True)

                for value in values:
                    properties = value.properties.copy()
                    properties["REF_virial"] = np.array(properties["REF_virial"])
                    value.properties = properties
                    new_value = adapter.get_atoms(value)
                    write(key, new_value, parallel=False, format="extxyz", append=True)
        database_dir = Path().cwd()

    else:
        if isinstance(database_dir, str):  # data_prep_job.output is returned as string
            database_dir = Path(database_dir)

    train_files = [
        "train.extxyz",
        "without_regularization/train.extxyz",
        "phonon/train.extxyz",
        "rattled/train.extxyz",
    ]
    test_files = [
        "test.extxyz",
        "without_regularization/test.extxyz",
        "phonon/test.extxyz",
        "rattled/test.extxyz",
    ]

    mlip_paths = []

    if mlip_type == "GAP":
        for train_name, test_name in zip(train_files, test_files):
            if (database_dir / train_name).exists() and (
                database_dir / test_name
            ).exists():
                train_test_error = gap_fitting(
                    db_dir=database_dir,
                    path_to_hyperparameters=path_to_hyperparameters,
                    species_list=species_list,
                    num_processes_fit=num_processes_fit,
                    auto_delta=auto_delta,
                    glue_xml=glue_xml,
                    glue_file_path=glue_file_path,
                    ref_energy_name=ref_energy_name,
                    ref_force_name=ref_force_name,
                    ref_virial_name=ref_virial_name,
                    train_name=train_name,
                    test_name=test_name,
                    fit_kwargs=fit_kwargs,
                )
                mlip_paths.append(train_test_error["mlip_path"])

    elif mlip_type == "J-ACE":
        train_test_error = jace_fitting(
            db_dir=database_dir,
            path_to_hyperparameters=path_to_hyperparameters,
            isolated_atom_energies=isolated_atom_energies,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            num_processes_fit=num_processes_fit,
            fit_kwargs=fit_kwargs,
        )
        mlip_paths.append(train_test_error["mlip_path"])

    elif mlip_type == "NEQUIP":
        train_test_error = nequip_fitting(
            db_dir=database_dir,
            path_to_hyperparameters=path_to_hyperparameters,
            isolated_atom_energies=isolated_atom_energies,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            fit_kwargs=fit_kwargs,
            device=device,
        )
        mlip_paths.append(train_test_error["mlip_path"])

    elif mlip_type == "M3GNET":
        train_test_error = m3gnet_fitting(
            db_dir=database_dir,
            path_to_hyperparameters=path_to_hyperparameters,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            fit_kwargs=fit_kwargs,
            device=device,
        )
        mlip_paths.append(train_test_error["mlip_path"])

    elif mlip_type == "MACE":
        train_test_error = mace_fitting(
            db_dir=database_dir,
            path_to_hyperparameters=path_to_hyperparameters,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            use_defaults=use_defaults,
            device=device,
            fit_kwargs=fit_kwargs,
        )
        mlip_paths.append(train_test_error["mlip_path"])

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": mlip_paths,
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
        "database_dir": database_dir,
    }
