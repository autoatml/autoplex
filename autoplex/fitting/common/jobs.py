"""General fitting jobs using several MLIPs available."""
from __future__ import annotations

from pathlib import Path

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
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def machine_learning_fit(
    database_dir: str | Path,
    species_list: list,
    isolated_atoms_energies: dict | None = None,
    num_processes_fit: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    mlip_type: str | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    device: str = "cuda",
    hyper_param_optimization: bool = False,
    **fit_kwargs,
):
    """
    Job for fitting potential(s).

    Parameters
    ----------
    database_dir: str | Path
        the database directory.
    isolated_atoms_energies: dict | None
        Dict of isolated atoms energies.
    num_processes_fit: int
        number of processes for fitting.
    auto_delta: bool
        automatically determine delta for 2b, 3b and SOAP terms.
    glue_xml: bool
        use the glue.xml core potential instead of fitting 2b terms.
    mlip_type: str
        Choose one specific MLIP type:
        'GAP' | 'J-ACE' | 'P-ACE' | 'NEQUIP' | 'M3GNET' | 'MACE'
    species_list : list.
            List of element names (str)
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    device: str
        specify device to use cuda or cpu.
    hyper_param_optimization: bool
        call hyperparameter optimization (HPO) or not
    fit_kwargs : dict.
            dict including more fit keyword args.
    """
    if isinstance(database_dir, str):  # data_prep_job.output is returned as string
        database_dir = Path(database_dir)

    train_files = [
        "train.extxyz",
        "train_wo_sigma.extxyz",
        "train_phonon.extxyz",
        "train_rand_struc.extxyz",
    ]
    test_files = [
        "test.extxyz",
        "test.extxyz",
        "test_phonon.extxyz",
        "test_rand_struc.extxyz",
    ]

    if mlip_type == "GAP":
        for train_name, test_name in zip(train_files, test_files):
            if (database_dir / train_name).exists() and (
                database_dir / test_name
            ).exists():
                train_test_error = gap_fitting(
                    db_dir=database_dir,
                    species_list=species_list,
                    num_processes_fit=num_processes_fit,
                    auto_delta=auto_delta,
                    glue_xml=glue_xml,
                    ref_energy_name=ref_energy_name,
                    ref_force_name=ref_force_name,
                    ref_virial_name=ref_virial_name,
                    train_name=train_name,
                    test_name=test_name,
                    fit_kwargs=fit_kwargs,
                )

    elif mlip_type == "J-ACE":
        train_test_error = jace_fitting(
            db_dir=database_dir,
            isolated_atoms_energies=isolated_atoms_energies,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            num_processes_fit=num_processes_fit,
            fit_kwargs=fit_kwargs,
        )

    elif mlip_type == "NEQUIP":
        train_test_error = nequip_fitting(
            db_dir=database_dir,
            isolated_atoms_energies=isolated_atoms_energies,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            fit_kwargs=fit_kwargs,
            device=device,
        )

    elif mlip_type == "M3GNET":
        train_test_error = m3gnet_fitting(
            db_dir=database_dir,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            fit_kwargs=fit_kwargs,
            device=device,
        )

    elif mlip_type == "MACE":
        train_test_error = mace_fitting(
            db_dir=database_dir,
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
            fit_kwargs=fit_kwargs,
            device=device,
        )

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": train_test_error["mlip_path"],
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }
