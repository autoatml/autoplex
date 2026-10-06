"""Jobs to collect NMR data (shielding and EFG tensors) from CASTEP outputs."""

import contextlib
import logging
import os
from pathlib import Path

from ase.io import read, write
from jobflow.core.job import job

with contextlib.suppress(ImportError):
    pass

from autoplex.data.common.jobs import (
    check_convergence_castep,
    safe_strip_hostname,
)


@job
def collect_nmr_data(
    nmr_ref_file: str = "nmr_ref.extxyz",
    nmr_dirs: list | None = None,
) -> dict:
    """
    Collect NMR data from specified directories.

    Parameters
    ----------
    nmr_ref_file : str
        Reference file for NMR-labelled data. Default is 'nmr_ref.extxyz'.

    nmr_dirs : list
        List of directories containing NMR data

    Returns
    -------
    dict:
        A dictionary containing

        - 'nmr_ref_dir': Directory of the nmr reference file.
    """
    if nmr_dirs is None:
        raise ValueError("nmr_dirs must be provided")

    dirs = [safe_strip_hostname(value) for value in nmr_dirs]

    logging.info("Attempting collecting NMR...")

    if dirs is None:
        raise ValueError("Dirs is None!")

    structures = []

    for val in dirs:
        structure = None
        has_magres_output = os.path.exists(os.path.join(val, "castep.magres.gz"))
        has_castep_output = os.path.exists(os.path.join(val, "castep.castep.gz"))

        if has_magres_output and has_castep_output:
            converged = check_convergence_castep(os.path.join(val, "castep.castep.gz"))
            if converged:
                structure = read(os.path.join(val, "castep.magres.gz"), format="magres")
                # unsure if there are cases where energy does not converge and MS and EFG tensors converge
            else:
                logging.warning(
                    f"Calculation did not converge for path: {os.path.join(val, 'castep.castep.gz')}"
                )

        else:
            logging.warning(
                f"There is no castep.magres and/or castep.castep output at directory {val}!"
            )

        if structure is not None:
            n = len(structure)
            if "ms" in structure.arrays:
                structure.arrays["REF_ms"] = structure.arrays.pop("ms").reshape(n, 9)

                # extxyz cannot store 3x3 tensors, reshape to (Nx9)
            else:
                logging.warning(
                    f"There appears to be no magnetic shielding tensor at the structure in directory {val}!"
                )

            if "efg" in structure.arrays:
                structure.arrays["REF_efg"] = structure.arrays.pop("efg").reshape(n, 9)

            structures.append(structure)

    logging.info(f"Total {len(structures)} structures are exactly collected.")

    write(nmr_ref_file, structures, format="extxyz", parallel=False)

    dir_path = Path.cwd()

    nmr_ref_dir = os.path.join(dir_path, nmr_ref_file)

    return {"nmr_ref_dir": nmr_ref_dir}
