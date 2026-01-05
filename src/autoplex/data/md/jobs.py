"""Jobs for running MD."""

import logging
import os
from pathlib import Path

from jobflow import job

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@job
def collect_md_trajs(md_outputs: dict) -> list[Path]:
    """
    Collect molecular dynamics (MD) trajectory file paths from multiple job outputs.

    Parameters
    ----------
    md_outputs: dict
        A dictionary mapping job identifiers (e.g., "md_job_0", "md_job_1", ...)
        to their output objects. Each output object must have a `dir_name` attribute
        that points to the directory containing the MD results.

    Returns
    -------
    list[Path]
        A list of absolute file paths to the "MD.traj" files from each job output.
    """
    return [os.path.join(out.dir_name, "MD.traj") for out in md_outputs.values()]
