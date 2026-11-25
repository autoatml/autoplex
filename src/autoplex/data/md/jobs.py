"""Jobs for running MD."""

import os
from pathlib import Path

from jobflow import job


@job
def collect_md_trajs(md_outputs: dict, traj_file: str | Path = "MD.traj") -> list[str]:
    """
    Collect molecular dynamics (MD) trajectory file paths from multiple job outputs.

    Parameters
    ----------
    md_outputs: dict
        A dictionary mapping job identifiers (e.g., "md_job_0", "md_job_1", ...)
        to their output objects. Each output object must have a `dir_name` attribute
        that points to the directory containing the MD results.
    traj_file: str | Path
        Name of the trajectory file each MD job wrote (its `traj_file`).

    Returns
    -------
    list[str]
        Absolute paths to the trajectory file of each MD job.
    """
    return [os.path.join(out.dir_name, traj_file) for out in md_outputs.values()]
