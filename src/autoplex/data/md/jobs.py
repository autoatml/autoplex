"""Jobs for running MD."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from atomate2.forcefields.md import ForceFieldMDMaker
from emmet.core.math import Matrix3D
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure

from autoplex.data.common.utils import scale_cell
from autoplex.data.md.utils import generate_temperature_profile

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class MDAseMaker(ForceFieldMDMaker):
    """
    Maker to perform molecular dynamics (MD) simulations using ASE.

    This class extends the `ForceFieldMDMaker` from atomate2, with all its
    standard keywords preserved. For reference, see:
    https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/md.py

    In addition, this MD maker supports running **any combination of quenching
    (cooling)** and **heating** stages.  By specifying multiple temperature points,
    the maker automatically builds a stepwise temperature path, enabling complex
    thermal protocols such as:
        • isothermal equilibration
        • multi-stage quench (e.g., 3000 -> 1500 -> 300 K)
        • annealing or cyclic heating-cooling loops

    This maker provides a flexible way to design and execute
    temperature-controlled MD workflows directly within atomate2.

    Parameters
    ----------
    temperature_list : list[float]
        A list of values defines a multi-stage quench or anneal profile.
    eqm_step_list : list[int] | None, optional
        Number of MD steps to hold each temperature value.
        Length must match `temperatures`. If None, defaults to 10,000.
    rate_list : list[float] | None, optional
        Relative cooling/heating rates between stages (len = len(temperatures) - 1).
        If None, linear interpolation is used. A larger `rate` value produces more
        intermediate temperatures (slower quench), while a smaller value gives fewer steps
        (faster quench). For example, rate=10 -> 10^14 K/s; rate=100 -> 10^13 K/s.
    name : str
        Job name.
    volume_scale_factor_range: list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume -+10%.
    supercell_matrix: Matrix3D.
        Matrix for obtaining the supercell.

    Returns
    -------
    list[Path]
        A list of absolute file paths to the "MD.traj" files from each job output.
    """

    temperature_list: list[float] = field(default_factory=list)
    eqm_step_list: list[int] | None = None
    rate_list: list[int] | None = None
    name: str = "ML-driven_MD_with_ASE"
    volume_custom_scale_factors: list[float] | None = None
    supercell_matrix: Matrix3D | None = None

    @job
    def make(self, structure: Structure):
        """Maker to run MD simulations.

        Parameters
        ----------
        structure: Structure
            Pymatgen structure.
        """
        if self.supercell_matrix is not None:
            logging.info(f"Applying supercell_matrix:\n{self.supercell_matrix}")
            structure = structure * self.supercell_matrix
            logging.info(f"Supercell generated: {structure}.")

        if self.volume_custom_scale_factors is not None:
            n_structures = len(self.volume_custom_scale_factors)
            logging.info("Applying custom volume scaling:")
            logging.info(f"  Scale factors: {self.volume_custom_scale_factors}")
            logging.info(f"  Number of scaled structures to generate: {n_structures}")
            structure = scale_cell(
                structure=structure,
                n_structures=n_structures,
                volume_custom_scale_factors=self.volume_custom_scale_factors,
            )

        temps, n_steps = generate_temperature_profile(
            temperature_list=self.temperature_list,
            eqm_step_list=self.eqm_step_list,
            rate_list=self.rate_list,
        )
        self.temperature = temps
        self.n_steps = int(n_steps)

        job_list = []
        job_output = {}
        structures = [structure] if isinstance(structure, Structure) else structure
        for idx, struct in enumerate(structures):
            md_job = ForceFieldMDMaker.make(self, structure=struct)
            job_list.append(md_job)
            job_output[f"md_job_{idx}"] = md_job.output

        collec_trajs = collect_md_trajs(job_output)
        job_list.append(collec_trajs)

        return Response(replace=Flow(job_list), output=collec_trajs.output)


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
