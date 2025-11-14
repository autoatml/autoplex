"""Flows for running MD."""

import logging
from dataclasses import dataclass
from typing import Literal

from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure

from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.data.common.jobs import collect_dft_data, sample_data
from autoplex.data.md.jobs import MDAseMaker

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class MDMaker(DFTStaticLabelling):
    """
    Run molecular dynamics (MD) simulations and label selected configurations with DFT.

    Parameters
    ----------
    name: str
        Name of the flow.
    md_maker: MDAseMaker | None
        A maker responsible for performing the actual MD simulation.
    dft_ref_file : str
        Output filename for the generated and labeled configurations.
        Default "dft_md_ref.extxyz".
    config_type : str
        Tag attached to generated structures.  Default "md".
    selection_method : Literal['random', 'uniform']
       Method for selecting samples. Options include:
        - 'random': Random selection.
        - 'uniform': Uniform selection.
    random_seed: int, optional
        Seed for random number generation, ensuring reproducibility of sampling.
    num_of_selection: int
       Number of structures to be sampled.
    remove_traj_files: bool
        Remove all trajectory files raised by MD.
    """

    name: str = "do_md"
    md_maker: MDAseMaker | None = None
    dft_ref_file: str = "dft_md_ref.extxyz"
    config_type: str = "md"
    selection_method: Literal["cur", "random", "uniform"] = "uniform"
    random_seed: int = 42
    num_of_selection: int = 5
    remove_traj_files: bool = False

    @job
    def make(self, structure: Structure):
        """
        Generate and label a set of MD-rattled structures.

        Parameters
        ----------
        structure : Structure
            Input structure to serve as the starting point.

        Returns
        -------
        dict
            A dictionary containing:
            - `pre_database_dir`: Path to the directory with collected DFT data.
            - `isolated_atom_energies`: Mapping of element symbols to their isolated-atom reference
              energies.
        """
        md_job = self.md_maker.make(structure=structure)
        do_data_sampling = sample_data(
            selection_method=self.selection_method,
            num_of_selection=self.num_of_selection,
            traj_path=md_job.output,
            traj_type="md",
            random_seed=self.random_seed,
            remove_traj_files=self.remove_traj_files,
        )
        do_dft_static = DFTStaticLabelling.make(
            self, structures=do_data_sampling.output
        )
        do_data_collection = collect_dft_data(
            dft_ref_file=self.dft_ref_file,
            dft_dirs=do_dft_static.output,
        )
        job_list = [md_job, do_data_sampling, do_dft_static, do_data_collection]

        return Response(
            replace=Flow(job_list),
            output={
                "pre_database_dir": do_data_collection.output,
                "isolated_atom_energies": do_data_collection.output[
                    "isolated_atom_energies"
                ],
            },
        )
