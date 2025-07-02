"""Flows for running RSS."""

from dataclasses import dataclass

from jobflow import Flow, Maker, Response, job

from autoplex.data.common.jobs import (
    sample_data,
)
from autoplex.data.rss.jobs import RandomizedStructure

__all__ = ["BuildMultiRandomizedStructure"]


@dataclass
class BuildMultiRandomizedStructure(Maker):
    """
    Maker to create random structures by 'buildcell'.

    Parameters
    ----------
    tag: str
        Tag of systems. It can also be used for setting up elements and stoichiometry.
        For example, the tag of 'SiO2' will be recognized as a 1:2 ratio of Si to O and
        passed into the parameters of buildcell. However, note that this will be overwritten
        if the stoichiometric ratio of elements is defined in the 'cell_seed_paths' or 'buildcell_options'.
    generated_struct_numbers: list[int]
        Expected number of generated randomized unit cells.
    cell_seed_paths: list[str]
        A list of paths to the custom buildcell control files, which ends with '.cell'. If these files exist,
        the buildcell_options argument will no longer take effect.
    buildcell_options: list[dict]
        A list of customized buildcell parameters.
    fragment_file: Atoms | list[Atoms] (optional)
        Fragment(s) for random structures, e.g. molecules, to be placed indivudally intact.
        atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
        atoms.cell must be defined (e.g. Atoms.cell = np.eye(3)*20).
    fragment_numbers: list[str] (optional)
        Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
    remove_tmp_files: bool
        Remove all temporary files raised by buildcell to save memory.
    initial_selection_enabled: bool
        If true, sample structures using CUR.
    selected_struct_numbers: list
        Number of structures to be sampled.
    bcur_params: dict
        Parameters for Boltzmann CUR selection.
    random_seed: int
        A seed to ensure reproducibility of CUR selection.
    num_processes: int
        Number of processes to use for parallel computation.
    name: str
        Name of the flows produced by this maker.

    """

    tag: str
    generated_struct_numbers: list[int]
    cell_seed_paths: list[str] | None = None
    buildcell_options: list[dict] | None = None
    fragment_file: str | None = None
    fragment_numbers: list[str] | None = None
    remove_tmp_files: bool = True
    initial_selection_enabled: bool = False
    selected_struct_numbers: list[int] | None = None
    bcur_params: dict | None = None
    random_seed: int | None = None
    num_processes: int = 1
    name: str = "do_randomized_structure_generation"

    @job
    def make(self):
        """Maker to create random structures by buildcell."""
        job_list = []
        final_structures = []
        for i, struct_number in enumerate(self.generated_struct_numbers):
            cell_seed_path = None
            buildcell_option = None
            if self.cell_seed_paths is not None:
                assert len(self.generated_struct_numbers) == len(self.cell_seed_paths)
                cell_seed_path = self.cell_seed_paths[i]
            elif self.buildcell_options is not None:
                assert len(self.generated_struct_numbers) == len(self.buildcell_options)
                buildcell_option = self.buildcell_options[i]
            job_struct = RandomizedStructure(
                tag=self.tag,
                struct_number=struct_number,
                remove_tmp_files=self.remove_tmp_files,
                cell_seed_path=cell_seed_path,
                buildcell_option=buildcell_option,
                fragment_file=self.fragment_file,
                fragment_numbers=self.fragment_numbers,
                num_processes=self.num_processes,
            ).make()
            job_struct.name = f"{self.name}_{i}"

            if self.initial_selection_enabled:
                assert len(self.generated_struct_numbers) == len(
                    self.selected_struct_numbers
                )
                job_cur = sample_data(
                    selection_method="cur",
                    num_of_selection=self.selected_struct_numbers[i],
                    bcur_params=self.bcur_params,
                    dir=job_struct.output,
                    random_seed=self.random_seed,
                )
                job_cur.name = f"sampling_{i}"
                job_list.append(job_struct)
                job_list.append(job_cur)
                final_structures.append(job_cur.output)
            else:
                job_list.append(job_struct)
                final_structures.append(job_struct.output)

        return Response(
            replace=Flow(job_list),
            output=final_structures,
        )
