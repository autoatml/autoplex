"""CASTEP flows makers."""

from dataclasses import dataclass, field

from jobflow import Flow, Maker, Response
from pymatgen.core import Structure

from autoplex.misc.castep.jobs import CastepMagresMaker


@dataclass
class CastepMagresFlowMaker(Maker):
    """
    Run CASTEP magres (NMR) calculations on a list of structures.

    Parameters
    ----------
    name: str
        Name of the flow
    magres_maker: CastepMagresMaker
        Job maker for task: magres
    """

    name: str = "castep_magres_flow"
    magres_maker: CastepMagresMaker = field(default_factory=CastepMagresMaker)

    def make(self, structures: list[Structure]) -> Flow:
        """
        Run a flow consisting of jobs for each structure in structures.

        Parameters
        ----------
        structures: list[Structure]
            List of structures.

        Returns
        -------
        List of directories for each .castep and .magres file
        """
        dirs = []
        job_list = []
        for i, structure in enumerate(structures):
            job = self.magres_maker.make(structure=structure)
            job.name = f"{self.magres_maker.name}_{i + 1}"
            dirs.append(job.output.dir_name)
            job_list.append(job)

        return Response(replace=Flow(job_list), output=dirs)
