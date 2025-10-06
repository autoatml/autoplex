from __future__ import annotations

import os
from dataclasses import dataclass

from ase import Atoms
from pymatgen.core import Structure

from jobflow import Flow, Maker

from .run import run_qe_static
from .schema import QeKpointsSettings, QeRunSettings
from .utils import QeStaticInputGenerator


@dataclass
class QeStaticMaker(Maker):
    """
    StaticMaker for Quantum ESPRESSO:
    - assemble and write one .pwi per structure using InputGenerator;
    - create a `run_qe_static` job for each input;
    - assemble flow with all jobs.

    Parameters
    ----------
    name : str
        Name of the Flow.
    command : str
        Command to execute QE (e.g. "pw.x" or "mpirun -np 4 pw.x -nk 2").
    workdir : str | None
        Directory used to write input/output files. Default: "<cwd>/qe_static".
    run_settings : QeRunSettings | None
        Update namelists (&control, &system, &electrons).
    kpoints : QeKpointsSettings | None
        Set up for K_POINTS if it is not contained in the template.
    pseudo : dict[str, str] | None
        Dictionary of atomic symbols and corresponding pseudopotential files.
    """

    name: str = "qe_static"
    command: str = "pw.x"
    workdir: str | None = None
    run_settings: QeRunSettings | None = None
    kpoints: QeKpointsSettings | None = None
    pseudo: dict[str, str] | None = None

    def make(
        self,
        structures: Atoms | list[Atoms] | Structure | list[Structure] | str | list[str]
        ) -> Flow:
        """
        Create a Flow to run static SCF calculations with QE for given structures.

        Parameters
        ----------
        structures : Atoms | list[Atoms] | Structure | list[Structure] | str | list[str]
            Single or list of ASE Atoms, pymatgen Structures, or ASE-readable files.
        
        Returns
        -------
        Flow
            A jobflow Flow with one `run_qe_static` job per structure.
        """
        workdir = self.workdir or os.path.join(os.getcwd(), "qe_static")
        os.makedirs(workdir, exist_ok=True)

        # Generate one input per structure
        generator = QeStaticInputGenerator(
            run_settings=self.run_settings or QeRunSettings(),
            kpoints=self.kpoints or QeKpointsSettings(),
            pseudo=self.pseudo or {},
        )
        input_sets = generator.generate_for_structures(
            structures=structures, workdir=workdir, seed_prefix="structure"
        )

        # If single structure, generate one job
        if len(input_sets) == 1:
            job = run_qe_static(input_sets[0], command=self.command)
            job.name = self.name
            return job

        # Else create one SCF job per structure and assemble flow
        jobs = []
        tasks = []
        for i, inp in enumerate(input_sets):
            j = run_qe_static(inp, command=self.command)
            j.name = f"{self.name}_{i}"
            jobs.append(j)
            tasks.append(j.output)

        return Flow(jobs=jobs, output=tasks, name=self.name)