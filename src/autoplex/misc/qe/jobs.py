from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from jobflow import Flow, Maker

from .run import run_qe_static
from .schema import QeKpointsSettings, QeRunSettings
from .utils import QeStaticInputGenerator


@dataclass
class QEStaticMaker(Maker):
    """
    StaticMaker for Quantum ESPRESSO:
    - assemble and write one .pwi per structure using InputGenerator;
    - create a `run_qe_static` job for each input;
    - assemble flow with all jobs.

    Parameters
    --------------------
    name : str
        Name of the Flow.
    command : str
        Command to execute QE (e.g. "pw.x" or "mpirun -np 4 pw.x -nk 2").
    template_pwi : str | None
        Path to template `.pwi` with namelists: &control, &system, &electrons.
    structures : str | list[str] | None
        ASE-readables file (or list of files) containing the structures.
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
    template_pwi: Optional[str] = None
    structures: Optional[str | List[str]] = None
    workdir: Optional[str] = None
    run_settings: Optional[QeRunSettings] = None
    kpoints: Optional[QeKpointsSettings] = None
    pseudo: Optional[dict[str, str]] = None

    def make(self) -> Flow:
        workdir = self.workdir or os.path.join(os.getcwd(), "qe_static")
        os.makedirs(workdir, exist_ok=True)

        # Generate one input per structure
        generator = QeStaticInputGenerator(
            template_pwi=self.template_pwi,
            run_settings=self.run_settings or QeRunSettings(),
            kpoints=self.kpoints or QeKpointsSettings(),
            pseudo=self.pseudo,
        )
        input_sets = generator.generate_for_structures(
            structures=self.structures, workdir=workdir, seed_prefix="structure"
        )

        # Create one SCF job per structure
        jobs = []
        outputs = []
        for i, inp in enumerate(input_sets):
            j = run_qe_static(pwi_path=inp.pwi_path, command=self.command)
            j.name = f"{self.name}_{i}"
            jobs.append(j)
            outputs.append(j.output)

        return Flow(jobs=jobs, output=outputs, name=self.name)