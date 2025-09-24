"""CASTEP job makers"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ase.calculators.castep import Castep
from ase.io import read
from jobflow import Maker, job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.misc.castep.run import run_castep
from autoplex.misc.castep.utils import (
    CastepInputGenerator,
    CastepStaticSetGenerator,
    gzip_castep_outputs,
)


@dataclass
class BaseCastepMaker(Maker):
    """
    Base CASTEP job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : CastepInputGenerator
        Generator used to create the CASTEP input set,
        including .param and .cell settings.
    pspot: str | None
        Path to store pseudopotentials.
    """

    name: str = "castep_job"
    input_set_generator: CastepInputGenerator = field(
        default_factory=CastepInputGenerator
    )
    pspot: str | None = None

    @job
    def make(self, structure: Structure):
        """
        Run a CASTEP calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.

        Returns
        -------
        output: dict
        """
        input_set = self.input_set_generator.get_input_set(structure)

        atoms = AseAtomsAdaptor().get_atoms(structure)

        calc = Castep(
            keyword_tolerance=0,
            _prepare_input_only=True,
            _copy_pspots=True,
        )

        atoms.calc = calc

        for key, value in input_set["param"].items():
            setattr(atoms.calc.param, key, value)

        for key, value in input_set["cell"].items():
            setattr(atoms.calc.cell, key, value)

        if self.pspot:
            atoms.set_pspot(self.pspot)

        calc.initialize()
        run_castep(calc)

        workdir = os.path.join(os.getcwd(), "CASTEP")
        atoms = read(os.path.join(workdir, "castep.castep"))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        gzip_castep_outputs(workdir=workdir)

        return {
            "energy": energy,
            "forces": forces,
            "structure": structure,
            "dir_name": workdir,
            "task_label": self.name,
        }


@dataclass
class CastepStaticMaker(BaseCastepMaker):
    """
    Maker to create CASTEP static (single-point energy) jobs.

    This class creates static energy calculations using CASTEP,
    similar to VASP StaticMaker in atomate2.

    Parameters
    ----------
    name : str
        The job name (default: "static").
    input_set_generator : CastepInputGenerator
        Generator used to create the CASTEP input set,
        including .param and .cell settings.
        (default: CastepStaticSetGenerator()).
    """

    name: str = "static"
    input_set_generator: CastepInputGenerator = field(
        default_factory=CastepStaticSetGenerator
    )
