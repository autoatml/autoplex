"""CASTEP job makers"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from ase.calculators.castep import Castep
from jobflow import Maker, job
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.castep.utils import CastepInputGenerator, gzip_castep_outputs
from pymatgen.core import Structure
from pathlib import Path


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

        atoms.calc = Castep(keyword_tolerance=0)

        for key, value in input_set["param"].items():
            setattr(atoms.calc.param, key, value)

        for key, value in input_set["cell"].items():
            setattr(atoms.calc.cell, key, value)
            
        if self.pspot:
            atoms.set_pspot(self.pspot)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        workdir = Path.cwd()
        gzip_castep_outputs(workdir=workdir)
        
        return {
            "energy": energy,
            "forces": forces,
            "structure": structure,
            "directory": workdir,
            "task_label": self.name,
        }
