"""Definition of base CASTEP job maker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ase.calculators.castep import Castep
from jobflow import Maker, job
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.castep.jobs.base import field
from autoplex.castep.utils import CastepTaskDoc
from autoplex.data.castep_support.utils import CastepInputGenerator, Path

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import Structure

# Data objects for CASTEP jobs
_CASTEP_DATA_OBJECTS = [
    Trajectory,
    "final_structure",
    "final_energy",
]

# CASTEP input files
_CASTEP_INPUT_FILES = [
    "*.cell",
    "*.param",
    "*.usp",
    "*.recpot",
    "castep_keywords.json",
]

# CASTEP output files
_CASTEP_OUTPUT_FILES = [
    "*.castep",
    "*.geom",
    "*.md",
    "*.bands",
    "*.phonon",
    "*.elf",
    "*.chdiff",
    "*.den_fmt",
    "*.pot_fmt",
    "*.wvfn_fmt",
    "final_atoms_object.xyz",
    "final_atoms_object.traj",
]

# Files to zip for CASTEP
_CASTEP_FILES_TO_ZIP = _CASTEP_INPUT_FILES + _CASTEP_OUTPUT_FILES


def castep_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of CASTEP job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all CASTEP jobs. It ensures that trajectory and structure data
    are stored appropriately.

    Any makers that return CASTEP jobs should decorate the ``make`` method
    with @castep_job. For example:

    .. code-block:: python

        class MyCastepMaker(BaseCastepMaker):
            @castep_job
            def make(structure):
                # code to run CASTEP job.
                pass

    Parameters
    ----------
    method : callable
        A BaseCastepMaker.make method.

    Returns
    -------
    callable
        A decorated version of the make function that will generate CASTEP jobs.
    """
    return job(method, output_schema=CastepTaskDoc)


@dataclass
class BaseCastepMaker(Maker):
    """
    Base CASTEP job maker.

    Parameters
    ----------
    name : str
        The job name.
    castep_kwargs : dict
        Keyword arguments for running CASTEP.
    pspot: str | None
        Path to store pseudopotentials.
    """

    name: str = "castep_job"
    input_set_generator: CastepInputGenerator = field(default_factory=CastepInputGenerator)
    castep_kwargs: dict | None = None
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

        # Apply param settings
        for key, value in input_set["param"].items():
            setattr(atoms.calc.param, key, value)
            
        # Apply cell settings  
        for key, value in input_set["cell"].items():
            setattr(atoms.calc.cell, key, value)
            
        # Run calculation
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Return results
        return {
            "energy": energy,
            "forces": forces,
            "structure": structure,
            "directory": str(Path.cwd()),
            "task_label": self.name,
        }


        # if self.castep_kwargs:
        #     for key, value in self.castep_kwargs.items():
        #         if key in {"kpoint_mp_grid", "kpoint_mp_offset", "kpoint_mp_spacing"}:
        #             setattr(atoms.calc.cell, key, value)
        #         else:
        #             setattr(atoms.calc.param, key, value)

        # if self.pspot:
        #     atoms.set_pspot(self.pspot)

        # output = {
        #     "energy": atoms.get_potential_energy(),
        #     "volume": atoms.get_volume(),
        #     "forces": atoms.get_forces(),
        #     "lattice_parameters": atoms.get_cell_lengths_and_angles(),
        #     "directory": os.getcwd(),
        # }

        return output
