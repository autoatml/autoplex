"""Definition of base CASTEP job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.tasks import TaskDoc
from jobflow import Maker, Response, job
from monty.serialization import dumpfn
from pymatgen.core.trajectory import Trajectory

from atomate2 import SETTINGS
from atomate2.common.files import gzip_output_folder

from autoplex.castep.utils import CastepTaskDoc
from datetime import datetime

from ase.calculators.castep import Castep
import shutil
import os

from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
import tempfile
import time

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
    cut_off_energy : float
        Plane wave cutoff energy in eV.
    kspacing : float
        K-point spacing.
    xc_functional : str
        Exchange-correlation functional.
    task : str
        CASTEP task type.
    run_castep_kwargs : dict
        Keyword arguments for running CASTEP.
    task_document_kwargs : dict
        Keyword arguments for creating TaskDoc.
    """
    name: str = "castep_job"
    castep_kwargs: dict | None = None
    pspot: string | None = None

    @job
    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> Response:
        """
        Run a CASTEP calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous CASTEP calculation directory to copy output files from.

        Returns
        -------
        Response
            A response object containing the output of the CASTEP run.
        """

        # Convert structure to ASE atoms
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
    
        
        # Create CASTEP calculator
        atoms.calc = Castep()

        if self.castep_kwargs:
            for key, value in self.castep_kwargs.items():
                setattr(atoms.calc.param, key, value)

        if self.pspot:
            atoms.set_pspot(self.pspot)

        output = {'energy': atoms.get_potential_energy(),
        'volume' : atoms.get_volume(),
        'forces' : atoms.get_forces(),
        'lattice_parameters' : atoms.get_cell_lengths_and_angles(),
        'directory' : os.getcwd()
        }
        return output
            


        
        
   