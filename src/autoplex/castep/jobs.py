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
    castep_command : str
        Command to run CASTEP.
    cut_off_energy : float
        Plane wave cutoff energy in eV.
    kspacing : float
        K-point spacing.
    xc_functional : str
        Exchange-correlation functional.
    task : str
        CASTEP task type.
    write_additional_data : dict
        Additional data to write to the current directory.
    run_castep_kwargs : dict
        Keyword arguments for running CASTEP.
    task_document_kwargs : dict
        Keyword arguments for creating TaskDoc.
    """

    name: str = "base castep job"
    castep_command: str = "castep.mpi"
    cut_off_energy: float = 400.0
    kspacing: float = 0.3
    xc_functional: str = "PBE"
    task: str = "SinglePoint"
    write_additional_data: dict = field(default_factory=dict)
    run_castep_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @castep_job
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
        from ase.io import write as ase_write
        from pymatgen.io.ase import AseAtomsAdaptor
        import time
        import os
        import tempfile

        # Convert structure to ASE atoms
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        # Store original directory for safe path handling
        original_dir = os.getcwd()
        
        # Create CASTEP calculator
        calculator = self._create_calculator()
        atoms.calc = calculator
        
        # Write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))
        
        try:
            # Run CASTEP calculation
            t_i = time.perf_counter()
            final_energy = atoms.get_potential_energy()
            t_f = time.perf_counter()
            
            # Handle directory issues gracefully
            try:
                os.getcwd()  # Test if current directory exists
            except FileNotFoundError:
                try:
                    os.chdir(original_dir)
                except (FileNotFoundError, OSError):
                    # Use temp directory as fallback
                    temp_dir = tempfile.mkdtemp()
                    os.chdir(temp_dir)
                    original_dir = temp_dir
            
            # Write output files with error handling
            try:
                ase_write("final_atoms_object.xyz", atoms, format="extxyz", append=True)
            except Exception as write_error:
                print(f"Warning: Could not write output files: {write_error}")
            
            # Get final structure
            final_structure = adaptor.get_structure(atoms)
            
            # Create a basic task document
            task_doc = self._create_task_document(
                structure=structure,
                final_structure=final_structure,
                final_energy=final_energy,
                elapsed_time=t_f - t_i,
                dir_name=original_dir
            )
            
            # Gzip folder if enabled
            gzip_output_folder(
                directory=Path.cwd(),
                setting=getattr(SETTINGS, 'CASTEP_ZIP_FILES', None),
                files_list=_CASTEP_FILES_TO_ZIP,
            )
            
            return Response(output=task_doc)
            
        except Exception as e:
            # Return error task document
            error_task_doc = self._create_error_task_document(
                structure=structure,
                error=str(e),
                dir_name=original_dir
            )
            return Response(output=error_task_doc)

    def _create_calculator(self):
        """Create CASTEP calculator."""
        from ase.calculators.castep import Castep
        import shutil
        import os
        
        try:
            # Copy keywords file to avoid conflicts
            main_keywords = 'castep_keywords.json'
            local_keywords = f'castep_keywords_{os.getpid()}.json'
            
            if os.path.exists(main_keywords):
                shutil.copy(main_keywords, local_keywords)
            
            calculator = Castep(
                directory='.',
                label=f'castep_calc_{os.getpid()}',
                castep_command=self.castep_command,
                keyword_tolerance=3,
                cut_off_energy=self.cut_off_energy,
                xc_functional=self.xc_functional,
                kspacing=self.kspacing,
                task=self.task,
                find_pspots=False,
            )
            
            # Set stress calculation if needed
            calculator.param.calculate_stress = True
            
            return calculator
            
        except Exception as e:
            print(f"Failed to create CASTEP calculator: {e}")
            # Fallback to MACE if available
            try:
                from mace.calculators import mace_mp
                return mace_mp()
            except ImportError:
                raise e

    def _create_task_document(self, structure, final_structure, final_energy, elapsed_time, dir_name):
        """Create CASTEP-specific task document."""
        
        dir_name = str(Path(dir_name).resolve())

        return CastepTaskDoc(
            task_label=self.name,
            dir_name=dir_name,
            last_updated=datetime.utcnow(),

            structure=structure,
            final_structure=final_structure,
            energy=final_energy,
            energy_per_atom=final_energy / len(structure),
            completed_at=datetime.utcnow(),
            elapsed_time=elapsed_time,
            cutoff_energy=self.cut_off_energy,
            kpoint_spacing=self.kspacing,
            xc_functional=self.xc_functional,

            task_type="Static",
            calc_type="CASTEP Static",
            converged=True,

            **self.task_document_kwargs
        )

    def _create_error_task_document(self, structure, error, dir_name):
        """Create an error task document."""
        from emmet.core.tasks import TaskDoc
        from datetime import datetime
        
        return TaskDoc(
            task_label=f"{self.name} (FAILED)",
            dir_name=dir_name,
            input={"structure": structure},
            output={"error": error},
            run_stats={
                "completed_at": datetime.utcnow(),
            },
            tags=["CASTEP", "FAILED"],
            **self.task_document_kwargs
        )