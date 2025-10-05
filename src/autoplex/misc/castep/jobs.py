"""CASTEP job makers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.castep import (
    Castep,
    CastepKeywords,
    make_cell_dict,
    make_param_dict,
)
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa
from atomate2.common.files import gzip_files
from jobflow import Maker, job
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.misc.castep.run import run_castep
from autoplex.misc.castep.schema import InputDoc, OutputDoc, TaskDoc
from autoplex.misc.castep.utils import (
    CASTEP_INPUT_FILES,
    CASTEP_OUTPUT_FILES,
    CastepInputGenerator,
    CastepStaticSetGenerator,
)
from autoplex.settings import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import Structure

# add larger objects to the database in the future, e.g., band structures
_DATA_OBJECTS = []
_FILES_TO_ZIP = [*CASTEP_INPUT_FILES, *CASTEP_OUTPUT_FILES]


def castep_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of CASTEP job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all CASTEP jobs. For example, it ensures that large data objects
    are all stored in the atomate2 data store. It also configures the output schema
    to be a CASTEP :obj:`.TaskDoc`.

    Any makers that return CASTEP jobs (not flows) should decorate the ``make`` method
    with @castep_job.

    For example:

    .. code-block:: python

        class MyCastepMaker(BaseCastepMaker):
            @castep_job
            def make(structure):
                # code to run CASTEP job.
                pass

    Parameters
    ----------
    method : callable
        A BaseCastepMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate Castep jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=TaskDoc)


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
        """
        The following five lines until 'castep_keywords' are adopted and copied from ASE

        References
        ----------
        *    Title: ASE package from ase.calculators.castep
        *    Date 26/09/2025
        *    Code version: 3.24.0
        *    Availability: https://gitlab.com/ase/
        """
        with open(SETTINGS.CASTEP_KEYWORDS) as fd:
            kwdata = json.load(fd)
        # This is a bit awkward, but it's necessary for backwards compatibility
        param_dict = make_param_dict(kwdata["param"])
        cell_dict = make_cell_dict(kwdata["cell"])

        # This is a bit awkward, but it's necessary for backwards compatibility
        param_dict = make_param_dict(kwdata["param"])
        cell_dict = make_cell_dict(kwdata["cell"])

        castep_keywords = CastepKeywords(
            param_dict,
            cell_dict,
            kwdata["types"],
            kwdata["levels"],
            kwdata["castep_version"],
        )

        calc = Castep(
            keyword_tolerance=0,
            _prepare_input_only=True,
            _copy_pspots=True,
            castep_command=SETTINGS.CASTEP_CMD,
            castep_keywords=castep_keywords,
        )

        atoms.calc = calc

        for key, value in input_set["param"].items():
            setattr(atoms.calc.param, key, value)

        for key, value in input_set["cell"].items():
            setattr(atoms.calc.cell, key, value)

        if self.pspot:
            atoms.set_pspot(self.pspot)

        calc.prepare_input_files(atoms)
        run_castep(calc)

        workdir = os.path.join(os.getcwd(), "CASTEP")
        atoms = read(os.path.join(workdir, "castep.castep"))
        gzip_files(directory=workdir, include_files=_FILES_TO_ZIP, allow_missing=True)

        # should pass the final structure!
        final_structure = AseAtomsAdaptor().get_structure(atoms)
        final_energy = atoms.get_potential_energy()
        # stress= None
        # atoms.get_stress()

        try:
            forces = atoms.get_forces()
        except PropertyNotImplementedError:
            forces = None

        try:
            stress = voigt_6_to_full_3x3_stress(atoms.get_stress() * -10 / GPa)
        except PropertyNotImplementedError:
            stress = None

        return TaskDoc(
            structure=final_structure,
            dir_name=workdir,
            task_label=self.name,
            input=InputDoc(input_set=input_set),
            output=OutputDoc(
                structure=final_structure,
                energy_per_atom=final_energy / len(final_structure),
                energy=final_energy,
                forces=forces,
                stress=stress,
            ),
        )


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
