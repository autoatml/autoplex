"""CASTEP job makers"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ase.calculators.castep import Castep
from ase.io import read
from autoplex.settings import SETTINGS
from jobflow import Maker, job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.misc.castep.run import run_castep
from autoplex.misc.castep.utils import (
    CastepInputGenerator,
    CastepStaticSetGenerator,
    gzip_castep_outputs,
)
from autoplex.misc.castep.schema import TaskDoc, InputDoc, OutputDoc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

# add larger objects to the database in the future, e.g., band structures
_DATA_OBJECTS=[]


from ase.calculators.castep import Castep

def castep_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of CASTEP job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all CASTEP jobs. For example, it ensures that large data objects
     are all stored in the
    atomate2 data store. It also configures the output schema to be a CASTEP
    :obj:`.TaskDoc`.

    Any makers that return CASTEP jobs (not flows) should decorate the ``make`` method
    with @vasp_job. For example:

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
        from ase.calculators.castep import CastepKeywords
        import json
        with open("/home/jgeorge/RZ-Dienste/hpc-user/PycharmProjects/2025_castep_interface/autoplex/src/autoplex/misc/castep_keywords.json") as fd:
            kwdata = json.load(fd)
        from ase.calculators.castep import make_cell_dict, make_param_dict
        # This is a bit awkward, but it's necessary for backwards compatibility
        param_dict = make_param_dict(kwdata['param'])
        cell_dict = make_cell_dict(kwdata['cell'])

        castep_keywords = CastepKeywords(param_dict, cell_dict,
                                         kwdata['types'], kwdata['levels'],
                                         kwdata['castep_version'])

        calc = Castep(
            keyword_tolerance=0,
            _prepare_input_only=True,
            _copy_pspots=True,
            castep_command=SETTINGS.castep_cmd,
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
        gzip_castep_outputs(workdir=workdir)


        # should pass the final structure!
        final_structure = AseAtomsAdaptor().get_structure(atoms)
        final_energy= atoms.get_potential_energy()

        return TaskDoc(
            structure=final_structure,
            dir_name=workdir,
            task_label=self.name,
            input=InputDoc(input_set=input_set),
            output=OutputDoc(structure=final_structure,
                             energy_per_atom=final_energy / len(final_structure),
                             energy=final_energy,
                             forces=atoms.get_forces())
                             #stress=atoms.get_stress()*-10.0,
                             #n_steps=atoms.calc.n_steps)
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
