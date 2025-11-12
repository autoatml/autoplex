"""Document Schema for CASTEP.

The following code has been taken and modified from
https://github.com/materialsproject/atomate2/blob/main/src/atomate2/ase/schemas.py

The code has been released under BSD 3-Clause License
and the following copyright applies:
atomate2 Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.

The original schema definitions there haven been taken and generalized to
generic ASE calculators from
https://github.com/materialsvirtuallab/m3gnet
The code has been released under BSD 3-Clause License
and the following copyright applies:
Copyright (c) 2022, Materials Virtual Lab.
"""

from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure


class InputDoc(BaseModel):
    """The inputs used to run this job."""

    input_set: dict | None = Field(
        None, description="Input set describing the input for CASTEP."
    )


class OutputDoc(BaseModel):
    """The outputs of this job."""

    structure: Structure | None = Field(
        None, description="Final output structure from the task."
    )

    energy: float | None = Field(None, description="Total energy in units of eV.")

    energy_per_atom: float | None = Field(
        None,
        description="Energy per atom of the final molecule or structure "
        "in units of eV/atom.",
    )

    forces: list[Vector3D] | None = Field(
        None,
        description=(
            "The force on each atom in units of eV/A for the final molecule "
            "or structure."
        ),
    )

    # NOTE: units for stresses were converted to kbar (* -10 from standard output)
    #       to comply with MP convention
    stress: Matrix3D | None = Field(
        None, description="The stress on the cell in units of kbar."
    )


class TaskDoc(StructureMetadata):
    """Document containing information on structure manipulation using CASTEP."""

    structure: Structure = Field(
        None, description="Final output structure from the task"
    )

    input: InputDoc = Field(
        None, description="The input information used to run this job."
    )

    output: OutputDoc = Field(None, description="The output information from this job.")

    task_label: str = Field(
        None,
        description="Description of the CASTEP task (e.g., static, relax)",
    )

    dir_name: str | None = Field(
        None, description="Directory where the ASE calculations are performed."
    )
