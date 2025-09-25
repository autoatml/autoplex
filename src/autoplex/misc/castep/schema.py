from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from emmet.core.math import Matrix3D, Vector3D
from atomate2.ase.schemas import IonicStep
class InputDoc(BaseModel):
    """The inputs used to run this job."""

    input_set: dict|None = Field(None, description="Input set describing the input for CASTEP.")

class OutputDoc(BaseModel):
    """The outputs of this job."""

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

    # NOTE: the ionic_steps can also be a dict when these are in blob storage and
    #       retrieved as objects.
    ionic_steps: list[IonicStep] | dict | None = Field(
        None, description="Step-by-step trajectory of the relaxation."
    )

    elapsed_time: float | None = Field(
        None, description="The time taken to run the ASE calculation in seconds."
    )

    n_steps: int | None = Field(
        None, description="total number of steps needed in the relaxation."
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



