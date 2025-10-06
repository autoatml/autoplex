from __future__ import annotations

from typing import Dict, List, Optional

from pymatgen.core import Structure
from emmet.core.structure import StructureMetadata
from emmet.core.math import Matrix3D, Vector3D

from pydantic import BaseModel, Field, field_validator


class QeRunSettings(BaseModel):
    """
    Set QE namelists for static calculation SCF
    Standard namelists: CONTROL, SYSTEM, ELECTRONS
    """
    control: Dict[str, object] = Field(default_factory=dict)
    system: Dict[str, object] = Field(default_factory=dict)
    electrons: Dict[str, object] = Field(default_factory=dict)

    @field_validator("control", "system", "electrons")
    @classmethod
    def _lowercase_keys(cls, v: Dict[str, object]) -> Dict[str, object]:
        # default lowercase keywords
        return {str(k).lower(): v[k] for k in v}


class QeKpointsSettings(BaseModel):
    """
    K-points use k-space resoultion with automatic Monkhorst-Pack
    if K_POINTS are not defined in reference 'template.pwi'
    """
    kspace_resolution: Optional[float] = None  # angstrom^-1
    koffset: List[bool] = Field(default_factory=lambda: [False, False, False])

    @field_validator("koffset")
    @classmethod
    def _len3(cls, v: List[bool]) -> List[bool]:
        if len(v) != 3:
            raise ValueError("koffset must be a list of 3 booleans.")
        return v


class InputDoc(BaseModel):
    """
    Inputs and contexts used to run the static SCF job
    """
    workdir: str
    pwi_path: str
    seed: str
    run_settings: QeRunSettings = Field(None, description="QE namelist section with: &control, &system, &electrons")
    kpoints: QeKpointsSettings = Field(None, description="QE K_POINTS settings")


class OutputDoc(BaseModel):
    """
    The outputs of this jobs
    """
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
    """Document containing information on structure manipulation using Quantum ESPRESSO."""

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
        None, description="Directory where the QE calculations are performed."
    )    