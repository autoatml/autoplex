from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QeRunSettings(BaseModel):
    """
    Set QE namelists for static calculation SCF
    Standard namelists: CONTROL, SYSTEM, ELECTRONS
    """

    control: dict[str, object] = Field(default_factory=dict)
    system: dict[str, object] = Field(default_factory=dict)
    electrons: dict[str, object] = Field(default_factory=dict)

    @field_validator("control", "system", "electrons")
    @classmethod
    def _lowercase_keys(cls, v: dict[str, object]) -> dict[str, object]:
        # default lowercase keywords
        return {str(k).lower(): v[k] for k in v}


class QeKpointsSettings(BaseModel):
    """
    K-points use k-space resoultion with automatic Monkhorst-Pack
    if K_POINTS are not defined in reference 'template.pwi'
    """

    kspace_resolution: float | None = None  # angstrom^-1
    koffset: list[bool] = Field(default_factory=lambda: [False, False, False])

    @field_validator("koffset")
    @classmethod
    def _len3(cls, v: list[bool]) -> list[bool]:
        if len(v) != 3:
            raise ValueError("koffset must be a list of 3 booleans.")
        return v


class QeInputSet(BaseModel):
    """
    Files and contexts for static SCF job
    """

    workdir: str
    pwi_path: str
    seed: str


class QeRunResult(BaseModel):
    """
    Results of the SCF job
    """

    success: bool
    pwi: str
    pwo: str
    outdir: str
    total_energy_ev: float | None = None
