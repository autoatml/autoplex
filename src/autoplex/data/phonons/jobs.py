"""Jobs to create training data for ML potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

from autoplex.data.phonons.utils import reduce_supercell_size


@job
def reduce_supercell_size_job(
    structure: Structure,
    min_length: float = 18,
    max_length: float = 22,
    fallback_min_length: float = 12,
    min_atoms: int = 100,
    max_atoms: int = 500,
    step_size: float = 1,
):
    """
    Reduce phonopy supercell size.

    Parameters
    ----------
    structure: Structure
        pymatgen Structure object.
    min_length: float
        min length of the supercell that will be built.
    max_length: float
        max length of the supercell that will be built.
    max_atoms: int
        maximally allowed number of atoms in the supercell.
    min_atoms: int
        minimum number of atoms in the supercell that shall be reached.
    fallback_min_length: float
        fallback option for minimum length for exceptional cases.
    step_size: float
        step_size which is used to increase the supercell.
        If allow_orthorhombic and force_90_degrees are both set to True,
        the chosen step_size will be automatically multiplied by 5 to
        prevent a too long search for the possible supercell.

    Returns
    -------
    reduced_supercell_size call.
    """
    return reduce_supercell_size(
        structure=structure,
        min_length=min_length,
        max_length=max_length,
        fallback_min_length=fallback_min_length,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        step_size=step_size,
    )
