"""Jobs to create training data for ML potentials."""

from jobflow import job
from pymatgen.core.structure import Structure

from autoplex.data.phonons.utils import reduce_supercell_size


@job
def reduce_supercell_size_job(
    structure: Structure,
    min_length: float = 18,
    max_length: float = 20,
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
        The pymatgen Structure object.
    min_length: float
        Minimum length of the supercell that will be built.
    max_length: float
        Maximum length of the supercell that will be built.
    max_atoms: int
        Maximally allowed number of atoms in the supercell.
    min_atoms: int
        Minimum number of atoms in the supercell that shall be reached.
    fallback_min_length: float
        Fallback option for minimum length for exceptional cases.
    step_size: float
        The step_size which is used to increase the supercell.
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
