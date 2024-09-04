"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jobflow import Response, job
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@job
def reduce_supercell_size(
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
    list
        supercell matrix.
    """
    for minimum in range(int(min_length), int(fallback_min_length), -1):
        try:
            transformation = CubicSupercellTransformation(
                min_length=minimum,
                max_length=max_length,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                step_size=step_size,
                allow_orthorhombic=True,
                force_90_degrees=True,
            )
            new_structure = transformation.apply_transformation(structure=structure)
            if min_atoms <= new_structure.num_sites <= max_atoms:
                return transformation.transformation_matrix.transpose().tolist()
        except AttributeError:
            try:
                transformation = CubicSupercellTransformation(
                    min_length=minimum,
                    max_length=max_length,
                    min_atoms=min_atoms,
                    max_atoms=max_atoms,
                    step_size=step_size,
                    allow_orthorhombic=True,
                    force_90_degrees=False,
                )
                new_structure = transformation.apply_transformation(structure=structure)
                if min_atoms <= new_structure.num_sites <= max_atoms:
                    return transformation.transformation_matrix.transpose().tolist()
            except AttributeError:
                try:
                    transformation = CubicSupercellTransformation(
                        min_length=minimum,
                        max_length=max_length,
                        min_atoms=min_atoms,
                        max_atoms=max_atoms,
                        step_size=step_size,
                    )
                    new_structure = transformation.apply_transformation(
                        structure=structure
                    )
                    if min_atoms <= new_structure.num_sites <= max_atoms:
                        return transformation.transformation_matrix.transpose().tolist()
                except AttributeError:
                    pass

    a, b, c = structure.lattice.abc
    a_factor = np.max((np.floor(max_length / a), 1))
    b_factor = np.max((np.floor(max_length / b), 1))
    c_factor = np.max((np.floor(max_length / c), 1))

    matrix = np.array([[a_factor, 0, 0], [0, b_factor, 0], [0, 0, c_factor]])
    return Response(output=matrix.transpose().tolist())
