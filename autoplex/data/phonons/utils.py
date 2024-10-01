"""Utility functions for data generation jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.forcefields.jobs import (
        ForceFieldRelaxMaker,
        ForceFieldStaticMaker,
    )
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
    from pymatgen.core import Structure
import logging

import numpy as np

from autoplex.data.phonons.jobs import reduce_supercell_size

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ml_phonon_maker_preparation(
    calculator_kwargs: dict,
    relax_maker_kwargs: dict | None,
    static_maker_kwargs: dict | None,
    bulk_relax_maker: ForceFieldRelaxMaker,
    phonon_displacement_maker: ForceFieldStaticMaker,
    static_energy_maker: ForceFieldStaticMaker,
) -> tuple[
    ForceFieldRelaxMaker | None,
    ForceFieldStaticMaker | None,
    ForceFieldStaticMaker | None,
]:
    """
    Prepare the MLPhononMaker for the respective MLIP model.

    bulk_relax_maker: .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker: .ForceFieldStaticMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    phonon_displacement_maker: .ForceFieldStaticMaker or None
        Maker used to compute the forces for a supercell.
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    """
    if bulk_relax_maker is not None:
        bulk_relax_maker = bulk_relax_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if relax_maker_kwargs is not None:
            bulk_relax_maker = bulk_relax_maker.update_kwargs(
                update={**relax_maker_kwargs}
            )

    if phonon_displacement_maker is not None:
        phonon_displacement_maker = phonon_displacement_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if static_maker_kwargs is not None:
            phonon_displacement_maker = phonon_displacement_maker.update_kwargs(
                {**static_maker_kwargs}
            )
    if static_energy_maker is not None:
        static_energy_maker = static_energy_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if static_maker_kwargs is not None:
            static_energy_maker = static_energy_maker.update_kwargs(
                update={**static_maker_kwargs}
            )

    return bulk_relax_maker, phonon_displacement_maker, static_energy_maker


def update_phonon_displacement_maker(
    lattice, phonon_displacement_maker
) -> PhononDisplacementMaker:
    """
    Update the phonon_displacement_maker.

    Parameters
    ----------
    lattice:
        (Average) lattice of the structure.
    phonon_displacement_maker:
        Maker used to compute the forces for a supercell.

    Returns
    -------
    Updated phonon_displacement_maker

    """
    if lattice > 10:
        density = 350 - 15 * int(round(lattice, 0))
        if lattice > 20:
            density = 50
        phonon_displacement_maker.input_set_generator.user_kpoints_settings = {
            "reciprocal_density": density
        }
    return phonon_displacement_maker


def check_supercells(
    structure_list: list[Structure],
    structure_names: list[str] | None = None,
    min_length: float = 18,
    max_length: float = 25,
    fallback_min_length: float = 10,
    min_atoms: int = 100,
    max_atoms: int = 500,
    tolerance: float = 0.1,
):
    """
    Check the supercell size.

    Prints log output regarding the structures matching the supercell requirements.

    Parameters
    ----------
    structure_list: list[Structure]
        list of pymatgen Structure object.
    structure_names: list[str]
        list of structure names.
    min_length: float
        min length of the supercell that will be built.
    max_length: float
        max length of the supercell that will be built.
    max_atoms: int
        maximally allowed number of atoms in the supercell.
    min_atoms: int
        minimum number of atoms in the supercell that shall be reached.
    fallback_min_length: float
        fallback option for minimum length for exceptional cases
    tolerance: float
        tolerance for min_atoms and max_atoms

    """
    structure_names = (
        [structure.composition.reduced_formula for structure in structure_list]
        if structure_names is None
        else structure_names
    )

    min_tolerance = 1 - tolerance
    max_tolerance = 1 + tolerance

    for name, structure in zip(structure_names, structure_list):
        matrix = reduce_supercell_size.original(
            structure,
            min_length=min_length,
            max_length=max_length,
            fallback_min_length=fallback_min_length,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )
        supercell = structure.make_supercell(np.array(matrix).transpose())
        a, b, c = supercell.lattice.abc
        num_atoms = supercell.num_sites

        # check if supercells are in the requirements with a certain tolerance
        if (
            not (min_atoms * min_tolerance <= num_atoms <= max_atoms * max_tolerance)
            or (
                not fallback_min_length * min_tolerance
                <= a
                < max_length * max_tolerance
            )
            or (
                not fallback_min_length * min_tolerance
                <= b
                < max_length * max_tolerance
            )
            or (
                not fallback_min_length * min_tolerance
                <= c
                < max_length * max_tolerance
            )
        ):
            logger.warning("You should not include structure %s \n", name)
            logger.info(
                "because the found supercell has the following lattice parameters: %f, %f, %f \n",
                a,
                b,
                c,
            )
            logger.info("and it has the following sites: %d \n", num_atoms)
            logger.info(
                "which usually leads to convergence issues during the DFT steps."
            )
        else:
            logger.info("%s has passed the supercell check. \n", name)
