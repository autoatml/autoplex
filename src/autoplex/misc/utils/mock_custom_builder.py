"""Simple example generator for amorphous structures.

Generates structures according to a formula and target density using a random hard sphere approach.
Used for testing the `CustomRandomizedStructure` maker. Note that this script deliberately does NOT
create any jobs or flows, as it is designed to mimic an external tool similar to `buildcell`.
"""

import random
import sys

import ase
import ase.data
import ase.formula
import ase.io
import ase.units
import numpy as np


def mock_generate_structure():
    """Generate a unit cell and populate it at random.

    Creates an empty unit cell atomic and then populates it with
    randomly positioned carbon atoms according to a target number
    of atoms (`population`) and the target average `density`.

    Notes
    -----
    The function will calculate the required cell size based on this and
    the sum of the atomic masses, using `calculate_cell_size`
    """
    SLACK = 0.1
    MIN_SEPARATION = 2.36 * 0.9

    formula = ase.formula.Formula(sys.argv[1], strict=True)
    density = float(sys.argv[2])

    lattice_param = calculate_cell_size(formula, density, slack=SLACK)
    structure = ase.Atoms(cell=(lattice_param, lattice_param, lattice_param), pbc=True)

    for element in formula:
        add_atom(structure, element, MIN_SEPARATION)

    ase.io.write("-", structure, format="xyz")


def add_atom(structure: ase.Atoms, element: str, min_separation: float) -> None:
    """Find a suitable position for a new atom.

    Picks a location within the simulation cell at random, ensures no other atoms are present
    within a fixed radius (`min_separation`), and adds an atom with symbol `element` to `structure`.
    If another atom is found within the close contact radius, the candidate atom is removed and
    the search process loops automatically.

    Parameters
    ----------
    structure : ase.Atoms
        The cell to place the new atom in
    element : str
        The chemical symbol for the new atom to be added
    min_separation: float
        The minimum separation to be kept between all atoms (in Ang)
    """
    safely_placed = False
    lattice_params = structure.cell.cellpar()

    while not safely_placed:

        new_atom = ase.Atom(
            element,
            position=(
                random.uniform(0, lattice_params[0]),
                random.uniform(0, lattice_params[1]),
                random.uniform(0, lattice_params[2]),
            ),
        )

        structure.append(new_atom)
        close_contact_detected = False

        for atom in structure[:-1]:
            dist = structure.get_distance(atom.index, structure[-1].index, mic=True)

            if dist < min_separation:
                close_contact_detected = True
                structure.pop()
                break

        if not close_contact_detected:
            safely_placed = True


def calculate_cell_size(
    formula: ase.formula.Formula, density: float, slack: float = 0.0
) -> float:
    """Calculate the lattice parameter for the simulation cell.

    Calculates the total mass of the given composition (`formula`),
    and divides by the `density` (given in g/cm^3) to give a cubic lattice parameter.
    A percentage variation from `density` can be specified (`slack') to generate
    similuar cells of different volumes (useful if creating a convex hull, for instance).

    Parameters
    ----------
    formula : ase.formula.Formula
        The chemical formula of the target compositions
    density : float
        The target average density of the unit cell, in g/cm^
    slack : float
        How much the cell should be allowed to deviate from the specified `density`. If greater
        than 0, actual density will be selected according to
        `random.uniform(density * (1 - slack), density * (1 + slack))`. Default is 0.0

    Returns
    -------
    float
        The lattice parameter (in Ang)
    """
    if slack > 0.0:
        density = random.uniform(density * (1 - slack), density * (1 + slack))

    population_mass = 0
    for element in formula:
        atomic_num = ase.data.atomic_numbers[element]
        population_mass += ase.data.atomic_masses[atomic_num]

    atomic_density = density * (ase.units.kg * 1e-3) / ((ase.units.m**3) * 1e-6)

    return np.cbrt(population_mass / atomic_density)


if __name__ == "__main__":
    mock_generate_structure()
