"""
CASTEP Input Set Generators for Atomate2-style Workflows

This module provides CASTEP input generation classes that follow the same
patterns as VASP input generators in atomate2. It enables creation of
standardized CASTEP calculations with proper parameter management.

Dependencies:
- pymatgen
- ase
- jobflow
- atomate2 (for base classes)

Usage Example:
    from pymatgen.core import Structure

    # Load your structure
    structure = Structure.from_file("POSCAR")

    # Create a static calculation maker
    maker = CastepStaticMaker(
        input_set_generator=CastepStaticSetGenerator(
            user_param_settings={
                "cut_off_energy": "600.0 eV",
                "elec_energy_tol": "1e-07 eV"
            }
        )
    )

    # Run the calculation
    result = maker.make(structure)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

# ASE CASTEP calculator
# Core pymatgen imports
from pymatgen.core import Structure

from autoplex.castep.jobs import BaseCastepMaker

# Jobflow/atomate2 imports (you may need to adjust these based on your setup)
try:
    from jobflow import Maker, job
except ImportError:
    # Fallback for basic functionality without jobflow
    def job(func):
        """Simple decorator fallback if jobflow not available."""
        return func

    class Maker:
        """Simple base class fallback if jobflow not available."""


@dataclass
class CastepInputSet:
    """
    Base class for CASTEP input set generation.

    Similar to VaspInputSet but for CASTEP parameters.
    Manages both .param and .cell file settings for CASTEP calculations.

    Parameters
    ----------
    structure : Structure | None
        The crystal structure for the calculation
    config_dict : dict
        Base configuration dictionary with default CASTEP settings
    files_to_transfer : dict
        Files to copy from previous calculations
    user_param_settings : dict
        User-specified .param file settings (equivalent to VASP INCAR)
    user_cell_settings : dict
        User-specified .cell file settings
    sort_structure : bool
        Whether to sort atoms by electronegativity before calculation
    """

    structure: Structure | None = None
    config_dict: dict = field(default_factory=dict)
    files_to_transfer: dict = field(default_factory=dict)
    user_param_settings: dict = field(
        default_factory=dict
    )  # equivalent to user_incar_settings
    user_cell_settings: dict = field(default_factory=dict)  # CASTEP cell file settings
    sort_structure: bool = True

    def __post_init__(self) -> None:
        """Perform validation and setup after initialization."""
        self.user_param_settings = self.user_param_settings or {}
        self.user_cell_settings = self.user_cell_settings or {}

        if hasattr(self, "CONFIG"):
            self.config_dict = self.CONFIG

        self._config_dict = deepcopy(self.config_dict)

        if not isinstance(self.structure, Structure):
            self._structure: Structure | None = None
        else:
            self.structure = self.structure

    @property
    def structure(self) -> Structure | None:
        """Get the structure."""
        return self._structure

    @structure.setter
    def structure(self, structure: Structure | None) -> None:
        """Set the structure with optional sorting."""
        if isinstance(structure, Structure):
            if self.sort_structure:
                structure = structure.get_sorted_structure()
        self._structure = structure

    @property
    def param_updates(self) -> dict:
        """
        Updates to the PARAM config for this calculation type.

        Override this method in subclasses to define calculation-specific
        parameter settings.

        Returns
        -------
        dict
            Dictionary of CASTEP .param file parameters
        """
        return {}

    @property
    def cell_updates(self) -> dict:
        """
        Updates to the CELL config for this calculation type.

        Override this method in subclasses to define calculation-specific
        cell settings.

        Returns
        -------
        dict
            Dictionary of CASTEP .cell file parameters
        """
        return {}

    def get_input_set(self, structure: Structure | None = None) -> dict:
        """
        Generate CASTEP input set as dictionary.

        Parameters
        ----------
        structure : Structure | None
            Structure to use for calculation. If None, uses self.structure

        Returns
        -------
        dict
            Dictionary containing 'param', 'cell', and 'structure' keys

        Raises
        ------
        ValueError
            If no structure is available
        """
        if structure is not None:
            self.structure = structure

        if self.structure is None:
            raise ValueError("Structure must be provided")

        # Start with base config
        param_settings = dict(self._config_dict.get("PARAM", {}))
        cell_settings = dict(self._config_dict.get("CELL", {}))

        # Apply updates from input set generator
        param_settings.update(self.param_updates)
        cell_settings.update(self.cell_updates)

        # Apply user settings (highest priority)
        param_settings.update(self.user_param_settings)
        cell_settings.update(self.user_cell_settings)

        return {
            "param": param_settings,
            "cell": cell_settings,
            "structure": self.structure,
        }


# Alias for consistency with atomate2 naming convention
CastepInputGenerator = CastepInputSet


@dataclass
class CastepStaticSetGenerator(CastepInputGenerator):
    """
    Class to generate CASTEP static (single-point) input sets.

    This class creates input parameters for CASTEP static energy calculations,
    similar to VASP StaticSetGenerator.

    Parameters
    ----------
    lepsilon : bool
        Whether to calculate dielectric properties (similar to VASP LEPSILON)
    lcalcpol : bool
        Whether to calculate polarization (similar to VASP LCALCPOL)
    **kwargs
        Other keyword arguments passed to CastepInputGenerator
    """

    lepsilon: bool = False
    lcalcpol: bool = False

    @property
    def param_updates(self) -> dict:
        """
        Get updates to the PARAM for a static CASTEP job.

        Returns
        -------
        dict
            Dictionary of CASTEP .param file parameters for static calculations
        """
        updates = {
            "task": "singlepoint",  # Static calculation
            "cut_off_energy": "520.0 eV",  # Equivalent to VASP ENCUT
            "basis_precision": "precise",  # Equivalent to VASP PREC: Accurate
            "xc_functional": "PBE",  # Exchange-correlation functional
            "elec_energy_tol": "1e-06 eV",  # Equivalent to VASP EDIFF
            "max_scf_cycles": 100,  # Equivalent to VASP NELM
            "smearing_width": "0.01 eV",  # Equivalent to VASP SIGMA
            "write_checkpoint": "none",  # Equivalent to VASP LWAVE: False
            "num_dump_cycles": 0,  # Minimize output files
        }

        if self.lepsilon:
            # Add dielectric calculation settings
            updates.update(
                {
                    "calculate_epsilon": True,
                    "task": "singlepoint",  # Keep as singlepoint but calculate epsilon
                }
            )

        if self.lcalcpol:
            # Add polarization calculation settings
            updates.update({"calculate_polarisation": True})

        return updates

    @property
    def cell_updates(self) -> dict:
        """
        Get updates to the CELL for a static CASTEP job.

        Returns
        -------
        dict
            Dictionary of CASTEP .cell file parameters for static calculations
        """
        updates = {
            "kpoint_mp_grid": "1 1 1",
            "fix_all_cell": True,  # Fixed cell for static calc
        }
        return updates


@dataclass
class CastepStaticMaker(BaseCastepMaker):
    """
    Maker to create CASTEP static (single-point energy) jobs.

    This class creates static energy calculations using CASTEP,
    similar to VASP StaticMaker in atomate2.

    Parameters
    ----------
    name : str
        The job name (default: "static")
    input_set_generator : CastepInputGenerator
        A generator used to make the input set (default: CastepStaticSetGenerator)
    **kwargs
        Other keyword arguments passed to BaseCastepMaker
    """

    name: str = "static"
    input_set_generator: CastepInputGenerator = field(
        default_factory=CastepStaticSetGenerator
    )


# # Example usage and testing functions
# def example_usage():
#     """
#     Demonstrate how to use the CASTEP input generators.
#     """
#     print("CASTEP Input Generator Example Usage")
#     print("=" * 40)

#     # Create a simple cubic structure for testing
#     from pymatgen.core import Lattice

#     lattice = Lattice.cubic(4.0)
#     structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

#     print(f"Created test structure: {structure.formula}")

#     # Example 1: Basic static calculation
#     print("\n1. Basic Static Calculation:")
#     maker = CastepStaticMaker()
#     input_set = maker.input_set_generator.get_input_set(structure)

#     print("PARAM settings:", input_set["param"])
#     print("CELL settings:", input_set["cell"])

#     # Example 2: Custom parameters
#     print("\n2. Custom Parameters:")
#     custom_maker = CastepStaticMaker(
#         input_set_generator=CastepStaticSetGenerator(
#             user_param_settings={
#                 "cut_off_energy": "600.0 eV",
#                 "elec_energy_tol": "1e-07 eV",
#                 "xc_functional": "PBE0"
#             },
#             user_cell_settings={
#                 "kpoint_mp_spacing": "0.15 1/ang"
#             }
#         )
#     )

#     custom_input_set = custom_maker.input_set_generator.get_input_set(structure)
#     print("Custom PARAM settings:", custom_input_set["param"])

#     # Example 3: Dielectric calculation
#     print("\n3. Dielectric Calculation:")
#     dielectric_maker = CastepStaticMaker(
#         input_set_generator=CastepStaticSetGenerator(
#             lepsilon=True,
#             user_param_settings={"cut_off_energy": "700.0 eV"}
#         )
#     )

#     dielectric_input = dielectric_maker.input_set_generator.get_input_set(structure)
#     print("Dielectric PARAM settings:", dielectric_input["param"])


# if __name__ == "__main__":
#     example_usage()
