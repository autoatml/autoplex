"""Utils for CASTEP."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from pymatgen.core import Structure

CASTEP_INPUT_FILES = [
    "*.cell",
    "*.param",
    "*.usp",
    "*.recpot",
    "castep_keywords.json",
]

CASTEP_OUTPUT_FILES = [
    "*.castep",
    "*.castep_bin",
    "*.cst_esp",
    "*.check",
    "*.geom",
    "*.md",
    "*.bands",
    "*.bib",
    "*.phonon",
    "*.elf",
    "*.chdiff",
    "*.den_fmt",
    "*.pot_fmt",
    "*.wvfn_fmt",
    "final_atoms_object.xyz",
    "final_atoms_object.traj",
]


@dataclass
class CastepInputGenerator:
    """
    Base class for CASTEP input set generation.

    It is used to manage both .param and .cell file settings for
    CASTEP calculations.

    Parameters
    ----------
    structure : Structure | None
        The crystal structure for the calculation
    config_dict : dict
        Base configuration dictionary with default CASTEP settings
    user_param_settings : dict
        User-specified .param file settings (equivalent to VASP INCAR)
    user_cell_settings : dict
        User-specified .cell file settings
    sort_structure : bool
        Whether to sort atoms by electronegativity before calculation
    """

    structure: Structure | None = None
    config_dict: dict = field(default_factory=dict)
    user_param_settings: dict = field(default_factory=dict)
    user_cell_settings: dict = field(default_factory=dict)
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
        else:
            raise ValueError("Structure must be provided")

        param_settings = dict(self._config_dict.get("PARAM", {}))
        cell_settings = dict(self._config_dict.get("CELL", {}))

        param_settings.update(self.param_updates)
        cell_settings.update(self.cell_updates)

        param_settings.update(self.user_param_settings)
        cell_settings.update(self.user_cell_settings)

        return {
            "param": param_settings,
            "cell": cell_settings,
            "structure": self.structure,
        }


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

    CONFIG: dict = field(
        default_factory=lambda: {
            "PARAM": {
                "task": "SinglePoint",
                "calculate_stress": "True",
            }
        }
    )
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
            "cut_off_energy": 520.0,
            "xc_functional": "PBE",
            "elec_energy_tol": 1e-06,
            "max_scf_cycles": 1000,
            "smearing_width": 0.05,
            "finite_basis_corr": "automatic",
            "mixing_scheme": "Pulay",
            "mix_charge_amp": 0.6,
            "perc_extra_bands": 60.0,
            "num_dump_cycles": 0,
            "write_checkpoint": "none",
        }

        if self.lepsilon:
            updates.update({"calculate_epsilon": True})

        if self.lcalcpol:
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
        return {
            "kpoints_mp_spacing": "0.03",
        }
