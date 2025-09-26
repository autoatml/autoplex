from __future__ import annotations

import gzip
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pymatgen.core import Structure


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

    CONFIG = {
        "PARAM": {
            "task": "SinglePoint",
            "calculate_stress": "True",
        }
    }
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
            "max_scf_cycles": 100,
            "smearing_width": 0.05,
            "write_checkpoint": "none",
            "num_dump_cycles": 0,
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
        updates = {
            "kpoints_mp_spacing": "0.04",
        }
        return updates


def gzip_castep_outputs(
    workdir: str | Path | None = None,
    glob_patterns: List[str] | None = None,
    remove_originals: bool = True,
) -> List[str]:
    """
    Gzip CASTEP input/output files to save disk space.

    Parameters
    ----------
    workdir : str | Path | None
        Directory to search for files.
    glob_patterns : list[str] | None
        Glob patterns for files to compress.
    remove_originals : bool
        Whether to remove original files after compression.
    """
    workdir = Path(workdir or os.getcwd())
    if glob_patterns is None:
        _CASTEP_INPUT_FILES = [
            "*.cell",
            "*.param",
            "*.usp",
            "*.recpot",
            "castep_keywords.json",
        ]

        _CASTEP_OUTPUT_FILES = [
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

        glob_patterns = _CASTEP_INPUT_FILES + _CASTEP_OUTPUT_FILES

    for pat in glob_patterns:
        for filepath in workdir.glob(pat):
            if not filepath.is_file():
                continue
            gz_path = filepath.with_suffix(filepath.suffix + ".gz")
            with open(filepath, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            if remove_originals and os.path.isfile(filepath):
                os.remove(filepath)
