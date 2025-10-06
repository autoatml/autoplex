from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms, Atom
from ase.data import atomic_masses, atomic_numbers
from ase.io import read

from .schema import InputDoc, QeKpointsSettings, QeRunSettings

logger = logging.getLogger(__name__)

@dataclass
class QeStaticInputGenerator:
    """
    Input generator to run static SCF calculations with Quantum Espresso (.pwi).
    Used to write one .pwi for each structure inside `workdir`.

    Parameters
    ----------
    run_settings : QeRunSettings | None
        Update namelists (&control, &system, &electrons).
    kpoints : QeKpointsSettings | None
        Set up for K_POINTS.
    pseudo : dict[str, str] | None
        Dictionary of atomic symbols and corresponding pseudopotential files.
    """

    def __init__(
        self,
        run_settings: QeRunSettings | None = None,
        kpoints: QeKpointsSettings | None = None,
        pseudo: dict[str, str] | None = None,
    ) -> None:
        self.run_settings = run_settings or QeRunSettings()
        self.kpoints = kpoints or QeKpointsSettings()
        self.pseudo = pseudo or {}

    def generate_for_structures(
        self,
        structures: Atoms | list[Atoms] | Structure | list[Structure] | str | list[str],
        workdir: str,
        seed_prefix: str = "structure",
    ) -> list[InputDoc]:
        """
        Generate one .pwi for each structure read from ASE-readable files.

        Parameters
        ----------
        structures : Atoms | list[Atoms] | Structure | list[Structure] | str | list[str]
            Single or list of ASE Atoms, pymatgen Structures, or ASE-readable files.
        workdir : str
            Directory used to write input/output files.
        seed_prefix : str
            Prefix for naming input/output QE files.
        
        Returns
        -------
        list[InputDoc]
            List of InputDoc containing path to generated .pwi files and metadata.
        """
        os.makedirs(workdir, exist_ok=True)
        atoms_list = _load_structures(structures)
        input_sets: list[InputDoc] = []

        for i, atoms in enumerate(atoms_list):
            seed = f"{seed_prefix}_{i}"
            pwi_path = os.path.join(workdir, f"{seed}.pwi")
            self._write_pwi(
                pwi_output=pwi_path,
                atoms=atoms,
            )
            input_sets.append(
                InputDoc(
                    workdir=workdir,
                    pwi_path=pwi_path,
                    seed=seed,
                    run_settings=self.run_settings,
                    kpoints=self.kpoints,
                    pseudo=self.pseudo,
            ))
        return input_sets

    def _write_pwi(
        self,
        *,
        pwi_output: str,
        atoms: Atoms,
    ) -> None:
        """
        Write .pwi file for the current Atoms.

        Parameters
        ----------
        pwi_output : str
            Path to output .pwi file.
        atoms : Atoms
            ASE Atoms object.
        
        Raises
        ------
        ValueError
            If required information is missing to write the .pwi file.
        """
        # Minimal template with main QE namelists
        template_lines = _render_minimal_namelists(self.run_settings)
        pwi = list(template_lines)

        # Find indices of key lines in template
        idx_diskio = None
        idx_outdir = None
        idx_nat = None
        idx_ntyp = None
        idx_kpts = None
        for idx, line in enumerate(pwi):
            ls = line.lower()
            if "nat" in ls and "!" not in ls:
                idx_nat = idx
            elif "ntyp" in ls and "!" not in ls:
                idx_ntyp = idx
            elif "disk_io" in ls:
                idx_diskio = idx
            elif "outdir" in ls:
                idx_outdir = idx
            elif "k_points" in ls:
                idx_kpts = idx

        # Number of atoms
        nat = len(atoms)
        if idx_nat is None:
            raise ValueError("Template must define a 'nat = <int>' line (in &system).")
        pwi[idx_nat] = f"   nat = {nat}\n"

        # Output directory
        structure_id = os.path.basename(pwi_output).replace(".pwi", "")
        if idx_diskio is None or "none" not in pwi[idx_diskio].lower():
            if idx_outdir is None:
                insert_at = (idx_diskio + 1) if idx_diskio is not None else len(pwi)
                pwi.insert(insert_at, f"   outdir = '{structure_id}'\n")
            else:
                pwi[idx_outdir] = f"   outdir = '{structure_id}'\n"
        else:
            if idx_outdir is None:
                insert_at = (idx_diskio + 1) if idx_diskio is not None else len(pwi)
                pwi.insert(insert_at, "   outdir = 'OUT'\n")

        # ATOMIC_SPECIES
        if self.pseudo is None:
            raise ValueError(
                "Pseudo dictionary must be provided to write ATOMIC_SPECIES."
            )
        species = set(atoms.get_chemical_symbols())
        ntyp = len(species)
        pwi[idx_ntyp] = f"   ntyp = {ntyp}\n"
        species_lines = ["\nATOMIC_SPECIES\n"]
        for s in sorted(species):
            if s not in self.pseudo:
                raise ValueError(
                    f"Missing pseudo for atomic symbol '{s}' in pseudo dictionary."
                )
            mass = atomic_masses[atomic_numbers[s]]
            species_lines.append(f"{s}  {mass:.4f} {self.pseudo[s]}\n")
        species_lines.append("\n")

        # K-POINTS
        kpt_lines = _render_kpoints(
            template_lines=pwi, idx_kpts=idx_kpts, atoms=atoms, kpoints=self.kpoints
        )
        kpt_lines.append("\n")

        # CELL_PARAMETERS
        cell = atoms.cell
        cell_lines = ["\nCELL_PARAMETERS (angstrom)\n"]
        for i in range(3):
            cell_lines.append(
                f"{cell[i, 0]:.10f} {cell[i, 1]:.10f} {cell[i, 2]:.10f}\n"
            )
        cell_lines.append("\n")

        # ATOMIC_POSITIONS
        pos_lines = ["\nATOMIC_POSITIONS (angstrom)\n"]
        for i, atom in enumerate(atoms):
            x, y, z = atoms.positions[i]
            pos_lines.append(f"{atom.symbol} {x:.10f} {y:.10f} {z:.10f}\n")
        pos_lines.append("\n")

        # Assemble and write
        with open(pwi_output, "w") as fh:
            fh.writelines(pwi)
            fh.writelines(species_lines)
            fh.writelines(kpt_lines)
            fh.writelines(cell_lines)
            fh.writelines(pos_lines)

# --------- Utils ---------


def _read_template(path: str | None) -> list[str]:
    """Read template .pwi file if provided, else return empty list."""
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"template_pwi not found: {path}")
    with open(path) as fh:
        return fh.readlines()


def _load_structures(
    structures: Atoms | list[Atoms] | Structure | list[Structure] | str | list[str]
) -> list[Atoms]:
    """
    Load structures from various input types and return a list of ASE Atoms objects.

    Parameters
    ----------
    structures : Atoms | list[Atoms] | Structure | list[Structure] | str | list[str]
        Single or list of ASE Atoms, pymatgen Structures, or ASE-readable files.
    
    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects.
    """

    # ASE readable files
    # Sinlge ASE-readable file
    if isinstance(structures, str):
        atoms_list = read(structures, index=":")
    # List of ASE readable files
    elif isinstance(structures, list) and all(isinstance(s, str) for s in structures):
        atom_list = []
        for fname in structures:
            atoms_list += read(fname, index=":")


    # ASE Atoms
    elif isinstance(structures, Atoms):
        # List of ASE Atoms objects
        if isinstance(structures[0], Atoms):
            atoms_list = structures
        # Single ASE Atoms
        elif isinstance(structures, Atom):
            atoms_list = [structures]
        else:
            raise ValueError("Unsupported ASE Atoms input.")
    # List of ASE Atoms objects
    elif isinstance(structures, list) and all(isinstance(s, Atoms) for s in structures):
        atoms_list = structures    
                   

    # Pymatgen Structures
    # Single pymatgen structure
    elif isinstance(structures, Structure):
        atoms_list = [AseAtomsAdaptor().get_atoms(structures)]
    # List of pymatgen structures
    elif isinstance(structures, list) and all(isinstance(s, Structure) for s in structures):
        adaptor = AseAtomsAdaptor()
        atoms_list = [adaptor.get_atoms(s) for s in structures]

    
    else:
        raise ValueError("Unsupported type for structures input.")

    return atoms_list


def _render_minimal_namelists(settings: QeRunSettings) -> list[str]:
    """
    Create draft lines for namelists &control, &system, &electrons.
    If some namelist is missing in `settings`, it will be created with minimal entries.

    Parameters
    ----------
    settings : QeRunSettings
        QE namelist settings.
    
    Returns
    -------
    list[str]
        List of lines for the namelists to be written in the .pwi file.
    """

    def _render_block(name: str, kv: dict) -> list[str]:
        lines = [f"&{name}\n"]
        for k, v in kv.items():
            if isinstance(v, bool):
                vv = ".true." if v else ".false."
            elif isinstance(v, (int, float)):
                vv = v
            else:
                vv = f"'{v}'"
            lines.append(f"   {k} = {vv}\n")
        lines.append("/\n\n")
        return lines

    out: list[str] = []
    out += _render_block("control", settings.control)
    out += _render_block("system", settings.system)
    out += _render_block("electrons", settings.electrons)
    return out


def _render_kpoints(
    *,
    template_lines: list[str],
    idx_kpts: int | None,
    atoms: Atoms,
    kpoints: QeKpointsSettings,
) -> list[str]:
    """
    Create lines for K_POINTS section.

    Parameters
    ----------
    template_lines : list[str]
        Lines of the template .pwi file.
    idx_kpts : int | None
        Index of K_POINTS line in template_lines, or None if not present.
    atoms : Atoms
        ASE Atoms object.
    kpoints : QeKpointsSettings
        K-points settings.
    
    Returns
    -------
    list[str]
        Lines for K_POINTS section to be written in the .pwi file.
    """
    # if K_POINTS in template keep it
    if idx_kpts is not None and idx_kpts >= 0:
        line = template_lines[idx_kpts].lower()
        if "gamma" in line:
            kpts = template_lines[idx_kpts : idx_kpts + 1]
        elif "automatic" in line:
            kpts = template_lines[idx_kpts : idx_kpts + 2]
        elif "tpiba" in line or "crystal" in line:
            n = int(template_lines[idx_kpts + 1].split()[0])
            kpts = template_lines[idx_kpts : idx_kpts + n + 2]
        else:
            raise ValueError(
                f"Unknown K_POINTS format: {template_lines[idx_kpts].strip()}"
            )
        del template_lines[idx_kpts:]
        return ["\n"] + kpts

    # Otherwise create K_POINTS using kspace_resolution
    if kpoints.kspace_resolution is None:
        raise ValueError(
            "K_POINTS not found in template and kspace_resolution is None. "
            "Either add K_POINTS to template or set kspace_resolution."
        )

    # Generate Monkhorst–Pack mesh
    mp_mesh = _compute_kpoints_grid(atoms.cell, kpoints.kspace_resolution)
    line = f"{mp_mesh[0]} {mp_mesh[1]} {mp_mesh[2]}"
    for off in kpoints.koffset:
        line += " 1" if off else " 0"
    return ["\nK_POINTS automatic\n", f"{line}\n"]


def _compute_kpoints_grid(cell: np.ndarray, kspace_resolution: float) -> list[int]:
    """
    Compute Monkhorst-Pack mesh from cell and kspace_resolution (in angstrom^-1).
    
    Parameters
    ----------
    cell : np.ndarray
        3x3 array with cell vectors in angstrom.
    kspace_resolution : float
        Desired k-space resolution in angstrom^-1.
    
    Returns
    -------
    list[int]
        List of 3 integers defining the Monkhorst-Pack mesh.
    """
    rec = 2.0 * np.pi * np.linalg.inv(cell).T
    lengths = np.linalg.norm(rec, axis=1)
    mesh = [max(1, int(np.ceil(L / kspace_resolution))) for L in lengths]
    logger.debug(
        "QE MP mesh %s using k-resolution %s angstrom^-1", mesh, kspace_resolution
    )
    return mesh