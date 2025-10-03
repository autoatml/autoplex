from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io import read

from .schema import QeKpointsSettings, QeInputSet, QeRunSettings

logger = logging.getLogger(__name__)


class QeStaticInputGenerator:
    """
    Input generator to run static SCF calculations with Quantum Espresso (.pwi), using:
    - .pwi template containing computational parameters (&control, &system, &electrons);
    - ASE structures;
    - k-points optional settings if K_POINTS is not defined within template

    Used to write one .pwi for each structure inside `workdir`.
    """

    def __init__(
        self,
        template_pwi: Optional[str] = None,
        run_settings: Optional[QeRunSettings] = None,
        kpoints: Optional[QeKpointsSettings] = None,
        pseudo: Optional[dict[str, str]] = None,
    ) -> None:
        self.template_pwi = template_pwi
        self.run_settings = run_settings or QeRunSettings()
        self.kpoints = kpoints or QeKpointsSettings()
        self.pseudo = pseudo

    def generate_for_structures(
        self,
        structures: str | List[str] | None,
        workdir: str,
        seed_prefix: str = "structure",
    ) -> List[QeInputSet]:
        """
        Generate one .pwi for each structure read from ASE-readable files.
        """
        os.makedirs(workdir, exist_ok=True)
        atoms_list = _load_structures(structures)
        input_sets: List[QeInputSet] = []
        template_lines = _read_template(self.template_pwi) if self.template_pwi else None

        for i, atoms in enumerate(atoms_list):
            seed = f"{seed_prefix}_{i}"
            pwi_path = os.path.join(workdir, f"{seed}.pwi")
            self._write_pwi(
                pwi_output=pwi_path,
                atoms=atoms,
                template_lines=list(template_lines) if template_lines else None,
                kpoints=self.kpoints,
                pseudo=self.pseudo,
            )
            input_sets.append(QeInputSet(workdir=workdir, pwi_path=pwi_path, seed=seed))
        return input_sets

    def _write_pwi(
        self,
        *,
        pwi_output: str,
        atoms: Atoms,
        template_lines: Optional[List[str]],
        kpoints: QeKpointsSettings,
        pseudo: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Write .pwi file using template (if present) + K_POINTS/cell/positions
        Updating `nat`, `outdir`
        """
        if template_lines is None:
            # Minimal template with main QE namelists
            template_lines = _render_minimal_namelists(self.run_settings)

        pwi = list(template_lines)

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
        if pseudo is None:
            raise ValueError("Pseudo dictionary must be provided to write ATOMIC_SPECIES.")
        species = set(atoms.get_chemical_symbols())
        ntyp = len(species)
        pwi[idx_ntyp] = f"   ntyp = {ntyp}\n"
        species_lines = ["\nATOMIC_SPECIES\n"]
        for s in sorted(species):
            if s not in pseudo:
                raise ValueError(f"Missing pseudo for atomic symbol '{s}' in pseudo dictionary.")
            mass = atomic_masses[atomic_numbers[s]]
            species_lines.append(f"{s}  {mass:.4f} {pseudo[s]}\n")
        species_lines.append("\n")

        # K-POINTS
        kpt_lines = _render_kpoints(
            template_lines=pwi, idx_kpts=idx_kpts, atoms=atoms, kpoints=kpoints
        )
        kpt_lines.append("\n")

        # CELL_PARAMETERS
        cell = atoms.cell
        cell_lines = ["\nCELL_PARAMETERS (angstrom)\n"]
        for i in range(3):
            cell_lines.append(f"{cell[i, 0]:.10f} {cell[i, 1]:.10f} {cell[i, 2]:.10f}\n")
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


# --------- Utilities ---------

def _load_structures(paths: str | List[str] | None) -> List[Atoms]:
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    atoms_list: List[Atoms] = []
    for fname in paths:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Structure file not found: {fname}")
        atoms_list += read(fname, index=":")
    return atoms_list


def _read_template(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"template_pwi not found: {path}")
    with open(path, "r") as fh:
        return fh.readlines()


def _render_minimal_namelists(settings: QeRunSettings) -> List[str]:
    """Create draft lines for namelists &control, &system, &electrons."""
    def _render_block(name: str, kv: dict) -> List[str]:
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

    out: List[str] = []
    out += _render_block("control", settings.control)
    out += _render_block("system", settings.system)
    out += _render_block("electrons", settings.electrons)
    return out


def _render_kpoints(
    *,
    template_lines: List[str],
    idx_kpts: int | None,
    atoms: Atoms,
    kpoints: QeKpointsSettings,
) -> List[str]:
    """Create lines for K_POINTS section."""

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


def _compute_kpoints_grid(cell: np.ndarray, kspace_resolution: float) -> List[int]:
    """Compute Monkhorst-Pack mesh from cell and kspace_resolution (in angstrom^-1)."""
    rec = 2.0 * np.pi * np.linalg.inv(cell).T
    lengths = np.linalg.norm(rec, axis=1)
    mesh = [max(1, int(np.ceil(L / kspace_resolution))) for L in lengths]
    logger.debug("QE MP mesh %s using k-resolution %s angstrom^-1", mesh, kspace_resolution)
    return mesh