from __future__ import annotations

import logging
import os
import re
import subprocess

from ase.io import read
from ase.units import GPa
from pymatgen.io.ase import AseAtomsAdaptor

from jobflow import job

from .schema import InputDoc, OutputDoc, TaskDoc

logger = logging.getLogger(__name__)


_ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-\d\.Ee+]+)\s+Ry")


def _parse_total_energy_ev(pwo_path: str) -> float | None:
    """
    Extract and return total energy (eV) if found in QE output (.pwo)

    Parameters
    ----------
    pwo_path : str
        Path to QE output file (.pwo)
    
    Returns
    -------
    float | None
        Total energy in eV if found, else None
    """
    if not os.path.exists(pwo_path):
        return None

    try:
        with open(pwo_path, errors="ignore") as fh:
            for line in fh:
                m = _ENERGY_RE.search(line)
                if m:
                    # convert energy in eV
                    ry = float(m.group(1))
                    return ry * 13.605693009
    except Exception:
        return None
    return None


@job
def run_qe_static(input: InputDoc, command: str) -> TaskDoc:
    """
    Execute single QE SCF static calculation from .pwi file.
    Parse output .pwo file to extract total energy, forces, stress, and final structure.

    Parameters
    ----------
    input : InputDoc
        Input document containing paths and settings for the QE run.
    command : str
        Command to execute QE (e.g. "pw.x" or "mpirun -np 4 pw.x -nk 2").
    
    Returns
    -------
    TaskDoc
        Document containing input, output, and metadata of the QE run.
    """
    pwi_path = input.pwi_path
    pwo_path = pwi_path.replace(".pwi", ".pwo")
    # Assemble pwscf run command e.g. "pw.x < input.pwi >> input.pwo"
    run_cmd = f"{command} < {pwi_path} >> {pwo_path}"

    success = False
    try:
        subprocess.run(run_cmd, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError as exc:
        logger.error("QE failed for %s: %s", pwi_path, exc)

    # # Manual parse of total energy in eV from .pwo
    # energy_ev = _parse_total_energy_ev(pwo_path)
    
    # Parse with ASE
    atoms = read(pwo_path)
    energy_ev = atoms.get_total_energy()
    forces_evA = atoms.get_forces()
    stress_kbar = atoms.get_stress(voigt=False)*(-10/GPa)
    final_structure = AseAtomsAdaptor().get_structure(atoms)

    output = OutputDoc(
        energy=energy_ev,
        forces=forces_evA.tolist(),
        stress=stress_kbar.tolist(),
        energy_per_atom=energy_ev / len(atoms) if energy_ev is not None else None,
    )

    return TaskDoc(
        structure=final_structure,
        dir_name=os.path.dirname(pwi_path),
        task_label="qe_scf",
        input=input,
        output=output,
    )