from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Optional

from jobflow import job

from .schema import QeRunResult

logger = logging.getLogger(__name__)


_ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-\d\.Ee+]+)\s+Ry")


def _parse_total_energy_ev(pwo_path: str) -> Optional[float]:
    """
    Extract and return total energy (eV) if found in QE output (.pwo)
    """
    if not os.path.exists(pwo_path):
        return None

    try:
        with open(pwo_path, "r", errors="ignore") as fh:
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
def run_qe_static(pwi_path: str, command: str) -> QeRunResult:
    """
    Execute single QE SCF static calculation from .pwi file.
    """
    pwo_path = pwi_path.replace(".pwi", ".pwo")
    # Assemble pwscf run command e.g. "pw.x < input.pwi >> input.pwo"
    run_cmd = f"{command} < {pwi_path} >> {pwo_path}"

    success = False
    try:
        subprocess.run(run_cmd, shell=True, check=True, executable="/bin/bash")
        success = True
    except subprocess.CalledProcessError as exc:
        logger.error("QE failed for %s: %s", pwi_path, exc)
        success = False

    # Return outdir read from .pwi
    outdir = ""
    try:
        with open(pwi_path, "r") as fh:
            for line in fh:
                if "outdir" in line:
                    outdir = line.split("=")[1].strip().strip("'").strip('"')
                    break
    except Exception:
        pass

    energy_ev = _parse_total_energy_ev(pwo_path)

    return QeRunResult(
        success=success,
        pwi=os.path.abspath(pwi_path),
        pwo=os.path.abspath(pwo_path),
        outdir=os.path.abspath(outdir) if outdir else "",
        total_energy_ev=energy_ev,
    )