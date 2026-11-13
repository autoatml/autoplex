"""Physical calculator contracts for MatGL-backed RSS."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize.precon import PreconLBFGS
from ase.units import GPa

from autoplex.data.rss import utils as rss_utils


def test_m3gnet_rss_stress_units(
    test_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RSS must convert MatGL's GPa stress to ASE units exactly once."""
    matgl = pytest.importorskip("matgl")
    from matgl.ext.ase import M3GNetCalculator

    model_path = test_dir / "fitting/M3GNET/m3gnet_results/training"
    atoms = read(test_dir / "data/rss.extxyz", index=0)
    reference = atoms.copy()
    reference.calc = M3GNetCalculator(
        potential=matgl.load_model(path=model_path), stress_unit="GPa"
    )
    expected_energy = reference.get_potential_energy()
    expected_forces = reference.get_forces()
    expected_stress = reference.get_stress() * GPa
    assert np.linalg.norm(expected_stress) > 1e-6

    optimizer_class = rss_utils.PreconLBFGS
    checked = []

    def check_optimizer_atoms(
        filtered_atoms: UnitCellFilter, *args: Any, **kwargs: Any
    ) -> PreconLBFGS:
        guarded_atoms = filtered_atoms.atoms
        np.testing.assert_allclose(
            guarded_atoms.get_potential_energy(), expected_energy, rtol=1e-6
        )
        np.testing.assert_allclose(
            guarded_atoms.get_forces(), expected_forces, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            guarded_atoms.get_stress(), expected_stress, rtol=1e-5, atol=1e-8
        )
        checked.append(True)
        return optimizer_class(filtered_atoms, *args, **kwargs)

    monkeypatch.setattr(rss_utils, "PreconLBFGS", check_optimizer_atoms)
    monkeypatch.chdir(tmp_path)
    atoms.info["unique_starting_index"] = "stress_units"
    result = rss_utils.process_rss(
        atoms,
        mlip_type="M3GNET",
        mlip_path=model_path,
        scalar_exp_pressure=0,
        scalar_pressure_exponential_width=0,
        max_steps=0,
        force_tol=1e6,
        stress_tol=1e6,
        keep_symmetry=False,
    )
    assert checked == [True]
    assert result is not None and Path(result).is_file()
