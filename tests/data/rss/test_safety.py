"""Regression tests for RSS geometry guards and process failures."""

import ctypes
import multiprocessing as mp
import os
import signal
import time
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from pydantic import ValidationError

from autoplex.data.rss import safety as safety_module
from autoplex.data.rss.safety import (
    GuardedCalculator,
    RssGeometryError,
    supervise_relaxations,
    validate_rss_geometry,
)
from autoplex.settings import RssConfig, RssSafety


class CountingCalculator(Calculator):
    """Count evaluations to detect validation happening after the backend."""

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Supply simple energies, forces, and stresses."""
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {
            "energy": 1.0,
            "free_energy": 1.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def _worker(atom):
    action = atom.info["action"]
    if action == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    if action == "timeout":
        # A process blocked inside native code must also be terminated.
        ctypes.CDLL(None).sleep(30)
    if action == "error":
        raise ValueError("model initialization failed")
    return str(atom.info["unique_starting_index"])


def _atom(action, index):
    atom = Atoms("O", cell=[3, 3, 3], pbc=True)
    atom.info.update(action=action, unique_starting_index=str(index))
    return atom


@pytest.mark.parametrize(
    "atom",
    [
        Atoms("O", cell=[3, 3, 3], pbc=True),
        Atoms("O2", positions=[[0, 0, 0], [1, 0, 0]]),
        Atoms("O", cell=[3, 3, 0], pbc=[True, True, False]),
        Atoms("O", cell=[3, 0, 0], pbc=[True, False, False]),
    ],
)
def test_valid_geometry(atom):
    """Exercise the RSS safety contract."""
    validate_rss_geometry(atom, RssSafety())


@pytest.mark.parametrize(
    "cell",
    [
        # PID 184052 from the stalled run: volume approximately 1.7e-5 A^3.
        [
            [0.041932205, -0.087278842, 0.105919613],
            [-0.158732687, 2.227780314, 0.255089209],
            [0.107487382, 1.275609105, 0.790139612],
        ],
        np.zeros((3, 3)),
        np.eye(3) * 0.01,
        [[3, 0, 0], [3, 1e-6, 0], [0, 0, 3]],
    ],
)
def test_collapsed_cells_rejected(cell):
    """Exercise the RSS safety contract."""
    with pytest.raises(RssGeometryError):
        validate_rss_geometry(Atoms("O", cell=cell, pbc=True), RssSafety())


def test_self_images_in_skew_cell():
    """Exercise the RSS safety contract."""
    # Both input vectors are long; their difference is only 0.014 A.
    atom = Atoms("O", cell=[[2, 0, 0], [1.99, 0.01, 0], [0, 0, 2]], pbc=True)
    with pytest.raises(RssGeometryError, match="self-image"):
        validate_rss_geometry(atom, RssSafety(min_volume_per_atom=None))


def test_periodic_pair_overlap():
    """Exercise the RSS safety contract."""
    atom = Atoms("O2", positions=[[0, 0, 0], [2.99, 0, 0]], cell=[3, 3, 3], pbc=True)
    with pytest.raises(RssGeometryError, match="interatomic"):
        validate_rss_geometry(atom, RssSafety())


@pytest.mark.parametrize("component", ["positions", "cell"])
def test_nonfinite_geometry(component):
    """Exercise the RSS safety contract."""
    atom = Atoms("O", cell=[3, 3, 3], pbc=True)
    getattr(atom, component)[0, 0] = np.nan
    with pytest.raises(RssGeometryError, match="nonfinite"):
        validate_rss_geometry(atom, RssSafety())


@pytest.mark.parametrize("property_name", ["energy", "free_energy", "forces", "stress"])
def test_trial_geometry_checked_before_backend(property_name):
    """Exercise the RSS safety contract."""
    atom = Atoms("O", cell=[3, 3, 3], pbc=True)
    backend = CountingCalculator()
    atom.calc = GuardedCalculator(backend, RssSafety())
    assert atom.get_potential_energy() == 1
    assert backend.calls == 1
    atom.set_cell(np.eye(3) * 0.01)
    with pytest.raises(RssGeometryError):
        atom.calc.get_property(property_name, atom)
    assert backend.calls == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"max_walltime": 0},
        {"max_walltime": float("inf")},
        {"min_distance": -1},
        {"min_volume_per_atom": float("nan")},
        {"max_cell_condition_number": 0.5},
        {"typo": 10},
    ],
)
def test_invalid_safety_settings(settings):
    """Exercise the RSS safety contract."""
    with pytest.raises(ValidationError):
        RssConfig(rss_safety=settings)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_killed_and_stuck_workers_do_not_block_successes(monkeypatch, caplog, max_workers):
    """Exercise the RSS safety contract."""
    # Fork keeps this failure-injection test fast on macOS as well as Linux.
    # The default/spawn context is exercised separately below.
    context = mp.get_context("fork")
    monkeypatch.setattr(safety_module.mp, "get_context", lambda: context)
    atoms = [_atom(action, i) for i, action in enumerate(["ok", "kill", "timeout", "error", "ok"])]
    previous_children = {p.pid for p in mp.active_children()}
    started = time.monotonic()
    results = supervise_relaxations(_worker, atoms, max_workers=max_workers, max_walltime=2)
    assert results == ["0", None, None, None, "4"]
    assert time.monotonic() - started < 10
    assert "structure 1 failed: worker exited with code" in caplog.text
    assert "structure 2 failed: exceeded wall-time" in caplog.text
    assert "structure 3 failed: ValueError: model initialization failed" in caplog.text
    assert not ({p.pid for p in mp.active_children()} - previous_children)


def test_spawn_context_preserves_order(monkeypatch):
    """Exercise the RSS safety contract."""
    context = mp.get_context("spawn")
    monkeypatch.setattr(safety_module.mp, "get_context", lambda: context)
    assert supervise_relaxations(_worker, [_atom("ok", 0), _atom("ok", 1)], max_workers=2, max_walltime=60) == [
        "0",
        "1",
    ]


def test_empty_input():
    """Exercise the RSS safety contract."""
    assert supervise_relaxations(_worker, [], max_workers=1, max_walltime=1) == []


def _random_worker(atom):
    return str(np.random.random())


def test_independent_reproducible_child_random_streams(monkeypatch):
    """One child per input must not repeat a fork-inherited pressure draw."""
    context = mp.get_context("fork")
    monkeypatch.setattr(safety_module.mp, "get_context", lambda: context)
    atoms = [_atom("ok", i) for i in range(3)]
    np.random.seed(42)
    first = supervise_relaxations(_random_worker, atoms, 2, 10)
    np.random.seed(42)
    second = supervise_relaxations(_random_worker, atoms, 2, 10)
    assert first == second
    assert len(set(first)) == 3
