"""Geometry validation and bounded process supervision for RSS relaxations."""

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic, minkowski_reduce
from threadpoolctl import threadpool_limits

from autoplex.settings import RssSafety

logger = logging.getLogger(__name__)


class RssGeometryError(Exception):
    """Abort an unsafe relaxation without ASE retrying the line search."""


def validate_rss_geometry(atoms, safety: RssSafety) -> None:
    """Reject unsafe cells and overlaps before constructing backend neighbours.

    Periodic self images are checked even for one-atom cells. Nonperiodic cell
    vectors may be zero (e.g. a molecule or a slab); only periodic vectors must
    be independent. Distance checks use a reduced lattice rather than assuming
    that the three input lattice vectors are the shortest translations.
    """
    if not len(atoms):
        raise RssGeometryError("empty structure")
    cell = np.asarray(atoms.cell)
    if not np.isfinite(atoms.positions).all() or not np.isfinite(cell).all():
        raise RssGeometryError("nonfinite coordinates or cell")

    pbc = atoms.pbc
    if pbc.any():
        # The SVD for the cell computes in fact the (square root) eigenvalues of its metric G = cell.cell^T
        # A nearly zero eigenvalue indicates a combination of basis vectors with almost zero length.
        # This however also excludes very skewed cells of a valid structure. It is possible to compute a reduced
        # cell via minkowski_reduce or alike, but I felt this reduction should happen somewhere else (maybe it does
        # already).
        singular_values = np.linalg.svd(cell[pbc], compute_uv=False)
        if singular_values[-1] <= 0 or (
            singular_values[0] / singular_values[-1] > safety.max_cell_condition_number
        ):
            raise RssGeometryError("singular or ill-conditioned periodic cell")
        if pbc.all() and safety.min_volume_per_atom is not None:
            volume_per_atom = abs(np.linalg.det(cell)) / len(atoms)
            if volume_per_atom < safety.min_volume_per_atom:
                raise RssGeometryError(
                    f"volume per atom {volume_per_atom:g} A^3 is below {safety.min_volume_per_atom:g} A^3"
                )

    if safety.min_distance is None:
        return
    if pbc.any():
        reduced_cell, _ = minkowski_reduce(cell, pbc=pbc)
        shortest_image = np.linalg.norm(reduced_cell[pbc], axis=1).min()
        if shortest_image < safety.min_distance:
            raise RssGeometryError(
                f"periodic self-image distance {shortest_image:g} A is below {safety.min_distance:g} A"
            )
    i, j = np.triu_indices(len(atoms), k=1)
    if len(i):
        displacements = atoms.positions[j] - atoms.positions[i]
        _, distances = find_mic(displacements, cell, pbc)
        if distances.min() < safety.min_distance:
            raise RssGeometryError(
                f"interatomic distance {distances.min():g} A is below {safety.min_distance:g} A"
            )


class GuardedCalculator(Calculator):
    """Validate every changed geometry before invoking an ASE calculator."""

    def __init__(self, calculator, safety: RssSafety):
        super().__init__()
        self.calculator = calculator
        self.safety = safety
        self.implemented_properties = calculator.implemented_properties

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Check trial cells, including calls inside an optimizer line search."""
        validate_rss_geometry(atoms, self.safety)
        super().calculate(atoms, properties, system_changes)
        # NequIP 0.6 inspects atoms.calc while converting the input structure.
        # Keep the same Atoms object so backend updates to info/arrays survive.
        original_calculator = atoms.calc
        try:
            atoms.calc = self.calculator
            self.results = {
                name: self.calculator.get_property(name, atoms) for name in properties
            }
        finally:
            atoms.calc = original_calculator


def _run_relaxation(function, atom, result_path, seed):
    """Publish a small result atomically; abrupt child death leaves no result."""
    try:
        np.random.seed(seed)
        with threadpool_limits(limits=1):
            value = function(atom)
        result = {"value": value}
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    path = Path(result_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(path)


def _stop_process(process):
    """Bound shutdown even when native code ignores a termination request."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
    if process.is_alive():
        raise RuntimeError(f"Could not stop RSS worker {process.pid}")
    process.join()
    process.close()


def supervise_relaxations(function, atoms, max_workers, max_walltime):
    """Run one structure per child, preserving input order and successful paths.

    Monitor process exit codes rather than waiting on pool results that can be
    lost when a child is killed. Results use atomic files, avoiding a corrupted
    multiprocessing pipe or a queue lock left behind by a killed worker. The
    timeout includes process startup and model loading, and starts separately
    for each child, not while its input is waiting to be scheduled.
    """
    if max_workers < 1:
        raise ValueError("num_processes_rss must be positive")
    if not np.isfinite(max_walltime) or max_walltime <= 0:
        raise ValueError("RSS max_walltime must be finite and positive")
    results = [None] * len(atoms)
    context = mp.get_context()
    active = {}
    next_index = 0
    with TemporaryDirectory(prefix="autoplex-rss-") as directory:
        try:
            while next_index < len(atoms) or active:
                while next_index < len(atoms) and len(active) < max_workers:
                    index = next_index
                    path = Path(directory) / f"{index}.json"
                    process = context.Process(
                        target=_run_relaxation,
                        args=(
                            function,
                            atoms[index],
                            str(path),
                            int(np.random.randint(2**32)),
                        ),
                    )
                    process.start()
                    active[index] = (process, time.monotonic(), path)
                    next_index += 1

                for index, (process, started, path) in list(active.items()):
                    reason = None
                    if not process.is_alive():
                        if process.exitcode != 0:
                            reason = f"worker exited with code {process.exitcode}"
                        elif not path.is_file():
                            reason = "worker exited without a result"
                        else:
                            result = json.loads(path.read_text(encoding="utf-8"))
                            reason = result.get("error")
                            results[index] = result.get("value")
                    elif time.monotonic() - started >= max_walltime:
                        reason = f"exceeded wall-time limit of {max_walltime:g} s"
                    else:
                        continue
                    if reason:
                        logger.warning(
                            "RSS structure %s failed: %s",
                            atoms[index].info.get("unique_starting_index", index),
                            reason,
                        )
                    _stop_process(process)
                    del active[index]
                if active:
                    time.sleep(0.05)
        finally:
            for process, _, _ in active.values():
                _stop_process(process)
    return results
