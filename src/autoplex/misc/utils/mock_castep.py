"""Utilities for testing castep calculations.

License / Copyright
-------------------
This code is adapted from `atomate2/src/atomate2/utils/testing/aims.py`.

The code has been released under BSD 3-Clause License
and the following copyright applies:

atomate2 Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.


Reference:

Ganose, A. M., Sahasrabuddhe, H., Asta, M., Beck, K., Biswas, T., Bonkowski, A.,
Bustamante, J., Chen, X., Chiang, Y., Chrzan, D. C., Clary, J., Cohen, O. A.,
Ertural, C., Gallant, M. C., George, J., Gerits, S., Goodall, R. E. A.,
Guha, R. D., Hautier, G., Horton, M., Inizan, T. J., Kaplan, A. D., Kingsbury, R. S.,
Kuner, M. C., Li, B., Linn, X., McDermott, M. J., Mohanakrishnan, R. S., Naik, A. A.,
Neaton, J. B., Parmar, S. M., Persson, K. A., Petretto, G., Purcell, T. A. R.,
Ricci, F., Rich, B., Riebesell, J., Rignanese, G.-M., Rosen, A. S., Scheffler, M.,
Schmidt, J., Shen, J.-X., Sobolev, A., Sundararaman, R., Tezak, C., Trinquet, V.,
Varley, J. B., Vigil-Fowler, D., Wang, D., Waroquiers, D., Wen, M., Yang, H.,
Zheng, H., Zheng, J., Zhu, Z., & Jain, A. (2025). Atomate2: modular workflows
for materials science. Digital Discovery. DOI: 10.1039/D5DD00019J.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from jobflow import CURRENT_JOB
from monty.os.path import zpath as monty_zpath

import autoplex.misc.castep.jobs
from autoplex.misc.castep.utils import CastepStaticSetGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from pytest import MonkeyPatch

    from autoplex.misc.castep.utils import CastepInputGenerator

logger = logging.getLogger("autoplex")


_VFILES: Final = ("castep.param", "castep.cell")
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_CASTEP_KWARGS: dict[str, dict] = {}


def zpath(path: str | Path) -> Path:
    """Return the path of a zip file.

    Returns an existing (zipped or unzipped) file path given the unzipped
    version. If no path exists, returns the unmodified path.
    """
    return Path(monty_zpath(str(path)))


def monkeypatch_castep(
    monkeypatch: MonkeyPatch, ref_path: Path
) -> Generator[Callable[[Any, Any], Any]]:
    """Fixture to mock CASTEP runs for tests.

    Usage
    -----
    1. "mock_castep" should be included as an argument to any test that would like to use
       its functionally.
    2. For each job in your workflow, you should prepare a reference directory
       containing two folders "inputs" (containing the reference input files expected
       to be produced by Castep().initialize()) and "outputs" (containing the expected
       output files to be produced by run_castep). These files should reside in a
       subdirectory of "tests/test_data/castep".
    3. Create a dictionary mapping each job name to its reference directory. Note that
       you should supply the reference directory relative to the "tests/test_data/castep"
       folder. For example, if your calculation has one job named "static" and the
       reference files are present in "tests/test_data/castep/Si_static", the dictionary
       would look like: ``{"static": "Si_static"}``.
    4. Inside the test function, call `mock_castep(ref_paths, fake_castep_kwargs)`, where
       ref_paths is the dictionary created in step 3.
    5. Run your castep job after calling `mock_castep`.
    For examples, see the tests in tests/misc/castep/test_jobs.py.

    Parameters
    ----------
    monkeypatch
        MonkeyPatch fixture used to set attributes.
    ref_path : Path
        Path to the root folder containing the ``tests/test_data/castep`` reference data.

    Returns
    -------
    Generator[Callable[..., None], None, None]
        A generator that yields a helper function which accepts the reference
        mapping and optional kwargs. After the test, the fixture undoes the
        monkeypatches and clears the internal mappings.
    """

    def mock_run_castep(*args, **kwargs) -> None:
        name = CURRENT_JOB.job.name
        try:
            ref_dir = ref_path / _REF_PATHS[name]
        except KeyError:
            raise ValueError(
                f"no reference directory found for job {name!r}; "
                f"reference paths received={_REF_PATHS}"
            ) from None
        fake_run_castep(ref_dir, **_FAKE_RUN_CASTEP_KWARGS.get(name, {}))

    get_input_set_orig = CastepStaticSetGenerator.get_input_set

    def mock_get_input_set(
        self: CastepStaticSetGenerator, *args, **kwargs
    ) -> CastepInputGenerator:
        return get_input_set_orig(self, *args, **kwargs)

    monkeypatch.setattr(autoplex.misc.castep.run, "run_castep", mock_run_castep)
    monkeypatch.setattr(autoplex.misc.castep.jobs, "run_castep", mock_run_castep)
    monkeypatch.setattr(CastepStaticSetGenerator, "get_input_set", mock_get_input_set)

    def _run(ref_paths: dict, fake_run_castep_kwargs: dict | None = None) -> None:
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_CASTEP_KWARGS.update(fake_run_castep_kwargs or {})

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_CASTEP_KWARGS.clear()


def fake_run_castep(
    ref_path: str | Path,
    input_settings: Sequence[str] | None = None,
    check_inputs: Sequence[Literal["castep.param", "castep.cell"]] = _VFILES,
    clear_inputs: bool = False,
) -> None:
    """Emulate running castep and validate castep input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with castep input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    input_settings
        A list of input settings to check.
    check_inputs
        A list of castep input files to check. Supported options are "castep.inp"
    clear_inputs
        Whether to clear input files before copying in the reference castep outputs.
    """
    logger.info("Running fake castep.")

    ref_path = Path(ref_path)

    logger.info("Verified inputs successfully")

    if clear_inputs:
        clear_castep_inputs()

    copy_castep_outputs(ref_path)

    # pretend to run castep by copying pre-generated outputs from reference dir
    logger.info("Generated fake castep outputs")


def clear_castep_inputs() -> None:
    """Clean up castep input files."""
    for castep_file in ("castep.param", "castep.cell"):
        if Path(castep_file).exists():
            Path(castep_file).unlink()
    logger.info("Cleared castep inputs")


def copy_castep_outputs(ref_path: str | Path) -> None:
    """Copy castep output files from the reference directory.

    Parameters
    ----------
    ref_path : str or Path
        Path to the reference directory containing an ``outputs`` subfolder
        with the CASTEP output files to copy.
    """
    output_path = Path(ref_path) / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, "./CASTEP")
