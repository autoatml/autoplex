"""Mock LOBSTER functions for executing tutorials.

This code has been taken from https://github.com/materialsproject/atomate2.
The code has been released under BSD 3-Clause License
and the following copyright applies:
atomate2 Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.

"""

import contextlib
import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

from atomate2.utils.testing.vasp import monkeypatch_vasp
from pytest import MonkeyPatch

TEST_ROOT = Path(__file__).parent.parent / "tests"
TEST_DIR = TEST_ROOT / "test_data"


@contextlib.contextmanager
def mock_vasp(ref_paths: dict, clean_folders=True) -> Generator:
    """Mock VASP functions.

    Parameters
    ----------
    ref_paths (dict): A dictionary of reference paths to the test data.

    Yields
    ------
        function: A function that mocks calls to VASP.
    """
    for mf in monkeypatch_vasp(MonkeyPatch(), TEST_DIR):
        fake_run_vasp_kwargs = {k: {"check_inputs": ()} for k in ref_paths}
        old_cwd = os.getcwd()
        new_path = tempfile.mkdtemp()
        os.chdir(new_path)
        try:
            yield mf(ref_paths, fake_run_vasp_kwargs)
        finally:
            os.chdir(old_cwd)
            if clean_folders:
                shutil.rmtree(new_path)
