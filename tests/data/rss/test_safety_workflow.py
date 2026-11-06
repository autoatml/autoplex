"""Integration tests for the RSS safety configuration and failure reporting."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.auto.rss.flows import RssMaker
from autoplex.auto.rss.jobs import do_rss_iterations
from autoplex.data.common import jobs as common_jobs
from autoplex.data.rss import jobs, utils
from autoplex.data.rss.safety import GuardedCalculator
from autoplex.settings import RssConfig, RssSafety


def test_invalid_initial_geometry_does_not_load_model(caplog):
    """Reject the structure before loading a nonexistent potential."""
    atom = Atoms("O", cell=np.eye(3) * 0.01, pbc=True)
    atom.info["unique_starting_index"] = "collapsed"
    assert utils.process_rss(atom, "GAP", "/nonexistent/model") is None
    assert "structure collapsed rejected" in caplog.text


def test_all_failed_group_has_clear_error(monkeypatch):
    """An RSS job must not silently succeed when every structure failed."""
    monkeypatch.setattr(utils, "supervise_relaxations", lambda *args: [None])
    structure = AseAtomsAdaptor.get_structure(Atoms("O", cell=[3, 3, 3], pbc=True))
    with pytest.raises(RuntimeError, match="All RSS relaxations failed"):
        utils.minimize_structures("GAP", "/unused", "1th", [structure])


def test_insufficient_trajectory_frames(monkeypatch):
    """Report the available frame count before trying to select structures."""
    monkeypatch.setattr(
        common_jobs,
        "handle_rss_trajectory",
        lambda *args: ([[Atoms("O"), Atoms("O")]], [[0, 0]]),
    )
    with pytest.raises(ValueError, match="only 2 usable trajectory frames"):
        common_jobs.sample_data.original(num_of_selection=3, traj_path=["unused"])


def test_safety_forwarded_through_jobs(monkeypatch):
    """The grouped and single-node jobs carry custom limits to the supervisor."""
    structure = AseAtomsAdaptor.get_structure(Atoms("O", cell=[3, 3, 3], pbc=True))
    safety = {"max_walltime": 123}
    response = jobs.do_rss_multi_node.original(
        "GAP", "/unused", "1th", structure=[structure], rss_safety=safety
    )
    single_job = response.replace.jobs[0]
    assert single_job.function_kwargs["rss_safety"] == safety
    captured = {}

    def minimize(**kwargs):
        captured.update(kwargs)
        return ["trajectory.extxyz"]

    monkeypatch.setattr(jobs, "minimize_structures", minimize)
    result = jobs.do_rss_single_node.original(**single_job.function_kwargs)
    assert result == ["trajectory.extxyz"]
    assert captured["rss_safety"] == safety


def test_safety_forwarded_to_recursive_iterations():
    """Custom limits survive the automatic next-iteration job."""
    safety = {"max_walltime": 123}
    response = do_rss_iterations.original(
        input={
            "test_error": 1,
            "mlip_path": "/unused",
            "pre_database_dir": "/unused",
            "isolated_atom_energies": {8: -1},
        },
        tag="O",
        generated_struct_numbers=[1],
        rss_safety=safety,
    )
    rss_job = response.detour[1]
    next_iteration = response.detour[-1]
    assert rss_job.function_kwargs["rss_safety"] == safety
    assert next_iteration.function_kwargs["rss_safety"] == safety


@pytest.mark.parametrize("train_from_scratch", [True, False])
def test_maker_routes_safety_only_to_rss(train_from_scratch):
    """RSS controls must not leak through initial fitting keyword arguments."""
    config = RssConfig(
        train_from_scratch=train_from_scratch, rss_safety={"max_walltime": 123}
    )
    maker = RssMaker(rss_config=config)
    response = maker.make.original(maker)
    rss_job = response.replace.jobs[-1]
    assert rss_job.function_kwargs["rss_safety"]["max_walltime"] == 123
    if train_from_scratch:
        assert "rss_safety" not in response.replace.jobs[0].function_kwargs


def test_invalid_line_search_trial_aborts_relaxation(monkeypatch, caplog):
    """The initial valid cell must not bypass checks on later trial cells."""

    class CollapsingOptimizer:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms

        def attach(self, callback):
            pass

        def run(self, **kwargs):
            self.atoms.atoms.set_cell(np.eye(3) * 0.01)
            self.atoms.get_potential_energy()

    monkeypatch.setattr(utils, "extract_gap_label", lambda path: "unused")
    monkeypatch.setattr(utils, "CustomPotential", lambda **kwargs: EMT())
    monkeypatch.setattr(utils, "PreconLBFGS", CollapsingOptimizer)
    atom = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    atom.info["unique_starting_index"] = "trial-collapse"
    assert utils.process_rss(atom, "GAP", "/unused", keep_symmetry=False) is None
    assert "structure trial-collapse failed: volume per atom" in caplog.text


def test_legacy_nequip_guarded_calculator(test_dir):
    """NequIP 0.6 AtomicData.from_ase must accept the guarded evaluation."""
    nequip_ase = pytest.importorskip("nequip.ase")
    if not hasattr(nequip_ase.NequIPCalculator, "from_deployed_model"):
        pytest.skip("This regression applies to the legacy NequIP ASE API")
    backend = nequip_ase.NequIPCalculator.from_deployed_model(
        model_path=str(test_dir / "fitting/NEQUIP/old/deployed_nequip_model.pth"),
        device="cpu",
        species_to_type_name={"Si": "Si"},
        set_global_options=False,
    )
    atom = read(test_dir / "data/rss.extxyz", index="0")
    atom.calc = backend
    reference = {
        "energy": atom.get_potential_energy(),
        "forces": atom.get_forces(),
        "stress": atom.get_stress(),
    }
    backend.reset()
    guard = GuardedCalculator(backend, RssSafety())
    atom.calc = guard
    for name, expected in reference.items():
        # The legacy model predicts in float32; repeated Torch evaluations
        # can differ by a few ulps even on the same CPU.
        np.testing.assert_allclose(
            atom.calc.get_property(name, atom), expected, rtol=1e-5, atol=1e-5
        )
        assert atom.calc is guard
