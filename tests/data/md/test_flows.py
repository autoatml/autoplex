from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.io import read
from atomate2.forcefields.md import ForceFieldMDMaker
from jobflow import run_locally
from pymatgen.io.ase import AseAtomsAdaptor
from ase import __version__ as ase_version
from packaging.version import Version

from autoplex.data.md.flows import ScaledMDMaker

CUBIC = AseAtomsAdaptor.get_structure(bulk("Si", "diamond", a=5.43, cubic=True))  # 8 atoms
PRIMITIVE = AseAtomsAdaptor.get_structure(bulk("Si", "diamond", a=5.43))  # 2 atoms, 60° angles
SUPERCELL_211 = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
PRIMITIVE_TO_CUBIC = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]  # non-diagonal: fcc primitive -> cubic


@pytest.fixture
def gap_nvt_maker(test_dir):
    """Short NVT MD with the small Si GAP potential from the test data."""
    return ForceFieldMDMaker(
        force_field_name="MLFF.GAP",
        calculator_kwargs={
            "param_filename": str(test_dir / "fitting/GAP/gap_file.xml")
        },
        time_step=1.0,
        ensemble="nvt",
        dynamics="langevin",
    )


def run_nvt(md_maker, structures, store, **cell_options):
    """Run a 2-step NVT MD per prepared cell; return the frames of every run. 
    This function is used by methods below.
    """
    md_job = ScaledMDMaker(
        md_maker=md_maker, temperature_list=[300.0], eqm_step_list=[2], **cell_options
    ).make(structures)
    run_locally(md_job, create_folders=True, ensure_success=True, store=store)

    traj_paths = md_job.output.resolve(store)
    assert len(set(traj_paths)) == len(traj_paths)  # every run writes its own file
    return [read(path, index=":") for path in traj_paths]


def test_scaledmdmaker_gap_nvt(test_dir, memory_jobstore, clean_dir):
    """Run a 2-step NVT MD with GAP."""
    md_job = ScaledMDMaker(
        md_engine="ase",
        md_maker=ForceFieldMDMaker(
            force_field_name="MLFF.GAP",
            calculator_kwargs={
                "param_filename": str(test_dir / "fitting/GAP/gap_file.xml")
            },
            time_step=1.0,
            traj_interval=1,
            ensemble="nvt",
            dynamics="langevin",
            traj_file="MD.traj",
            traj_file_fmt="ase",
            store_trajectory="partial",
        ),
        temperature_list=[300.0],
        eqm_step_list=[2],
    ).make(CUBIC)

    run_locally(md_job, create_folders=True, ensure_success=True, store=memory_jobstore)

    traj_path = md_job.output.resolve(memory_jobstore)

    assert Path(traj_path[0]).exists()

    frames = read(traj_path[0], index=":")

    assert len(frames) == 3


def test_scaledmdmaker_gap_npt(test_dir, memory_jobstore, clean_dir):
    """Run a 2-step NPT MD with GAP."""
    md_job = ScaledMDMaker(
        md_engine="ase",
        md_maker=ForceFieldMDMaker(
            force_field_name="MLFF.GAP",
            calculator_kwargs={
                "param_filename": str(test_dir / "fitting/GAP/gap_file.xml")
            },
            time_step=1.0,
            traj_interval=1,
            ensemble="npt",
            dynamics="nose-hoover",  # ASE NPT: works with every atomate2 version used in CI
            pressure=0.0,  # kbar
            ase_md_kwargs={
                "ttime": 25 * units.fs,  # thermostat time constant
                "pfactor": (75 * units.fs) ** 2 * 100 * units.GPa,  # (barostat time)^2 x bulk modulus of Si
            },
            traj_file="MD.traj",
            traj_file_fmt="ase",
            store_trajectory="partial",
        ),
        temperature_list=[300.0],
        eqm_step_list=[5],
    ).make(CUBIC)

    run_locally(md_job, create_folders=True, ensure_success=True, store=memory_jobstore)

    traj_path = md_job.output.resolve(memory_jobstore)

    assert Path(traj_path[0]).exists()

    frames = read(traj_path[0], index=":")

    # ASE < 3.27 (3.24 on the Python 3.10 CI) does not save the starting structure in NPT
    n_frames = 5 if Version(ase_version) < Version("3.27") else 6
    assert len(frames) == n_frames  # 5 MD steps (+ the starting structure)
    # NPT lets the cell change (NVT keeps it fixed)
    assert frames[-1].get_volume() != pytest.approx(frames[0].get_volume())


# expected per MD run: (atoms, volume relative to its input structure, cell angles)
@pytest.mark.parametrize(
    ("structures", "cell_options", "expected"),
    [
        pytest.param([CUBIC], {}, [(8, 1.0, 90)], id="no-supercell-no-scaling"),
        pytest.param(
            [CUBIC], {"supercell_matrix": SUPERCELL_211}, [(16, 2.0, 90)], id="supercell"
        ),
        pytest.param(
            [PRIMITIVE],
            {"supercell_matrix": PRIMITIVE_TO_CUBIC},
            [(8, 4.0, 90)],
            id="non-diagonal-supercell",
        ),
        pytest.param(
            [CUBIC],
            {"volume_custom_scale_factors": [0.95, 1.0, 1.05]},
            [(8, 0.95, 90), (8, 1.0, 90), (8, 1.05, 90)],
            id="volume-scaling",
        ),
        pytest.param(
            [PRIMITIVE, CUBIC],
            {"supercell_matrix": SUPERCELL_211, "volume_custom_scale_factors": [0.98, 1.02]},
            [(4, 1.96, 60), (4, 2.04, 60), (16, 1.96, 90), (16, 2.04, 90)],
            id="supercell-and-scaling-of-two-structures",
        ),
    ],
)
def test_nvt_on_supercells_and_scaled_cells(
    structures, cell_options, expected, gap_nvt_maker, memory_jobstore, clean_dir
):
    runs = run_nvt(gap_nvt_maker, structures, memory_jobstore, **cell_options)

    # runs are ordered by input structure, then by scale factor
    n_factors = len(cell_options.get("volume_custom_scale_factors", [1.0]))
    inputs = [s for s in structures for _ in range(n_factors)]

    assert len(runs) == len(expected)
    for frames, source, (n_atoms, volume_ratio, angle) in zip(runs, inputs, expected):
        start = frames[0]
        assert len(frames) == 3  # 2 NVT steps + the starting structure
        assert len(start) == n_atoms
        assert start.get_volume() == pytest.approx(volume_ratio * source.volume)
        np.testing.assert_allclose(start.cell.angles(), angle)
        # NVT keeps the prepared cell during the whole run
        np.testing.assert_allclose([f.get_volume() for f in frames], start.get_volume())


def test_volume_scaling_is_isotropic(gap_nvt_maker, memory_jobstore, clean_dir):
    """Scaling changes only the size of the cell: shape and fractional coordinates stay."""
    factors = [0.9, 1.1]
    runs = run_nvt(gap_nvt_maker, [CUBIC], memory_jobstore, volume_custom_scale_factors=factors)

    for frames, factor in zip(runs, factors):
        start = frames[0]
        np.testing.assert_allclose(
            start.cell.lengths(), np.array(CUBIC.lattice.abc) * factor ** (1 / 3)
        )
        np.testing.assert_allclose(start.cell.angles(), CUBIC.lattice.angles)
        np.testing.assert_allclose(
            start.get_scaled_positions(), CUBIC.frac_coords % 1.0, atol=1e-8
        )


def test_ramp_step_list_sets_the_md_length(gap_nvt_maker, memory_jobstore, clean_dir):
    """300 K for 2 steps, a 5-step ramp to 400 K, 2 steps at 400 K: 9 steps, 10 frames."""
    md_job = ScaledMDMaker(
        md_maker=gap_nvt_maker,
        temperature_list=[300.0, 400.0],
        eqm_step_list=[2, 2],
        ramp_step_list=[5],
    ).make(CUBIC)
    run_locally(md_job, create_folders=True, ensure_success=True, store=memory_jobstore)

    frames = read(md_job.output.resolve(memory_jobstore)[0], index=":")
    assert len(frames) == 10
