from pathlib import Path
from ase.build import bulk
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.data.md.flows import MDAseMaker
from jobflow import run_locally
import pytest
from ase import units

try:
    import mace  # noqa: F401

    has_mace = True
except ImportError:
    has_mace = False

@pytest.mark.skipif(not has_mace, reason="MACE is not installed")
def test_mdasemaker_mace_nvt(memory_jobstore, clean_dir):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    md_job = MDAseMaker(
        force_field_name="MLFF.MACE_MPA_0",
        calculator_kwargs={"device": "cpu"},
        temperature_list=[300.0],
        eqm_step_list=[2],
        time_step=1.0,
        traj_interval=1,
        ensemble="nvt",
        dynamics="langevin",
        traj_file="MD.traj",
        traj_file_fmt="ase",
        store_trajectory="partial",
    ).make(structure)

    run_locally(
        md_job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )
    
    traj_path = md_job.output.resolve(memory_jobstore)
    
    assert Path(traj_path[0]).exists()
    
    frames = read(traj_path[0], index=":")
    
    assert len(frames) == 3

def test_mdasemaker_gap_npt(test_dir, memory_jobstore, clean_dir):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    md_job = MDAseMaker(
        force_field_name="MLFF.GAP",
        calculator_kwargs={
            "param_filename": str(test_dir / "fitting/GAP/gap_file.xml")
        },
        temperature_list=[300.0],
        eqm_step_list=[5],
        time_step=1.0,
        traj_interval=1,
        ensemble="npt",
        dynamics="nose-hoover-chain",  # atomate2's NPT default (ASE MTKNPT)
        pressure=0.0,  # in kbar; required for NPT
        # MTKNPT needs thermostat/barostat damping times (ASE units)
        ase_md_kwargs={"tdamp": 100 * units.fs, "pdamp": 1000 * units.fs},
        traj_file="MD.traj",
        traj_file_fmt="ase",
        store_trajectory="partial",
    ).make(structure)

    run_locally(
        md_job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    traj_path = md_job.output.resolve(memory_jobstore)

    assert Path(traj_path[0]).exists()

    frames = read(traj_path[0], index=":")

    assert len(frames) == 6  # 5 MD steps + the starting structure
    # NPT lets the cell change (NVT keeps it fixed)
    assert frames[-1].get_volume() != pytest.approx(frames[0].get_volume())

