from pathlib import Path
from ase.build import bulk
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.data.md.flows import MDAseMaker
from jobflow import run_locally


def test_mdasemaker_output_structure(test_dir, mock_vasp, memory_jobstore, clean_dir):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    md_job = MDAseMaker(
        force_field_name="MLFF.MACE_MPA_0",
        calculator_kwargs={"device": "cpu"},
        temperature_list=[300.0],
        eqm_step_list=[5],
        rate_list=[0],
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
    
    assert len(frames) == 6