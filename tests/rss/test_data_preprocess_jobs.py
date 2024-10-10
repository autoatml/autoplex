import os 
os.environ["OMP_NUM_THREADS"] = "1"

from jobflow import run_locally
from ase.io import read
import shutil
from pathlib import Path
import numpy as np

def test_vasp_static(test_dir, memory_jobstore):
    from autoplex.data.common.jobs import preprocess_data
    test_files_dir = test_dir / "data/rss.extxyz"

    job = preprocess_data(test_ratio=0.1,
                             regularization=True,
                             distillation=True,
                             force_max=0.7,
                             vasp_ref_dir=test_files_dir,
                             pre_database_dir=None,)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    path_to_training_data = job.output.resolve(memory_jobstore)
    atom_train = read(os.path.join(path_to_training_data, 'train.extxyz'), index=":")
    atom_test = read(os.path.join(path_to_training_data, 'test.extxyz'), index=":")

    atoms = atom_train + atom_test
    f_component_max = []
    for at in atoms:
        forces = np.abs(at.arrays["REF_forces"])
        f_component_max.append(np.max(forces))

    assert len(atom_train) == 12
    assert len(atom_test) == 2
    assert "energy_sigma" in atom_train[0].info
    assert max(f_component_max) < 0.7

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


