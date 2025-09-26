import os
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import run_locally, Flow
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from ase.calculators.castep import Castep
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
import shutil
import glob
from ase.build import bulk


def test_BaseCastepMaker(test_dir, memory_jobstore, mock_castep):
    
    ref_paths = {
        "test_castep": "CASTEP_bulk1"
    }
    
    mock_castep(ref_paths)

    atoms = bulk("Si", "diamond", a=5.1)
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)
    
    castep_job = BaseCastepMaker(
        name="test_castep",
        input_set_generator=CastepInputGenerator(
            user_param_settings={
            'cut_off_energy': 100.0,
            'xc_functional': 'PBE',
            'task': 'SinglePoint',
            'max_scf_cycles': 100,
            },
            user_cell_settings={
            'kpoint_mp_grid': '1 1 1',
            'kpoint_mp_offset': '0.0 0.0 0.0',
            }
        )
    ).make(structure=pmg_structure)
    
    job_rss = Flow(castep_job, output=castep_job.output)
    run_locally(job_rss,
                ensure_success=True,
                create_folders=True,
                store=memory_jobstore)
    
    dict_castep = castep_job.output.resolve(memory_jobstore)
    
    assert abs(-329.6080395967 - dict_castep.output.energy) < 1e-4
    
    for d in glob.glob("job_*") + glob.glob("CASTEP*"):
        shutil.rmtree(d, ignore_errors=True)
    

def test_CastepStaticMaker(test_dir, memory_jobstore, mock_castep):
    
    ref_paths = {
        "test_static": "CASTEP_bulk1"
    }
    
    mock_castep(ref_paths)
    
    atoms = bulk("Si", "diamond", a=5.1)
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)

    static_job = CastepStaticMaker(
        name="test_static",
        input_set_generator=CastepStaticSetGenerator(
            user_param_settings={
            'cut_off_energy': 100.0,
            'xc_functional': 'PBE',
            'task': 'SinglePoint',
            'max_scf_cycles': 100,
            },
            user_cell_settings={
            'kpoint_mp_grid': '1 1 1',
            'kpoint_mp_offset': '0.0 0.0 0.0',
            }
        )
    ).make(structure=pmg_structure)

    flow = Flow(static_job, output=static_job.output)
    run_locally(flow,
                ensure_success=True,
                create_folders=True,
                store=memory_jobstore)

    dict_static = static_job.output.resolve(memory_jobstore)

    assert abs(-329.6080395967 - dict_static.output.energy) < 1e-4

    for d in glob.glob("job_*") + glob.glob("CASTEP*"):
        shutil.rmtree(d, ignore_errors=True)
        