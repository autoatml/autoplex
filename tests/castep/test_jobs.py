import os
os.environ["CASTEP_COMMAND"] = "/usr/local/CASTEP-20/castep.mpi"
os.environ["CASTEP_PP_PATH"] = "/usr/local/CASTEP-20/usp"
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import run_locally, Flow
from autoplex.castep.jobs import BaseCastepMaker, CastepStaticMaker
from ase.calculators.castep import Castep
from autoplex.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
import shutil
import glob


def test_BaseCastepMaker(memory_jobstore):
    atoms = Atoms('Si', positions=[[10.0, 10.0, 10.0]], cell=[20, 20, 20], pbc=True)
    calc = Castep()
    calc.param.cut_off_energy = 100.0
    calc.param.xc_functional = 'PBE'
    calc.param.task = 'SinglePoint'
    calc.param.max_scf_cycles = 100
    calc.param.elec_energy_tol = 1e-6
    calc.cell.kpoint_mp_grid = '1 1 1'
    calc.cell.kpoint_mp_offset = '0.0 0.0 0.0'
    calc.param.spin_polarized = False
    
    atoms.calc = calc
    ase_energy = atoms.get_potential_energy()
    print('ase_energy:', ase_energy)
    
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
    
    assert abs(ase_energy - dict_castep["energy"]) < 1e-4
    
    for d in glob.glob("job_*") + glob.glob("CASTEP*"):
        shutil.rmtree(d, ignore_errors=True)
    

def test_CastepStaticMaker(memory_jobstore):
    atoms = Atoms('Si', positions=[[10.0, 10.0, 10.0]], cell=[20, 20, 20], pbc=True)
    calc = Castep()
    calc.param.cut_off_energy = 100.0
    calc.param.xc_functional = 'PBE'
    calc.param.task = 'SinglePoint'
    calc.param.max_scf_cycles = 100
    calc.param.elec_energy_tol = 1e-6
    calc.param.smearing_width=0.05
    calc.param.basis_precision='precise'
    calc.cell.kpoint_mp_grid = '1 1 1'
    calc.cell.kpoint_mp_offset = '0.0 0.0 0.0'
    calc.param.spin_polarized = False

    atoms.calc = calc
    ase_energy = atoms.get_potential_energy()
    print('ase_energy:', ase_energy)

    pmg_structure = AseAtomsAdaptor.get_structure(atoms)

    static_job = CastepStaticMaker(
        name="test_static",
        input_set_generator=CastepStaticSetGenerator(
            user_param_settings={
                "cut_off_energy": 100.0,
                "xc_functional": "PBE",
                "max_scf_cycles": 100,
                "elec_energy_tol": 1e-6,
                "spin_polarized": False,
            },
            user_cell_settings={
                "kpoint_mp_grid": "1 1 1",
                "kpoint_mp_offset": "0.0 0.0 0.0",
            }
        )
    ).make(structure=pmg_structure)

    flow = Flow(static_job, output=static_job.output)
    run_locally(flow,
                ensure_success=True,
                create_folders=True,
                store=memory_jobstore)

    dict_static = static_job.output.resolve(memory_jobstore)

    assert abs(ase_energy - dict_static["energy"]) < 1e-4

    for d in glob.glob("job_*") + glob.glob("CASTEP*"):
        shutil.rmtree(d, ignore_errors=True)
        