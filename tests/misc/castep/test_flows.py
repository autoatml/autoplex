import os
os.environ["CASTEP_COMMAND"] = "/usr/local/CASTEP-20/castep.mpi"
os.environ["CASTEP_PP_PATH"] = "/usr/local/CASTEP-20/usp"
from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.misc.castep.jobs import CastepStaticMaker
from autoplex.misc.castep.utils import CastepStaticSetGenerator
from autoplex.data.common.jobs import collect_dft_data
from pymatgen.io.ase import AseAtomsAdaptor
import shutil
import glob


def test_DFTStaticLabelling_with_castep(memory_jobstore):
    atoms1 = bulk("Si", "diamond", a=5.43)
    atoms2 = bulk("Si", "fcc", a=5.43)
    struct1 = AseAtomsAdaptor.get_structure(atoms1)
    struct2 = AseAtomsAdaptor.get_structure(atoms2)

    structures = [struct1, struct2]
    
    castep_maker = CastepStaticMaker(
        name="static_castep",
        input_set_generator=CastepStaticSetGenerator(
            user_param_settings={
                "cut_off_energy": 200.0,
                "xc_functional": "PBE",
                "task": "SinglePoint",
                "max_scf_cycles": 200,
                "elec_energy_tol": 1e-5,
                "smearing_width": 0.1,
                "spin_polarized": False,
            },
            user_cell_settings={
                "kpoint_mp_grid": "1 1 1",
                "kpoint_mp_offset": "0.0 0.0 0.0",
            },
        ),
    )

    job_dft = DFTStaticLabelling(
        isolated_atom=True,
        e0_spin=True,
        isolatedatom_box=[20.0, 20.5, 21.0],
        dimer=False,
        static_energy_maker=castep_maker,
    ).make(structures=structures)
    
    job_collect_data = collect_dft_data(dft_dirs=job_dft.output)

    run_locally(
        Flow([job_dft, job_collect_data]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    dict_dft = job_collect_data.output.resolve(memory_jobstore)
    
    path_to_vasp, isol_energy = dict_dft['dft_ref_dir'], dict_dft['isolated_atom_energies']
    
    atoms = read(path_to_vasp, index=":")
    config_types = [at.info['config_type'] for at in atoms]

    assert abs(isol_energy['14'] - (-165.23)) < 1e-2
    assert len(config_types) == 3
    
    for d in glob.glob("job_*"):
        shutil.rmtree(d, ignore_errors=True)
        