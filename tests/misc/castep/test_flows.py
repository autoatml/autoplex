from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.misc.castep.jobs import CastepStaticMaker, CastepMagresMaker
from autoplex.misc.castep.utils import CastepStaticSetGenerator, CastepMagresSetGenerator
from autoplex.data.common.jobs import collect_dft_data
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Response
from autoplex.misc.castep.flows import CastepMagresFlowMaker
import numpy as np
def test_DFTStaticLabelling_with_castep(memory_jobstore, mock_castep, clean_dir):
    
    ref_paths = {
        "static_bulk_0": "static/CASTEP_bulk1",
        "static_bulk_1": "static/CASTEP_bulk2",
    }
    
    mock_castep(ref_paths)
    
    atoms1 = bulk("Si", "diamond", a=5.1)
    atoms2 = bulk("Si", "diamond", a=5.2)
    struct1 = AseAtomsAdaptor.get_structure(atoms1)
    struct2 = AseAtomsAdaptor.get_structure(atoms2)

    structures = [struct1, struct2]
    
    castep_maker = CastepStaticMaker(
        name="test_castep",
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
        ),
    )

    job_dft = DFTStaticLabelling(
        isolated_atom=False,
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
    
    path_to_vasp, _ = dict_dft['dft_ref_dir'], dict_dft['isolated_atom_energies']
    
    atoms = read(path_to_vasp, index=":")
    config_types = [at.info['config_type'] for at in atoms]
    
    assert len(config_types) == 2


def test_MagresLabelling_with_castep(memory_jobstore, mock_castep, castep_test_dir, clean_dir):
    """
    Test to see if MagresMaker works on multiple structures, using simplified mock version of DFTLabelling
    """
    ref_paths = {
        "magres_1": "magres/CASTEP_SNO_1",
        "magres_2": "magres/CASTEP_SNO_2",
    }

    mock_castep(ref_paths)

    ref_out = castep_test_dir / "magres" / "CASTEP_SNO_1" / "outputs"
    struct1 = AseAtomsAdaptor.get_structure(read(ref_out / "castep.castep"))

    ref_out = castep_test_dir / "magres" / "CASTEP_SNO_2" / "outputs"
    struct2 = AseAtomsAdaptor.get_structure(read(ref_out / "castep.castep"))

    structures = [struct1,struct2]
    
    
    magres_maker = CastepMagresMaker(
                    name="magres",
                    #gives the base name of the job (jobs in flow will be called name1,name2....)
                    input_set_generator=CastepMagresSetGenerator(
                        useEFG=False,           # the run was shielding-only
                        user_param_settings={"xc_functional": "R2SCAN", "cut_off_energy": 1000.0},
                        user_cell_settings={"kpoint_mp_grid": "5 5 4"}
                    )
                )
    magres_flow = CastepMagresFlowMaker(magres_maker = magres_maker).make(structures)
    
    run_locally(
        magres_flow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    dicts = [magres_job.output.resolve(memory_jobstore) for magres_job in magres_flow.output]
    
    assert len(dicts) == 2
    np.testing.assert_allclose(
        dicts[1].ms_tensor[0],
        [[20.7015, 0.0, 0.0], [0.0, 20.7015, 0.0], [0.0, 0.0, -0.9801]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        dicts[0].ms_tensor[0],
        [[23.2507, 0.0, 0.0], [0.0, 23.2507, 0.0], [0.0, 0.0, 0.9379]],
        atol=1e-4,
    )

"""
job_list = []
dirs = []
for idx, struct in enumerate(structures):
    magres_maker = CastepMagresMaker(
        name=f"magres{idx+1}",
        input_set_generator=CastepMagresSetGenerator(
            useEFG=False,           # the run was shielding-only
            user_param_settings={"xc_functional": "R2SCAN", "cut_off_energy": 1000.0},
            user_cell_settings={"kpoint_mp_grid": "5 5 4"}
        )
    )
    magres_job = magres_maker.make(structure=struct)

    job_list.append(magres_job)
    dirs.append(magres_job.output)

"""