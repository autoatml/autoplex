from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.misc.castep.jobs import CastepStaticMaker
from autoplex.misc.castep.utils import CastepStaticSetGenerator
from autoplex.data.common.jobs import collect_dft_data
from pymatgen.io.ase import AseAtomsAdaptor


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
        include_isolated_atom=False,
        include_dimer=False,
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