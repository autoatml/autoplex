from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.misc.castep.jobs import CastepStaticMaker, CastepMagresMaker
from autoplex.misc.castep.utils import CastepStaticSetGenerator, CastepMagresSetGenerator
from autoplex.data.common.jobs import collect_dft_data
from pymatgen.io.ase import AseAtomsAdaptor
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


def test_CastepMagresFlowMaker(memory_jobstore, mock_castep, castep_test_dir, clean_dir):
    """
    Tests CastepMagresFlowMaker.
    
    Example output taken from https://github.com/cbenmahm/anistropic-nmr-parameters-data,
    as described in https://pubs.aip.org/aip/jcp/article/163/2/024118/3351953/Graph-neural-network-predictions-of-solid-state.
    """
    ref_paths = {
        "test_magres_1": "magres/CASTEP_CRISTOBALITE_ALPHA",
        "test_magres_2": "magres/CASTEP_CRISTOBALITE_BETA",
    }

    mock_castep(ref_paths)

    ref_out = castep_test_dir / "magres" / "CASTEP_CRISTOBALITE_ALPHA" / "outputs"
    struct1 = AseAtomsAdaptor.get_structure(read(ref_out / "castep.castep"))

    ref_out = castep_test_dir / "magres" / "CASTEP_CRISTOBALITE_BETA" / "outputs"
    struct2 = AseAtomsAdaptor.get_structure(read(ref_out / "castep.castep"))

    structures = [struct1,struct2]
    
    
    magres_maker = CastepMagresMaker(
                        name="test_magres",
                        #gives the base name of the job (jobs in flow will be called name_1,name_2....)
                        input_set_generator=CastepMagresSetGenerator(
                            useEFG=True,           
                            user_param_settings={"xc_functional": "PBE", "cut_off_energy": 900.0},
                            user_cell_settings={"kpoint_mp_spacing": 0.05}
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
        [[230.9670, 33.2024, -33.2870], [33.2469, 230.8632, -33.2890], [-33.2720, -33.2131, 230.9253]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        dicts[0].ms_tensor[0],
        [[252.7565, 11.0014, -37.9952], [9.9670, 194.5567 , -7.3071], [-38.1106, -7.7682 , 222.5173]],
        atol=1e-4,
    )
