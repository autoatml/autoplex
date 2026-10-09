from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.misc.castep.jobs import CastepStaticMaker, CastepMagresMaker
from autoplex.misc.castep.utils import CastepStaticSetGenerator, CastepMagresSetGenerator
from autoplex.data.common.jobs import collect_dft_data, safe_strip_hostname
from autoplex.data.nmr.jobs import collect_nmr_data
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.flows import CastepMagresFlowMaker
import numpy as np
import os

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
    
    Reference structures from the dataset of Ben Mahmoud et al., J. Chem. Phys. 163, 024118 (2025),
    https://doi.org/10.1063/5.0274240; dataset: https://doi.org/10.5281/zenodo.15775328 (CC BY 4.0).
    
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
                            use_efg=True,           
                            user_param_settings={"xc_functional": "PBE", "cut_off_energy": 900.0},
                            user_cell_settings={"kpoint_mp_spacing": 0.05}
                        )
                    )
    magres_flow = CastepMagresFlowMaker(magres_maker = magres_maker).make(structures)

    nmr_collect_data = collect_nmr_data(nmr_dirs=magres_flow.output)

    run_locally(
        [magres_flow,nmr_collect_data],
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

   
    

    #TODO: create collect_nmr_data similiar to collect_dft_data
    
    dict_nmr = nmr_collect_data.output.resolve(memory_jobstore)
    
    path_to_castep = dict_nmr['nmr_ref_dir']
    
    structures = read(path_to_castep, index=":")
    
    assert len(structures) == 2
    np.testing.assert_allclose(
        structures[1].arrays["REF_ms"][0],
        [230.9670, 33.2024, -33.2870, 33.2469, 230.8632, -33.2890, -33.2720, -33.2131, 230.9253],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        structures[0].arrays["REF_ms"][0],
        [252.7565, 11.0014, -37.9952, 9.9670, 194.5567 , -7.3071, -38.1106, -7.7682 , 222.5173],
        atol=1e-4,
    )

    
    #test only the flow maker
    nmr_dirs = [value.resolve(memory_jobstore) for value in magres_flow.output]
    dirs = [safe_strip_hostname(value) for value in nmr_dirs] #same code as collect_dft_data
    assert len(dirs) == 2
    
    for d in dirs:
        assert os.path.basename(d) == "CASTEP"                               # dir_name points at the CASTEP folder
        assert os.path.exists(os.path.join(d, "castep.magres.gz"))           # and the NMR output is there
