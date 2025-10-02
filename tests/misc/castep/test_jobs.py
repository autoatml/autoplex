from dataclasses import dataclass, field
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import run_locally, Flow
from atomate2.common.flows.elastic import BaseElasticMaker
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from ase.build import bulk


def test_BaseCastepMaker(memory_jobstore, mock_castep, clean_dir):
    
    ref_paths = {
        "test_castep": "static/CASTEP_bulk1"
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
    

def test_CastepStaticMaker(memory_jobstore, mock_castep, clean_dir):
    
    ref_paths = {
        "test_static": "static/CASTEP_bulk1"
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
    assert abs(585.86621 - dict_static.output.stress[0][0]) < 1e-4
    

def test_ElasticMaker(memory_jobstore, mock_castep, clean_dir):
    
    @dataclass
    class ElasticMaker(BaseElasticMaker): 
    
        bulk_relax_maker: BaseCastepMaker | None = field(
            default_factory=lambda: BaseCastepMaker(
                input_set_generator=CastepInputGenerator(
                user_param_settings={
                'cut_off_energy': 600.0,
                'xc_functional': 'PBE',
                'task': 'GeometryOptimization',
                'max_scf_cycles': 100,
                'calculate_stress': 'True',
                'geom_energy_tol': 1e-7,
                'geom_force_tol': 1e-5,
                "finite_basis_corr": "automatic",
                "smearing_scheme": "Gaussian",
                "smearing_width": 0.05,  
                },
                user_cell_settings={
                'kpoint_mp_spacing': 0.03,
                'symmetry_generate': True,
                'symmetry_tol' : 1.0e-5,
                }
            ))
        )
        elastic_relax_maker:  BaseCastepMaker | None = field(
            default_factory=lambda: BaseCastepMaker(
                input_set_generator=CastepInputGenerator(
                user_param_settings={
                'cut_off_energy': 600.0,
                'xc_functional': 'PBE',
                'task': 'GeometryOptimization',
                'max_scf_cycles': 100,
                'calculate_stress': 'True',
                'geom_energy_tol': 1e-7,
                'geom_force_tol': 1e-5,
                "finite_basis_corr": "automatic",
                "smearing_scheme": "Gaussian",
                "smearing_width": 0.05,  
                },
                user_cell_settings={
                'kpoint_mp_spacing': 0.03,
                'symmetry_generate': True,
                'symmetry_tol' : 1.0e-5,
                'fix_all_cell': "True",
                }
            ))
        )
        
        @property
        def prev_calc_dir_argname(self) -> str | None:
            """Name of argument informing static maker of previous calculation directory.

            As this differs between different DFT codes (e.g., VASP, CP2K), it
            has been left as a property to be implemented by the inheriting class.

            Note: this is only applicable if a relax_maker is specified; i.e., two
            calculations are performed for each ordering (relax -> static)
            """
            return None
    
    
    ref_paths = {
        "castep_job": "elastic/CASTEP_relax",
        "castep_job 1/6": "elastic/CASTEP_deform_job1",
        "castep_job 2/6": "elastic/CASTEP_deform_job2",
        "castep_job 3/6": "elastic/CASTEP_deform_job3",
        "castep_job 4/6": "elastic/CASTEP_deform_job4",
        "castep_job 5/6": "elastic/CASTEP_deform_job5",
        "castep_job 6/6": "elastic/CASTEP_deform_job6",
    }
    
    mock_castep(ref_paths)

    atoms = bulk("Si", "diamond", a=5.1)
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)

    elastic_job = ElasticMaker().make(structure=pmg_structure)

    responses=run_locally(Flow([elastic_job], output=elastic_job.output),
                ensure_success=True,
                create_folders=True,
                store=memory_jobstore)
    
    dict_elastic = elastic_job.output.resolve(memory_jobstore)
    dp = dict_elastic.derived_properties
    k_voigt = dp.k_voigt
    k_reuss   = dp.k_reuss
    k_vrh  = dp.k_vrh
    g_voigt = dp.g_voigt
    g_reuss = dp.g_reuss
    g_vrh = dp.g_vrh
    
    # The reference data were obtained from the Materials Project (MP-149): 
    # https://next-gen.materialsproject.org/materials/mp-149.
    
    assert abs(k_voigt - 89) < 1
    assert abs(k_reuss - 89) < 1
    assert abs(k_vrh - 89) < 1
    assert abs(g_voigt - 64) < 1
    assert abs(g_reuss - 61) < 1
    assert abs(g_vrh - 62) < 1
    