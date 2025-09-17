from atomate2.forcefields.jobs import ForceFieldStaticMaker
from autoplex.settings import RssConfig
from autoplex.auto.rss.flows import RssMaker
from jobflow import Flow, run_locally
from autoplex.castep.jobs import BaseCastepMaker
from ase.calculators.castep import create_castep_keywords
import logging
import os
import shutil
from pathlib import Path


def setup_castep_keywords(castep_path='/usr/local/CASTEP-20/castep.mpi', keywords_file='castep_keywords.json'):
    """Pre-generate CASTEP keywords file in the main process"""
    print(f"Checking CASTEP executable: {castep_path}")
    print(f"CASTEP exists: {os.path.exists(castep_path)}")
    print(f"CASTEP is executable: {os.access(castep_path, os.X_OK) if os.path.exists(castep_path) else False}")
    
    if not os.path.exists(keywords_file):
        print("Generating CASTEP keywords file...")
        try:
            success = create_castep_keywords(
                castep_command=castep_path,
                filename=keywords_file,
                force_write=True,
                path='.'
            )
            if success:
                print(f"✓ Created {keywords_file}")
                
                # Also copy to ASE directory for future use
                ase_dir = os.path.expanduser('~/.ase')
                if not os.path.exists(ase_dir):
                    os.makedirs(ase_dir)
                shutil.copy(keywords_file, os.path.join(ase_dir, keywords_file))
                print(f"✓ Copied to {ase_dir}")
            else:
                print("✗ Failed to generate keywords file")
                return False
        except Exception as e:
            print(f"✗ Error generating keywords: {e}")
            return False
    else:
        print(f"✓ Keywords file already exists: {keywords_file}")
    
    return True


def create_castep_makers(castep_command='/usr/local/CASTEP-20/castep.mpi', 
                        cut_off_energy=200.0, 
                        kspacing=0.5,
                        isolated_cut_off_energy=150.0,
                        isolated_kspacing=2.0):
    """Create CASTEP makers for bulk and isolated atom calculations"""
    castep_maker = BaseCastepMaker(
        name="castep static",
        castep_command=castep_command,
        cut_off_energy=cut_off_energy,
        kspacing=kspacing,
        xc_functional='PBE',
        task='SinglePoint'
    )
    
    castep_isolated_maker = BaseCastepMaker(
        name="castep isolated",
        castep_command=castep_command,
        cut_off_energy=isolated_cut_off_energy,
        kspacing=isolated_kspacing,
        xc_functional='PBE',
        task='SinglePoint'
    )
    
    return castep_maker, castep_isolated_maker


def get_default_rss_config():
    """Get default RSS configuration dictionary"""
    return {
        'tag': 'Si', 
        'train_from_scratch': True,
        'resume_from_previous_state': {
            'test_error': None, 
            'pre_database_dir': None, 
            'mlip_path': None,
            'isolated_atom_energies': None
        }, 
        'generated_struct_numbers': [100],
        'buildcell_options': [
            {
                'ABFIX': False, 
                'NFORM': '1', 
                'SYMMOPS': '1-4', 
                'SYSTEM': None, 
                'SLACK': 0.25, 
                'OCTET': False,
                'OVERLAP': 0.1, 
                'MINSEP': None, 
                'NATOM': '{6,8,10,12}'
            }
        ],
        'fragment_file': None, 
        'fragment_numbers': None, 
        'num_processes_buildcell': 2,
        'num_of_initial_selected_structs': [50],
        'num_of_rss_selected_structs': 1,
        'initial_selection_enabled': True, 
        'rss_selection_method': 'bcur2i', 
        'bcur_params': {
            'soap_paras': {
                'l_max': 12, 
                'n_max': 12, 
                'atom_sigma': 0.0875, 
                'cutoff': 10.5,
                'cutoff_transition_width': 1.0, 
                'zeta': 4.0, 
                'average': True, 
                'species': True
            },
            'frac_of_bcur': 0.8, 
            'bolt_max_num': 3000
        }, 
        'random_seed': None, 
        'include_isolated_atom': True,
        'isolatedatom_box': [20.0, 20.0, 20.0], 
        'e0_spin': False, 
        'include_dimer': False,
        'dimer_box': [20.0, 20.0, 20.0], 
        'dimer_range': [1.0, 5.0], 
        'dimer_num': 21,
        'static_energy_maker_isolated_species_spin_polarization': None, 
        'vasp_ref_file': 'vasp_ref.extxyz',
        'castep_ref_file': 'castep_ref.extxyz',
        'calculator_type': 'castep',
        'config_types': ['initial', 'traj_early', 'traj'], 
        'rss_group': ['traj'], 
        'test_ratio': 0.0,
        'regularization': True, 
        'retain_existing_sigma': False, 
        'scheme': 'linear-hull',
        'reg_minmax': [[0.1, 1.0], [0.001, 0.1], [0.0316, 0.316], [0.0632, 0.632]], 
        'distillation': False,
        'force_max': None, 
        'force_label': None, 
        'pre_database_dir': None, 
        'mlip_type': 'GAP',
        'ref_energy_name': 'REF_energy', 
        'ref_force_name': 'REF_forces', 
        'ref_virial_name': 'REF_virial',
        'auto_delta': True, 
        'num_processes_fit': 16, 
        'device_for_fitting': 'cpu',
        'scalar_pressure_method': 'uniform', 
        'scalar_exp_pressure': 1,
        'scalar_pressure_exponential_width': 0.2, 
        'scalar_pressure_low': 0, 
        'scalar_pressure_high': 25,
        'max_steps': 200, 
        'force_tol': 0.01, 
        'stress_tol': 0.01, 
        'stop_criterion': 1e-14,
        'max_iteration_number': 2, 
        'num_groups': 6, 
        'initial_kt': 0.3, 
        'current_iter_index': 1,
        'hookean_repul': False, 
        'hookean_paras': None, 
        'keep_symmetry': False, 
        'write_traj': True,
        'num_processes_rss': 2, 
        'device_for_rss': 'cpu',
        'mlip_hypers': {
            'GAP': {
                'general': {
                    'at_file': 'train.extxyz', 
                    'default_sigma': '{0.0001 0.05 0.05 0}',
                    'energy_parameter_name': 'REF_energy', 
                    'force_parameter_name': 'REF_forces',
                    'virial_parameter_name': 'REF_virial', 
                    'sparse_jitter': 1e-08, 
                    'do_copy_at_file': 'F',
                    'openmp_chunk_size': 10000, 
                    'gp_file': 'gap_file.xml', 
                    'e0_offset': 0.0, 
                    'two_body': False,
                    'three_body': False, 
                    'soap': True
                },
                'twob': {
                    'distance_Nb_order': 2, 
                    'f0': 0.0, 
                    'add_species': 'T', 
                    'cutoff': 5.0, 
                    'n_sparse': 15,
                    'covariance_type': 'ard_se', 
                    'delta': 2.0, 
                    'theta_uniform': 0.5, 
                    'sparse_method': 'uniform',
                    'compact_clusters': 'T'
                },
                'threeb': {
                    'distance_Nb_order': 3, 
                    'f0': 0.0, 
                    'add_species': 'T', 
                    'cutoff': 3.25, 
                    'n_sparse': 100,
                    'covariance_type': 'ard_se', 
                    'delta': 2.0, 
                    'theta_uniform': 1.0, 
                    'sparse_method': 'uniform',
                    'compact_clusters': 'T'
                },
                'soap': {
                    'add_species': 'T', 
                    'l_max': 6, 
                    'n_max': 8, 
                    'atom_sigma': 0.5, 
                    'zeta': 4, 
                    'cutoff': 5.0,
                    'cutoff_transition_width': 1.0, 
                    'central_weight': 1.0, 
                    'n_sparse': 1000, 
                    'delta': 1.0, 
                    'f0': 0.0,
                    'covariance_type': 'dot_product', 
                    'sparse_method': 'cur_points'
                }
            },
        },
    }


def create_and_validate_rss_config(config_overrides=None):
    """Create and validate RSS configuration"""
    config = get_default_rss_config()
    
    # Apply any overrides
    if config_overrides:
        config.update(config_overrides)
    
    # Validate using Pydantic model
    rss_config_obj = RssConfig.model_validate(config)
    return rss_config_obj


def run_rss_workflow(rss_config_obj, castep_maker, castep_isolated_maker, job_name="rss_castep_final"):
    """Run the RSS workflow with given configuration and makers"""
    # Creating RSS job with CASTEP makers
    rss_job = RssMaker(
        name=job_name, 
        rss_config=rss_config_obj, 
        static_energy_maker=castep_maker,
        static_energy_maker_isolated_atoms=castep_isolated_maker
    ).make()

    print("Starting RSS workflow...")
    responses = run_locally(
        Flow(jobs=[rss_job], output=rss_job.output),
        create_folders=True,
        ensure_success=True,
    )
    
    return responses


def main():
    """Main function to run the complete RSS workflow"""
    # Pre-generate keywords file in main process
    print("Setting up CASTEP environment...")
    if not setup_castep_keywords():
        print("Failed to setup CASTEP keywords. Exiting.")
        return False
    
    # Creating CASTEP makers
    print("Setting up RSS workflow...")
    castep_maker, castep_isolated_maker = create_castep_makers()
    
    # Create and validate RSS configuration
    rss_config_obj = create_and_validate_rss_config()
    
    # Run the workflow
    try:
        responses = run_rss_workflow(rss_config_obj, castep_maker, castep_isolated_maker)
        print("RSS workflow completed!")
        return True
    except Exception as e:
        print(f"RSS workflow failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)