import os 
os.environ["OMP_NUM_THREADS"] = "1"

from autoplex.data.rss.jobs import RandomizedStructure, do_rss_single_node, do_rss_multi_node
from autoplex.data.common.jobs import sample_data
from jobflow import run_locally
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np


def test_extract_elements():
    rs = RandomizedStructure()
    elements = rs._extract_elements("SiO2")
    assert elements == {"Si": 1, "O": 2}
    
    elements = rs._extract_elements("H2O")
    assert elements == {"H": 2, "O": 1}
    
    elements = rs._extract_elements("C6H12O6")
    assert elements == {"C": 6, "H": 12, "O": 6}


def test_make_species():
    rs = RandomizedStructure()
    elements = {"Si": 1, "O": 2}
    species = rs._make_species(elements)
    assert species == "Si%NUM=1,O%NUM=2"
    
    elements = {"H": 2, "O": 1}
    species = rs._make_species(elements)
    assert species == "H%NUM=2,O%NUM=1"


def test_is_metal():
    rs = RandomizedStructure()
    assert rs._is_metal("Fe") == True
    assert rs._is_metal("Si") == False


def test_make_minsep():
    rs = RandomizedStructure()
    radii = {"Si": 1.1, "O": 0.66}
    minsep = rs._make_minsep(radii)
    assert "Si-Si=1.7600000000000002" in minsep  # r1 * 1.8
    assert "Si-O=1.4080000000000004" in minsep  # (r1 + r2) / 2 * 1.5
    assert "O-O=1.056" in minsep   # r1 * 1.8


def test_update_buildcell_options():
    rs = RandomizedStructure()
    options = {'VARVOL': 20, 'SPECIES': 'Si%NUM=1,O%NUM=2'}
    buildcell_parameters = ['VARVOL=15',
                            'NFORM=1-7',
                            ]
    buildcell_update = rs._update_buildcell_option(options, buildcell_parameters)
    print("Updated buildcell parameters:", buildcell_update)
    assert 'VARVOL=20' in buildcell_update
    assert 'SPECIES=Si%NUM=1,O%NUM=2' in buildcell_update


def test_output_from_scratch(memory_jobstore, clean_dir):
    from jobflow import run_locally
    from ase.io import read
    from pathlib import Path
    import shutil
    job = RandomizedStructure(struct_number=3,
                              tag='SiO2',
                              output_file_name='random_structs.extxyz',
                              buildcell_option={'VARVOL': 20,
                                                'SYMMOPS':'1-2'},
                              num_processes=4).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert len(read(job.output.resolve(memory_jobstore), index=":")) == 3

        
def test_fragment_buildcell(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from ase.io import read
    from pathlib import Path
    import shutil
    import numpy as np
    from ase.build import molecule
    from ase.io import write
    
    ice_density = 0.0307 # molecules/A^3
    h2o = molecule('H2O')
    h2o.arrays['fragment_id'] = np.array([0,0,0])
    h2o.cell = np.ones(3)*20
    write(f'{test_dir}/data/h2o.xyz', h2o)
    
    job = RandomizedStructure(struct_number=4,
                              tag='water',
                              output_file_name='random_h20_structs.extxyz',
                              buildcell_option={'TARGVOL': f'{1/ice_density*0.8}-{1/ice_density*1.2}',
                                                'SYMMOPS': '1-4',
                                                'NFORM': '500',
                                                'MINSEP': '2.0',
                                                'SLACK': 0.25,
                                                'OVERLAP': 0.1,
                                                'SYSTEM': 'Cubi'
                                                },
                              fragment_file=os.path.join(f'{test_dir}/data', 'h2o.xyz'),
                              fragment_numbers=None,
                              remove_tmp_files=True,
                              num_processes=4).make()
    
    _ = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    ats = read(job.output.resolve(memory_jobstore), index=":")
    assert len(ats) == 4 and np.all(ats[0].positions[0] != ats[0].positions[1])


def test_output_from_cell_seed(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from ase.io import read
    from pathlib import Path
    import shutil
    test_files_dir = test_dir / "data/SiO2.cell"
    job = RandomizedStructure(struct_number=3,
                              cell_seed_path=test_files_dir,
                              num_processes=3).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert len(read(job.output.resolve(memory_jobstore),index=":")) == 3


def test_build_multi_randomized_structure(memory_jobstore, clean_dir):
    from autoplex.data.rss.flows import BuildMultiRandomizedStructure
    from jobflow import run_locally, Flow
    from autoplex.data.common.utils import flatten
    from pathlib import Path
    import shutil
    bcur_params={'soap_paras': {'l_max': 3,
                                'n_max': 3,
                                'atom_sigma': 0.5,
                                'cutoff': 4.0,
                                'cutoff_transition_width': 1.0,
                                'zeta': 4.0,
                                'average': True,
                                'species': True,
                                },
                }
    generate_structure = BuildMultiRandomizedStructure(tag="Si",
        generated_struct_numbers=[50,50],
        buildcell_options=[{'VARVOL': 20, 
                            'VARVOL_RANGE': '0.75 1.25',
                            'NATOM': '{6,8,10,12,14,16,18,20,22,24}',
                            'NFORM': '1'}, 
                           {'SYMMOPS':'1-2',
                            'NATOM': '{7,9,11,13,15,17,19,21,23}',
                            'NFORM': '1'}],
        num_processes=8,
        initial_selection_enabled=True,
        selected_struct_numbers=[8,2],
        bcur_params=bcur_params,
        random_seed=None).make()

    job = Flow(generate_structure, output=generate_structure.output) 
    responses = run_locally(job, 
                            ensure_success=True, 
                            create_folders=True, 
                            store=memory_jobstore)

    structures = job.output.resolve(memory_jobstore)

    n_atoms = [struct.num_sites for struct in flatten(structures, recursive=False)]

    assert max(n_atoms) < 25

    even_count = sum(1 for n in n_atoms if n % 2 == 0)
    odd_count = sum(1 for n in n_atoms if n % 2 != 0)

    assert even_count == 8
    assert odd_count == 2


def test_vasp_static(test_dir, memory_jobstore, clean_dir):
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


def test_gap_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    job = do_rss_single_node(mlip_type='GAP',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 2


def test_gap_rss_multi_jobs(test_dir, memory_jobstore, clean_dir):
    from ase.units import GPa
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:2:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    job = do_rss_multi_node(mlip_type='GAP',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=1000,
                force_tol=0.01,
                stress_tol=0.0001,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938},
                num_groups=2,)
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i:
            output_filter.append(i)
   
    assert len(output_filter) == 2

    ats = read(output_filter[0][0])

    enthalpy_pseudo = ats.info["enthalpy"]
    enthalpy_cal = ats.get_potential_energy() + ats.info["RSS_applied_pressure"]*GPa*ats.get_volume()
    
    assert round(enthalpy_pseudo,3) == round(enthalpy_cal,3)


def test_nequip_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/NEQUIP"

    job = do_rss_single_node(mlip_type='NEQUIP',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_m3gnet_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/M3GNET/m3gnet_results/training"

    job = do_rss_single_node(mlip_type='M3GNET',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_mace_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/MACE"

    job = do_rss_single_node(mlip_type='MACE',
                iteration_index='0',
                mlip_path=mlip_path,
                structures=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isolated_atom_energies={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_sampling_cur_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job = sample_data(
        selection_method='cur',
        num_of_selection=5,
        bcur_params={'soap_paras': {'l_max': 3,
                                    'n_max': 3,
                                    'atom_sigma': 0.5,
                                    'cutoff': 4.0,
                                    'cutoff_transition_width': 1.0,
                                    'zeta': 4.0,
                                    'average': True,
                                    'species': True,
                                    },
                      },
        structure=structures,
    )
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_bcur1s_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='bcur1s',
                   num_of_selection=5,
                   bcur_params={'soap_paras': {'l_max': 3,
                                'n_max': 3,
                                'atom_sigma': 0.5,
                                'cutoff': 4.0,
                                'cutoff_transition_width': 1.0,
                                'zeta': 4.0,
                                'average': True,
                                'species': True,
                                },
                                'frac_of_bcur': 0.8,
                                'energy_label': 'REF_energy'
                    },
                    structure=structures, 
                    isolated_atom_energies={14: -0.84696938},
                    random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_random_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='random',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_sampling_uniform_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sample_data(selection_method='uniform',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))


def test_vasp_check_convergence(test_dir):
    from autoplex.data.common.jobs import check_convergence_vasp
    test_files_dir = test_dir / "vasp/rss/Si_bulk_1/outputs"
    converged = check_convergence_vasp(os.path.join(test_files_dir, 'vasprun.xml.gz'))
    assert converged == True 