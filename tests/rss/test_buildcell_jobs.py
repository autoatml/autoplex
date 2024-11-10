import os 
os.environ["OMP_NUM_THREADS"] = "1"

from autoplex.data.rss.jobs import RandomizedStructure


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

        
def test_fragment_buildcell(memory_jobstore, clean_dir):
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
    write('h2o.xyz', h2o)
    
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
                              fragment_file=os.path.join(os.getcwd(), 'h2o.xyz'),
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
