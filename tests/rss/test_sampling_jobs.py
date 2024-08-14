import os 
os.environ["OMP_NUM_THREADS"] = "1"

from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.data.common.jobs import sampling
from jobflow import run_locally
from pathlib import Path
from ase.io import read
from autoplex.data.common.utils import (cur_select,
                                        boltzhist_cur_one_shot,
                                        ElementCollection,
                                        boltzhist_cur_dualIter)
import shutil


def test_sampling_cur(test_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    num_of_selection=5
    soap_paras = {'l_max': 3,
                  'n_max': 3,
                  'atom_sigma': 0.5,
                  'cutoff': 3.0,
                  'cutoff_transition_width': 1.0,
                  'zeta': 4.0,
                  'average': True,
                  'species': True,
                 }
    n_species = ElementCollection(atoms).get_number_of_species()
    species_Z = ElementCollection(atoms).get_species_Z()
    descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                ' n_max=' + str(soap_paras['n_max']) + \
                ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                ' cutoff=' + str(soap_paras['cutoff']) + \
                ' n_species=' + str(n_species) + \
                ' species_Z=' + species_Z + \
                ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                ' average=' + str(soap_paras['average'])

    selected_atoms = cur_select(atoms=atoms,
                                selected_descriptor=descriptor,
                                kernel_exponent=4,
                                select_nums=num_of_selection,
                                stochastic=True,
                                random_seed=42)
    
    ref_energies = [-45.44429771,
                    -50.33287125,
                    -29.98566279,
                    -38.71543373,
                    -42.31881099]
    
    energies = [at.info['REF_energy'] for at in selected_atoms]
    
    assert energies == ref_energies


def test_sampling_cur_job(test_dir, memory_jobstore):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job = sampling(
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

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_sampling_bcur1s(test_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    num_of_selection=5
    soap_paras = {'l_max': 3,
                  'n_max': 3,
                  'atom_sigma': 0.5,
                  'cutoff': 3.0,
                  'cutoff_transition_width': 1.0,
                  'zeta': 4.0,
                  'average': True,
                  'species': True,
                 }
    n_species = ElementCollection(atoms).get_number_of_species()
    species_Z = ElementCollection(atoms).get_species_Z()
    descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                ' n_max=' + str(soap_paras['n_max']) + \
                ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                ' cutoff=' + str(soap_paras['cutoff']) + \
                ' n_species=' + str(n_species) + \
                ' species_Z=' + species_Z + \
                ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                ' average=' + str(soap_paras['average'])

    selected_atoms = boltzhist_cur_one_shot(atoms=atoms,
                                            isolated_atoms_energies={14: -0.81},
                                            bolt_frac=0.3,
                                            bolt_max_num=3000,
                                            cur_num=num_of_selection,
                                            kernel_exp=4,
                                            kT=0.1,
                                            energy_label='energy',
                                            pressures=None,
                                            descriptor=descriptor,
                                            random_seed=42,
                                            )
    
    ref_energies = [-32.294567459921545, 
                    -104.74329476438821, 
                    -43.23941249668027, 
                    -29.713705024188414, 
                    -49.96683567883993]
    
    energies = [at.get_potential_energy() for at in selected_atoms]

    assert energies == ref_energies

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_sampling_bcur2i():
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.units import fs
    from autoplex.data.common.utils import flatten
    import numpy as np
    a_0 = 4.05
    num_trajs = 10
    atoms = []
    pressures = []
    np.random.seed(42)
    for i in range(num_trajs):
        lattice_constant = a_0 + 0.01 * i
        at = bulk('Al', 'fcc', a=lattice_constant)
        at.set_calculator(EMT())
        MaxwellBoltzmannDistribution(at, temperature_K=100)
        dyn = VelocityVerlet(at, 1 * fs)
        current_trajectory = []
        p = []
        for step in range(10):
            dyn.run(1)
            current_atoms = at.copy() 
            current_atoms.info['energy'] = at.get_total_energy()
            current_atoms.info['unique_starting_index'] = f"{i}{i}"
            current_trajectory.append(current_atoms)
            p.append(0.1*i)

        atoms.append(current_trajectory)
        pressures.append(p)

    soap_paras = {'l_max': 3,
                  'n_max': 3,
                  'atom_sigma': 0.5,
                  'cutoff': 3.0,
                  'cutoff_transition_width': 1.0,
                  'zeta': 4.0,
                  'average': True,
                  'species': True,
                 }
    n_species = ElementCollection(flatten(atoms)).get_number_of_species()
    species_Z = ElementCollection(flatten(atoms)).get_species_Z()
    descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                ' n_max=' + str(soap_paras['n_max']) + \
                ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                ' cutoff=' + str(soap_paras['cutoff']) + \
                ' n_species=' + str(n_species) + \
                ' species_Z=' + species_Z + \
                ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                ' average=' + str(soap_paras['average'])

    selected_atoms = boltzhist_cur_dualIter(
                    atoms_list=atoms,
                    isolated_atoms_energies={13: 0.2},
                    bolt_frac=0.8, 
                    bolt_max_num=5,
                    cur_num=3, 
                    kernel_exponent=4,
                    kT=0.3, 
                    energy_label='energy',
                    pressures=pressures,
                    descriptor=descriptor,
                    random_seed=42,                          
                )
    
    ref_energies = [0.03375957623344295, 
                    0.03375957623344296, 
                    0.01553737425165725]

    energies = [at.info['energy'] for at in selected_atoms]
    
    assert energies == ref_energies

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_sampling_bcur1s_job(test_dir, memory_jobstore):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='bcur1s',
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
                   isol_es={14: -0.84696938},
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

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_sampling_random_job(test_dir, memory_jobstore):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='random',
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

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_sampling_uniform_job(test_dir, memory_jobstore):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='uniform',
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

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)
