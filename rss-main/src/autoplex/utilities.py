import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from quippy import descriptors
from ase.atoms import Atoms
from collections.abc import Iterable
import ase
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import xml.etree.ElementTree as ET


class Species:

    def __init__(self, atoms):
        
        self.atoms = atoms

    def get_species(self):

        sepcies_list = []

        for at in self.atoms:
            sym_all = at.get_chemical_symbols()
            syms = list(set(sym_all))
            for sym in syms:
                if sym in sepcies_list: 
                    continue
                else: 
                    sepcies_list.append(sym)

        return sepcies_list


    def find_element_pairs(self, symb_list = None):

        if symb_list is None:
            species_list = self.get_species()

        else:
            species_list = symb_list

        pairs = []  

        for i in range(len(species_list)):
            for j in range(i, len(species_list)):
                pair = (species_list[i], species_list[j])  
                pairs.append(pair)  

        return pairs
    

    def get_number_of_species(self):

        return int(len(self.get_species()))
    

    def get_species_Z(self):
    
        atom_numbers = []
        for atom_type in self.get_species():
            atom = Atoms(atom_type, [(0, 0, 0)]) 
            atom_numbers.append(int(atom.get_atomic_numbers()[0]))
        
        species_Z = '{'
        for i in range(len(atom_numbers)-1):
            species_Z += (str(atom_numbers[i]) + ' ')
        species_Z += str(atom_numbers[-1]) + '}'
        
        return species_Z
    


def flatten(o, recursive=False):
    '''Flatten an iterable fully, but excluding Atoms objects'''
    l = []

    if recursive:
        for ct, el in enumerate(o):
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes, ase.atoms.Atoms, ase.Atoms)):
                l += flatten(el, recursive=True)
            else:
                l += [el]
        return l

    else:
        return [item for sublist in o for item in sublist]
    

    
def gcm3_to_Vm(gcm3, mr, natoms=1):
    return 1 / (natoms * (gcm3 / mr) * 6.022e23 / (1e8)**3)



def get_atomic_numbers(species):
    
    atom_numbers = []
    for atom_type in species:
        atom = Atoms(atom_type, [(0, 0, 0)]) 
        atom_numbers.append(int(atom.get_atomic_numbers()[0]))

    return atom_numbers


def split_dataset(atoms, split_ratio):

    atom_bulk = []
    atom_isolated_and_dimer = []
    for at in atoms:
        if at.info['config_type'] != 'dimer' and at.info['config_type'] != 'isolated_atom':
            atom_bulk.append(at)
        else:
            atom_isolated_and_dimer.append(at)
    
    if len(atoms) != len(atom_bulk):
        atoms = atom_bulk

    average_energies = np.array([atom.info['REF_energy']/len(atom) for atom in atoms])
    # sort by energy
    sorted_indices = np.argsort(average_energies)
    atoms = [atoms[i] for i in sorted_indices]
    average_energies = average_energies[sorted_indices]

    stratified_average_energies = pd.qcut(average_energies, q=2, labels=False)
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    for train_index, test_index in split.split(atoms, stratified_average_energies):
        train_structures = [atoms[i] for i in train_index]
        test_structures = [atoms[i] for i in test_index]

    if atom_isolated_and_dimer:
        train_structures = atom_isolated_and_dimer + train_structures

    return train_structures, test_structures


def data_distillation(vasp_ref_dir, f_max):
    
    atoms = ase.io.read(vasp_ref_dir, index=':')

    atoms_distilled = []
    for at in atoms:

        forces = np.abs(at.arrays['REF_forces'])
        f_component_max = np.max(forces)

        if f_component_max  < f_max:
            atoms_distilled.append(at)

    print(f'After distillation, there are still {len(atoms_distilled)} data points remaining.')

    return atoms_distilled


def rms_dict(x_ref, x_pred):

    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))
    return {'rmse': average, 'std': std_}


def energy_remain(in_file):
    """ Plots the distribution of energy per atom on the output vs the input"""
    # read files
    in_atoms = ase.io.read(in_file, ':')
    #in_atoms = []
    # for at in in_atoms1:
    #     if at.info['config_type'] != 'dimer':
    #         in_atoms.append(at)
    ener_in = [at.info['REF_energy'] / len(at.get_chemical_symbols()) for at in in_atoms]
    ener_out = [at.info['energy'] / len(at.get_chemical_symbols()) for at in in_atoms]
    _rms = rms_dict(ener_in, ener_out)
    # print("RMSE:{:14.8f}".format(_rms['rmse']))
    return _rms['rmse']


def extract_gap_label(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    label = root.tag
    return label

