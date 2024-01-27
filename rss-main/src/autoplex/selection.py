from jobflow import job
from jobflow import Maker
from dataclasses import dataclass, field
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase.io
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from quippy import descriptors
import traceback
from autoplex.utilities import Species
from autoplex.cur import cur_select
from autoplex.boltzhist import boltzhist_CUR


@dataclass
class boltz_cur(Maker):
    """
    Maker to create random structures by 'buildcell'
    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    SOAP descriptor:
        {l_max, n_max, atom_sigma, cutoff, n_species, species_Z, cutoff_transition_width, average}
    T: float
        temperature / eV for boltzmann weighting
    P: float 
        list of pressures at which Atoms have been optimisied
          - required for boltzhist_CUR
    selection_method: str 
        CUR - pure CUR
        boltzhist_CUR - boltzmann flat histogram in enthalpy, then CUR
    num_of_cur: int
        pure cur selection number
    num_of_bcur : int
        flat boltzmann selection number
    zeta: float
        exponent for dot-product SOAP kernel
    """
    
    name: str = "sampling"

    soap_paras: dict = field(default_factory=lambda: {
                                                'l_max': 8,
                                                'n_max': 8,
                                                'atom_sigma': 0.75,
                                                'cutoff': 5.5,
                                                'cutoff_transition_width': 1.0,
                                                'zeta': 4.0,
                                                'average': True,
                                                'species': True,
                                            })

    kT: float = 0.3
    selection_method: str = None
    num_of_cur: int = 5
    frac_of_bcur: float = 0.1
    bolt_max_num: int = 3000
    kernel_exp: float = 4.0
    energy_label: str = 'energy'


    @job
    def make(self, dir=None, structure: list[Structure]=None, traj_info: list=None, isol_es=None):
        
        if dir is not None:
            atoms = ase.io.read(dir, index=':')

        elif structure is not None:
            atoms = [AseAtomsAdaptor().get_atoms(at) for at in structure]
            
        else:  
            atoms = []
            pressures = []
            for traj in traj_info:
                if traj is not None:
                    print('traj:', traj)
                    at = ase.io.read(traj['traj_path'],index=':')
                    atoms.extend(at)
                    pressure = [traj['pressure']] * len(at)
                    pressures.extend(pressure)

        n_species = Species(atoms).get_number_of_species()
        species_Z = Species(atoms).get_species_Z()

        descriptor = 'soap l_max=' + str(self.soap_paras['l_max']) + \
                     ' n_max=' + str(self.soap_paras['n_max']) + \
                     ' atom_sigma=' + str(self.soap_paras['atom_sigma']) + \
                     ' cutoff=' + str(self.soap_paras['cutoff']) + \
                     ' n_species=' + str(n_species) + \
                     ' species_Z=' + species_Z + \
                     ' cutoff_transition_width=' + str(self.soap_paras['cutoff_transition_width']) + \
                     ' average =' + str(self.soap_paras['average'])

        if (self.selection_method is None) or (self.selection_method == 'cur'):
        
            selected_atoms = cur_select(atoms=atoms, 
                                        selected_descriptor=descriptor,
                                        kernel_exp=self.kernel_exp, 
                                        select_nums=self.num_of_cur, 
                                        stochastic=True)

            ase.io.write('sample_cur.extxyz', selected_atoms, parallel=False)


        elif self.selection_method == 'boltzhist_CUR':

            isol_es = {int(k): v for k, v in isol_es.items()}

            selected_atoms = boltzhist_CUR(atoms=atoms,
                                           isol_es=isol_es,
                                           bolt_frac=self.frac_of_bcur, 
                                           bolt_max_num=self.bolt_max_num,
                                           cur_num=self.num_of_cur, 
                                           kernel_exp=self.kernel_exp, 
                                           kT=self.kT, 
                                           energy_label=self.energy_label,
                                           P=pressures,
                                           descriptor=descriptor
                                          )


        
        selected_atoms = [AseAtomsAdaptor().get_structure(at) for at in selected_atoms]
            
        return selected_atoms


@job
def random(structure: list[Structure], num_of_rand: int):

    try: 
        selection = np.random.choice(0, len(structure), num_of_rand)
        selected_atoms = [at for i, at in enumerate(structure) if i in selection]
   
    except:
        print('[log] The number of selected structures must be less than the total!')
        traceback.print_exc()

    return selected_atoms


@job
def all(structure: list[Structure], num_of_rand: int):

    return structure