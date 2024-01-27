import ase.io
import os
from jobflow import job, Maker, Response
from dataclasses import dataclass
import os
import ase.io
from pathlib import Path
from autoplex.utilities import split_dataset, data_distillation
import shutil
from autoplex.regularization import set_sigma
from autoplex.mlip_models import gap_fitting, ace_fitting


@dataclass
class data_preprocessing(Maker):

    """

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    vasp_ref_file: str 
        The file to strore the training datasets labeled by VASP

    """
        
    name: str = "data_preprocessing_for_fitting"
    split_ratio: float = 0.5
    regularization: bool = False
    distillation: bool = False
    f_max: float = 40.0

    @job
    def make(self, 
             vasp_ref_dir: str, 
             pre_database_dir: str,
             ):

        # reject strucutres with large force components
        if self.distillation:
            atoms = data_distillation(vasp_ref_dir, self.f_max)
        else:
            atoms = ase.io.read(vasp_ref_dir, index=':')

        # split dataset into training and testing datasets with a ratio of 9:1
        train_structures, test_structures = split_dataset(atoms, self.split_ratio)

        # Merging database
        if pre_database_dir and os.path.exists(pre_database_dir):
            files_to_copy = ['train.extxyz', 'test.extxyz']
            current_working_directory = os.getcwd()

            for file_name in files_to_copy:
                source_file_path = os.path.join(pre_database_dir, file_name)
                destination_file_path = os.path.join(current_working_directory, file_name)
                shutil.copy(source_file_path, destination_file_path)
                print(f"File {file_name} has been copied to {destination_file_path}")

        ase.io.write('train.extxyz', train_structures, format='extxyz', append='True')
        ase.io.write('test.extxyz', test_structures, format='extxyz', append='True')

        if self.regularization:
            atoms = ase.io.read('train.extxyz', index=':')
            atom_with_sigma = set_sigma(atoms, etup = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)])
            ase.io.write('train_with_sigma.extxyz',atom_with_sigma,format='extxyz')

        database_path = Path.cwd()

        return database_path


@dataclass
class mlip_fit(Maker):

    """
    Maker to fitting potential
    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mlip_type: str 
        Choose one specific MLIP type: 
        'GAP' | 'SNAP' | 'ACE' | 'Nequip' | 'Allegro' | 'MACE'
    HPO: bool
        call hyperparameter optimization (HPO) or not

    """
    name: str = 'MLIP_FIT'
    mlip_type: str = None
    HPO: bool = False

    @job
    def make(self, 
             database_dir: str, 
             gap_para={'two_body':True, 'three_body':True},
             ace_para={'energy_name':"REF_energy",
                       'force_name':"REF_forces",
                       'virial_name':"REF_virials",
                       'order':3, 
                       'totaldegree':6, 
                       'cutoff':2.0, 
                       'solver':'BLR',},
             isol_es=None,
             num_of_threads=128):

        database_path = database_dir
        mlip_path = Path.cwd()
        if os.path.join(database_path, 'train_with_sigma.extxyz'):
            shutil.copy(os.path.join(database_path, 'train_with_sigma.extxyz'), 
                        os.path.join(mlip_path, 'train_with_sigma.extxyz'))
        shutil.copy(os.path.join(database_path, 'test.extxyz'), 
            os.path.join(mlip_path, 'test.extxyz'))
        shutil.copy(os.path.join(database_path, 'train.extxyz'), 
            os.path.join(mlip_path, 'train.extxyz'))

        if self.mlip_type is None:   
            raise ValueError("MLIP type is not defined! The current version supports the fitting of GAP, SNAP, ACE, Nequip, Allegro, or MACE.")
        
        if self.mlip_type == 'GAP':
            train_error, test_error = gap_fitting(dir=database_dir, 
                                                  two_body=gap_para['two_body'], 
                                                  three_body=gap_para['three_body'], 
                                                  soap=True)
            
        if self.mlip_type == 'ACE':
            train_error, test_error = ace_fitting(dir=database_dir, 
                                                  energy_name=ace_para['energy_name'], 
                                                  force_name=ace_para['force_name'], 
                                                  virial_name=ace_para['virial_name'],
                                                  order=ace_para['order'],
                                                  totaldegree=ace_para['totaldegree'],
                                                  cutoff=ace_para['cutoff'],
                                                  solver=ace_para['solver'],
                                                  isol_es=isol_es,
                                                  num_of_threads=num_of_threads)

        if test_error < 0.01:
            return {'mlip_path':mlip_path, 
                    'train_error':train_error, 
                    'test_error':test_error,
                    'convergence': True}
        else:
            return {'mlip_path':mlip_path, 
                    'train_error':train_error, 
                    'test_error':test_error,
                    'convergence': False}

