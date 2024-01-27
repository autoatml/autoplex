from jobflow import job
from jobflow import Maker
from dataclasses import dataclass, field
from multiprocessing import Pool
from pymatgen.io.ase import AseAtomsAdaptor
import os
import ase.io
from subprocess import run
from ase.atoms import Atoms
import numpy as np
from ase.io.castep import write_castep_cell
from pathlib import Path


def _parallel_process(i, bt_file, tag, remove_tmp_files):

    tmp_file_name = "tmp." + str(i) + '.' + tag + '.cell'

    run("buildcell",
        stdin=open(bt_file, "r"),
        stdout=open(tmp_file_name, "w"),
        shell=True).check_returncode()
    
    atom = ase.io.read(tmp_file_name, parallel=False)
    atom.info["unique_starting_index"] = i

    if "castep_labels" in atom.arrays:
        del atom.arrays["castep_labels"]

    if "initial_magmoms" in atom.arrays:
        del atom.arrays["initial_magmoms"]

    if remove_tmp_files:
        os.remove(tmp_file_name)

    return atom


@dataclass
class random_structure(Maker):
    """
    Maker to create random structures by 'buildcell'
    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    struct_number : int
        Epected number of generated randomized unit cells.
    tag : (str) 
        name of the seed file for builcell.
    input_file_name: str 
        input file of buildcell to set parameters
    output_file_name : str
        A file to store all generated structures. 
    remove_tmp_files : bool 
        Remove all temporary files raised by buildcell to save memory

    """
    name: str = "Build_random_cells"
    struct_number: int = 20
    tag: str = 'Si'
    output_file_name: str = 'random_structs.extxyz'
    remove_tmp_files: bool = True
    buildcell_options: list = field(default_factory=lambda: [
                                    'TARGVOL=15-20',
                                    'SPECIES=Si',
                                    'NATOM={6,8,10,12,16,18,20,22,24}',
                                    'NFORM=1',
                                    'SYMMOPS=1-8',
                                    'SLACK=0.25',
                                    'OVERLAP=0.1',
                                    'COMPACT',
                                    'MINSEP=1.5',
                                ])
        
    @job
    def make(self):
        self._cell_seed(self.buildcell_options, self.tag)
        bt_file = '{}.cell'.format(self.tag)

        # 使用 multiprocessing.Pool 来并行处理
        with Pool() as pool:
            args = [(i, bt_file, self.tag, self.remove_tmp_files) for i in range(self.struct_number)]
            atoms_group = pool.starmap(_parallel_process, args)

        output_file = open(self.output_file_name, 'w')
        ase.io.write(output_file, atoms_group, parallel=False, format="extxyz")

        rss_atoms = [AseAtomsAdaptor().get_structure(at) for at in atoms_group]

        dir_path = Path.cwd()
        path = os.path.join(dir_path, self.output_file_name)

        return path
        

    def _cell_seed(self,
                  buildcell_options,
                  tag,
                  atoms=[], 
                  fragment=False, 
                  fragment_atoms=None):
        
        '''
        Prepares random cells in self.directory
        Arguments:
        buildcell options :: (list of str) e.g. ['VARVOL=20']
        atoms :: (optional - ignored if species in buildcell options) 
                    list of tuples for atom and number to include,
                    e.g. [(Mo1,1), (S1,3)]
                    
        will write the cells to self.init_atoms, a dictionary of ase.Atoms objects
        '''

        bc_file = '{}.cell'.format(tag)
        
        if 'SPECIES' not in ' '.join(buildcell_options):

            if fragment:
                at = fragment_atoms

            else:
                at = Atoms(symbols=''.join([i[0][:-1] for i in atoms]),
                        cell=np.identity(3)*2, pbc=True,
                        positions=np.zeros((len(atoms), 3))
                        )

            write_castep_cell(bc_file, at, positions_frac=True)
            
            with open(bc_file, 'r') as f:
                contents = f.readlines()
            
            flag = 0; ct = 0
            if not fragment:
                for i, val in enumerate(contents):
                    if '%BLOCK POSITIONS_FRAC' in val:
                        flag=1
                    elif '%ENDBLOCK POSITIONS_FRAC' in val:
                        flag=0
                    elif flag==1:
                        contents[i] = val.strip('\n') + ' # {} % NUM={}\n'.format(atoms[ct][0], atoms[ct][1])
                        ct+=1
            else:
                for i, val in enumerate(contents):
                    if '%BLOCK POSITIONS_FRAC' in val:
                        flag=1
                    elif '%ENDBLOCK POSITIONS_FRAC' in val:
                        flag=0
                    elif flag==1:
                        contents[i] = val.strip('\n') + ' # {} % NUM=1\n'.format(fragment_atoms.arrays['fragment_id'][ct])
                        ct+=1
                
        else:
            with open(bc_file, 'w') as f:
                contents=['']
                
        contents.extend(['#' + i + '\n' for i in buildcell_options])
        
        with open(bc_file, 'w') as f:
            f.writelines(contents)
    