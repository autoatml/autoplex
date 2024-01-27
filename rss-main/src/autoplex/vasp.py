from jobflow import Response, job
from jobflow import Flow, Maker
from dataclasses import dataclass
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun
import os
import ase.io
from ase.constraints import voigt_6_to_full_3x3_stress
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.utils.path import strip_hostname
from pathlib import Path
import traceback
from autoplex.utilities import Species
from custodian.vasp.handlers import (
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    NonConvergingErrorHandler,
    PotimErrorHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
)


@job
def VASP_static(structures: list[Structure] | None = None, 
                isolated_atom: bool = False,
                isolated_species: list[str] | None = None,
                e0_spin: bool = True,
                dimer: bool = False,
                dimer_species: list[str] | None = None,
                dimer_range: list[float] = [1.4, 3.0],
                dimer_num: int = 9):

    """
    INCAR parameter
    """

    custom_set={"ENCUT": 600,
                "EDIFF": 1E-6,
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "PREC": 'Accurate',
                "ADDGRID": None,
                "ISYM": 0,
                "KSPACING": 0.2,
                "NCORE": 4,
                "LWAVE": 'False',
                "LCHARG": 'False',
                "ALGO": 'Fast',
                "ENAUG": None,
                "GGA": None,
                "ISPIN": None,
                "LAECHG": None,
                "LASPH": None,
                "LELF": None,
                "LMIXTAU": None,
                "LORBIT": None,
                "LREAL": None,
                "LVTOT": None,
                "NSW": None,
                "SYMPREC": None,
                "NELM": None,
                }
        
    job_list = []

    dirs = {'dirs_of_vasp': [], 'config_type': []}

    custom_handlers = (VaspErrorHandler(),
                       MeshSymmetryErrorHandler(),
                       UnconvergedErrorHandler(),
                       NonConvergingErrorHandler(),
                       PotimErrorHandler(),
                       FrozenJobErrorHandler(),
                       StdErrHandler(),
                       LargeSigmaHandler(),
                       IncorrectSmearingHandler(),
                       )

    st_m = StaticMaker(
        input_set_generator = StaticSetGenerator(
        user_incar_settings = custom_set),
        run_vasp_kwargs = {"handlers": custom_handlers},
    )

    if structures is not None:
        for struct in structures:
            static_job = st_m.make(structure = struct)
            dirs['dirs_of_vasp'].append(static_job.output.dir_name)
            dirs['config_type'].append('bulk')
            job_list.append(static_job)


    if isolated_atom:
        try:
            if isolated_species is not None:
                
                syms = isolated_species

            elif (isolated_species is None) and (structures is not None):
                
                # Get the species from the database        
                atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                syms = Species(atoms).get_species()
        
            for sym in syms:
                lattice = Lattice.orthorhombic(20.0, 20.5, 21.0)
                isolated_atom_struct = Structure(lattice,[sym], [[0.0, 0.0, 0.0]])
                static_job = st_m.make(structure = isolated_atom_struct)
                static_job = update_user_incar_settings(static_job, {"KSPACING": 2.0})

                if e0_spin:
                    static_job = update_user_incar_settings(static_job, {"ISPIN": 2})
                
                dirs['dirs_of_vasp'].append(static_job.output.dir_name)
                dirs['config_type'].append('isolated_atom')
                job_list.append(static_job)
 
        except: 
            raise ValueError('[log] Unknown species of isolated atoms!') 
         
    if dimer:
        try:
            if dimer_species is not None:        
                dimer_syms = dimer_species
            elif (dimer_species is None) and (structures is not None):
                # Get the species from the database        
                atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                dimer_syms = Species(atoms).get_species()
            pairs_list = Species(atoms).find_element_pairs(dimer_syms)
            for pair in pairs_list:
                for dimer_i in range(dimer_num):
                    dimer_distance = dimer_range[0] + (dimer_range[1] - dimer_range[0]) * \
                                     float(dimer_i) / float(dimer_num - 1 + 0.000000000001)
                    
                    lattice = Lattice.orthorhombic(15.0, 15.5, 16.0)
                    dimer_struct = Structure(lattice,
                                            [pair[0], pair[1]], 
                                            [[0.0, 0.0, 0.0], 
                                             [dimer_distance, 0.0, 0.0]],
                                            coords_are_cartesian=True)
            
                    static_job = st_m.make(structure = dimer_struct)
                    static_job = update_user_incar_settings(static_job, {"KSPACING": 2.0})

                    if e0_spin:
                        static_job = update_user_incar_settings(static_job, {"ISPIN": 2})

                    dirs['dirs_of_vasp'].append(static_job.output.dir_name)
                    dirs['config_type'].append('dimer')
                    job_list.append(static_job)
                    
        except:
            raise ValueError('[log] Unknown atom types in dimers!') 
        
    return Response(replace=Flow(job_list, output=dirs))
    

@dataclass
class VASP_collect_data(Maker):
    
    """

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    vasp_ref_file: str 
        The file to strore the training datasets labeled by VASP

    """
        
    name: str = "collect_vasp_outputs"
    vasp_ref_file: str = 'vasp_ref.extxyz' 
    gap_rss_group: str = 'RSS'
    
    @job
    def make(self, vasp_dirs: dict):

        def safe_strip_hostname(value):
            try:
                return strip_hostname(value)
            except Exception as e:
                print(f"Error processing '{value}': {e}")
                return None

        dirs = [safe_strip_hostname(value) for value in vasp_dirs['dirs_of_vasp']]
        config_types = vasp_dirs['config_type']

        print('[log] Attempting collecting VASP...', flush=True)

        if dirs == None:
            raise ValueError('[log] dft_dir must be specified if collect_vasp is True')
        
        failed_count = 0
        atoms = []
        isol_es = {}

        for i, val in enumerate(dirs):

            if os.path.exists(os.path.join(val, 'vasprun.xml.gz')): 
                
                try:
                    converged = self._check_convergence_vasp(os.path.join(val, 'vasprun.xml.gz'))

                    if converged:
                        at = ase.io.read(os.path.join(val, 'vasprun.xml.gz'), index=':')
                        for at_i in at:
                            virial_list = -voigt_6_to_full_3x3_stress(at_i.get_stress()) * at_i.get_volume()
                            at_i.info['REF_virial'] = ' '.join(map(str, virial_list.flatten()))
                            del at_i.calc.results['stress']
                            at_i.arrays['REF_forces'] = at_i.calc.results['forces']
                            del at_i.calc.results['forces']
                            at_i.info['REF_energy'] = at_i.calc.results['free_energy']
                            del at_i.calc.results['energy']
                            del at_i.calc.results['free_energy']
                            atoms.append(at_i)
                            at_i.info['config_type'] = config_types[i]
                            if at_i.info['config_type'] != 'dimer' and at_i.info['config_type'] != 'isolated_atom':
                                at_i.pbc=True
                                at_i.info['gap_rss_group']= self.gap_rss_group
                            else:
                                at_i.info['gap_rss_nonperiodic'] = 'T'

                            if at_i.info['config_type'] == 'isolated_atom':
                                at_ids = at_i.get_atomic_numbers()
                                # array_key = at_ids.tostring()
                                isol_es[int(at_ids[0])] = at_i.info['REF_energy']
                
                except:
                    print('[log] Failed to collect number', i)
                    failed_count += 1
                    traceback.print_exc()
        
        print('[log] Total %d structures from VASP are exactly collected.' % len(atoms))
        
        ase.io.write(self.vasp_ref_file, 
                     atoms, 
                     format='extxyz',
                     parallel=False)

        # structures = [AseAtomsAdaptor().get_structure(atom) for atom in atoms]
    
        # It's a pity that we are not allowed to output a Atoms object directly. 

        dir_path = Path.cwd()

        vasp_ref_dir = os.path.join(dir_path, self.vasp_ref_file)
 
        return {'vasp_ref_dir':vasp_ref_dir, 'isol_es':isol_es}


    def _check_convergence_vasp(self, file):

        """
        Check if VASP calculation has converged.
        True if a run is converged both ionically and electronically.

        """
    
        vasprun = Vasprun(file)
        converged_e = vasprun.converged_electronic
        converged_i = vasprun.converged_ionic

        if converged_e and converged_i: 
            return True
        
        else: 
            return False
        
        