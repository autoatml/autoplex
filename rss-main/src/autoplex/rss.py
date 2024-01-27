import ase.io
import os
import numpy as np
import quippy.potential
from ase.optimize.precon import PreconLBFGS, Exp
from ase.constraints import UnitCellFilter
from ase.units import GPa
from ase.constraints import FixConstraint, slice2enlist
from ase.geometry import find_mic
from jobflow import job
from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
from multiprocessing import Pool
from autoplex.utilities import extract_gap_label
from ase.calculators.lammpslib import LAMMPSlib


class myPotential(quippy.potential.Potential):
    def calculate(self, *args, **kwargs):
        res = super().calculate(*args, **kwargs)
        if 'atoms' in kwargs.keys():
            atoms = kwargs['atoms']
        else:
            atoms = args[0]
        if 'forces' in self.results.keys():
            atoms.arrays['forces'] = self.results['forces'].copy()
        if 'energy' in self.results.keys():
            atoms.info['energy'] = self.results['energy'].copy()
        if 'virial' in self.extra_results['config'].keys():
            atoms.info['virial'] = self.extra_results['config']['virial'].copy()
        # if 'stress' in self.results.keys():
        #     atoms.info['stress'] = self.results['stress'].copy()
        return res


class HookeanRepulsion(FixConstraint):
    """Constrain atoms softly to minimum separation. Useful to avoid early versions of potentials
    causing crashes due to atoms being too close together."""

    def __init__(self, a1, a2, k, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
        restorative force when the distance between them exceeds rt. Can
        also be used to tether an atom to a fixed point in space or to a
        distance above a plane.

        a1 : int
           Index of atom 1
        a2 : one of three options
           1) index of atom 2
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
           4) 'cell' :: selects the unit cell to constrain
        k : float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length. Units of eV A^-2.
        rt : float
           The threshold length above which there is no force. The
           length is 1) between two atoms, 2) between atom and point.
           This argument is not supplied in case 3. Units of A.

        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).

        References:

           Andrew A. Peterson,  Topics in Catalysis volume 57, pages40–53 (2014)
           https://link.springer.com/article/10.1007%2Fs11244-013-0161-8
        """

        if isinstance(a2, int):
            self._type = 'two atoms'
            self.indices = [a1, a2]
        elif len(a2) == 3:
            self._type = 'point'
            self.index = a1
            self.origin = np.array(a2)
        elif len(a2) == 4:
            self._type = 'plane'
            self.index = a1
            self.plane = a2
        
        else:
            raise RuntimeError('Unknown type for a2')
        self.threshold = rt
        self.spring = k
        self.used = False

    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        dct = {'name': 'Hookean'}
        dct['kwargs'] = {'rt': self.threshold,
                         'k': self.spring}
        if self._type == 'two atoms':
            dct['kwargs']['a1'] = self.indices[0]
            dct['kwargs']['a2'] = self.indices[1]
        elif self._type == 'point':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.origin
        elif self._type == 'plane':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.plane
        else:
            raise NotImplementedError('Bad type: %s' % self._type)
        return dct

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d < 0:
                return
            magnitude = self.spring * d
            direction = - np.array((A, B, C)) / np.linalg.norm((A, B, C))
            forces[self.index] += direction * magnitude
            return
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin

        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)

        if bondlength < self.threshold:
            # print('Hookean adjusting forces, bondlength: ', bondlength, ' < ', self.threshold)
            self.used = True
            magnitude = self.spring * (self.threshold - bondlength)
            direction = displace / np.linalg.norm(displace)
            if self._type == 'two atoms':
                forces[self.indices[0]] -= direction * magnitude
                forces[self.indices[1]] += direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d > 0:
                return 0.5 * self.spring * d**2
            else:
                return 0.
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength < self.threshold:
            return 0.5 * self.spring * (bondlength - self.threshold)**2
        else:
            return 0.

    def get_indices(self):
        if self._type == 'two atoms':
            return self.indices
        elif self._type == 'point':
            return self.index
        elif self._type == 'plane':
            return self.index

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        if self._type == 'two atoms':
            newa = [-1, -1]  # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                for i, a in enumerate(self.indices):
                    if old == a:
                        newa[i] = new
            if newa[0] == -1 or newa[1] == -1:
                raise IndexError('Constraint not part of slice')
            self.indices = newa
        elif (self._type == 'point') or (self._type == 'plane'):
            newa = -1   # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                if old == self.index:
                    newa = new
                    break
            if newa == -1:
                raise IndexError('Constraint not part of slice')
            self.index = newa

    def __repr__(self):
        if self._type == 'two atoms':
            return 'Hookean(%d, %d)' % tuple(self.indices)
        elif self._type == 'point':
            return 'Hookean(%d) to cartesian' % self.index
        else:
            return 'Hookean(%d) to plane' % self.index


def process_rss(atom, 
                mlip_path, 
                output_file_name, 
                mlip_type, 
                scalar_pressure_method, 
                scalar_exp_pressure, 
                scalar_pressure_exponential_width, 
                scalar_pressure_low, 
                scalar_pressure_high, 
                max_steps, 
                force_tol, 
                stress_tol, 
                Hookean_repul, 
                hookean_paras, 
                write_traj):

    if mlip_type == 'GAP':
        gap_label = os.path.join(mlip_path, 'gap_file.xml')
        gap_control = 'Potential xml_label=' + extract_gap_label(gap_label)
        pot = myPotential(args_str=gap_control, param_filename=gap_label)

    elif mlip_type == 'ACE':
        ace_label = os.path.join(mlip_path,'acemodel.yace')
        ace_table = os.path.join(mlip_path,'acemodel_pairpot.table')
        cmds = ["pair_style     hybrid/overlay pace table spline 5000",
                f"pair_coeff     * * pace {ace_label} Si",
                f"pair_coeff     1 1 table {ace_table} Si_Si"]

        pot = LAMMPSlib(lmpcmds=cmds, log_file='test.log', keep_alive=True)

    unique_starting_index = atom.info['unique_starting_index']
    log_file_name = output_file_name + '_' + str(unique_starting_index) + '.log'
    log_file = open(log_file_name, 'w')

    if Hookean_repul:
        print('Hookean repulsion is used!')
        hks = []
        ans = atom.get_atomic_numbers()
        for i in range(len(ans)):
            for j in range(i+1, len(ans)):
                if (ans[i], ans[j]) in hookean_paras:
                    if hookean_paras[(ans[i], ans[j])][0] != 0 and hookean_paras[(ans[i], ans[j])][1] != 0:
                        hks.append(HookeanRepulsion(i, j, *hookean_paras[(ans[i], ans[j])]))
        atom.set_constraint(hks)

    atom.calc = pot
    if scalar_pressure_method == 'exp':
        scalar_pressure_tmp = scalar_exp_pressure * GPa
        if scalar_pressure_exponential_width > 0.0:
            scalar_pressure_tmp *= np.random.exponential(scalar_pressure_exponential_width)
    elif scalar_pressure_method == 'uniform':
        scalar_pressure_tmp = np.random.uniform(low=scalar_pressure_low, high=scalar_pressure_high) * GPa
    atom.info["RSS_applied_pressure"] = scalar_pressure_tmp / GPa
    atom = UnitCellFilter(atom, scalar_pressure=scalar_pressure_tmp)

    try:
        optimizer = PreconLBFGS(atom,
                                precon=Exp(3),
                                use_armijo=True,
                                logfile=log_file,
                                master=True)
        traj = []

        def build_traj():
            atom_copy = atom.copy()
            atom_copy.info['energy'] = atom.get_potential_energy().copy()
            atom_copy.info['forces'] = atom.get_forces().copy()
            traj.append(atom_copy)

        optimizer.attach(build_traj)
        optimizer.run(fmax=force_tol, smax=stress_tol, steps=max_steps)

        if optimizer.converged():
            minim_stat = "converged"
        else:
            minim_stat = "unconverged"

        for (traj_at_i, traj_at) in enumerate(traj):
            traj_at.info["RSS_minim_iter"] = traj_at_i
            traj_at.info["config_type"] = "traj"
            traj_at.info["minim_stat"] = minim_stat
        if write_traj:
            traj_file_name = output_file_name + '_traj_' + str(unique_starting_index) + '.extxyz'
            ase.io.write(traj_file_name, traj, parallel=False)
        del traj[-1].info["minim_stat"]
        traj[-1].info["config_type"] = minim_stat + "_minimum"

        local_minima = traj[-1]

        if local_minima.info["config_type"] != "unconverged_minimum":
            dir_path = Path.cwd()
            traj_dir = os.path.join(dir_path, traj_file_name)
            return {'traj_path': traj_dir, 'pressure': local_minima.info['RSS_applied_pressure']}
        else:
            return None

    except RuntimeError:
        print("RuntimeError occurred during optimization! Return none!")
        return None


def minimize_structures(mlip_path,
                        index, 
                        input_structure, 
                        output_file_name, 
                        mlip_type,
                        scalar_pressure_method,
                        scalar_exp_pressure, 
                        scalar_pressure_exponential_width, 
                        scalar_pressure_low, 
                        scalar_pressure_high, 
                        max_steps=2000, 
                        force_tol=1.0e-3, 
                        stress_tol=1.0e-3, 
                        Hookean_repul=False, 
                        hookean_paras={(14, 14): (1000, 2.5)}, 
                        write_traj=True):
    
    atoms = [AseAtomsAdaptor().get_atoms(structure) for structure in input_structure]
    
    for i, atom in enumerate(atoms):
        atom.info['unique_starting_index'] = index + f'{i}'

    args = [(atom, mlip_path, output_file_name, mlip_type, scalar_pressure_method, scalar_exp_pressure, 
             scalar_pressure_exponential_width, scalar_pressure_low, scalar_pressure_high, 
             max_steps, force_tol, stress_tol, Hookean_repul, hookean_paras, write_traj) for atom in atoms]

    with Pool() as pool:
        results = pool.starmap(process_rss, args)

    output = [result for result in results]

    return output

