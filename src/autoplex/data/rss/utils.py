"""Utility functions for rss."""

import ast
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import ase.io
import matgl
import numpy as np
from ase import Atoms
from ase.constraints import (
    FixConstraint,
    FixSymmetry,
    UnitCellFilter,
    slice2enlist,
)
from ase.data import atomic_numbers, chemical_symbols
from ase.geometry import find_mic
from ase.optimize.precon import Exp, PreconLBFGS
from ase.units import GPa
from mace.calculators import MACECalculator
from matgl.ext.ase import M3GNetCalculator
from nequip.ase import NequIPCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.fitting.common.utils import (
    CustomPotential,
    extract_gap_label,
)


def extract_pairstyle(
    ace_label: str, ace_json: str, ace_table: str
) -> tuple[dict[str, int], list[str]]:
    """
    Extract the pair style and coefficients from ACE potential files for running LAMMPS.

    Parameters
    ----------
        ace_label: str
            Label for the ACE potential.
        ace_json: str
            Path to the JSON file of ACE potential.
        ace_table: str
            Path to the table file containing pairwise coefficients of ACE potential.
    """
    with open(ace_json) as file:
        data = json.load(file)

    E0 = data["IP"]["components"][2]["E0"]

    elements = list(E0.keys())

    sorted_elements = sorted(elements, key=lambda x: atomic_numbers[x])

    with open(ace_table) as file:
        lines = file.readlines()

        for line in lines:
            if line.strip().startswith("N "):
                n_value = line.strip().split()[1]
                break

    elements_str = " ".join(sorted_elements)

    cmds = [
        f"pair_style     hybrid/overlay pace table spline {n_value}",
        f"pair_coeff     * * pace {ace_label} {elements_str}",
    ]

    atom_types = {}

    for i in range(len(sorted_elements)):
        atom_types[sorted_elements[i]] = i + 1

        for j in range(i, len(sorted_elements)):
            pairs = sorted_elements[i] + "_" + sorted_elements[j]
            cmds.append(f"pair_coeff     {i+1} {j+1} table {ace_table} {pairs}")

    return atom_types, cmds


class HookeanRepulsion(FixConstraint):
    """Constrain atoms softly to a minimum separation.

    Intended to avoid early iterations of potentials
    causing crashes due to overlapping atoms.

    It is recommended to calibrate the spring constant for your system
    dependent on the potential and the atomic species used. It is not guaranteed
    that the constraint will be either soft enough (e.g. non-exploding in MD) or
    strong enough (to avoid overlaps) for all spring constants and distances.

    References
    ----------
    *    Title: ASE constraints package at  at ase/ase/constraints.py
    *    Author: Ask Hjorth Larsen
    *    Date 07/10/2024
    *    Code version: 3.23.0
    *    Availability: https://gitlab.com/ase/
    """

    def __init__(
        self,
        a1: int,
        a2: (
            int
            | tuple[float, float, float]
            | tuple[float, float, float, float]
            | Literal["cell"]
        ),
        k: float,
        rt: float | None = None,
    ) -> None:
        """Apply a Hookean restorative force repel two atoms that are close.

        Parameters
        ----------
        a1: int
           Atom 1 index
        a2: can be one of three options
           1) Atom 2 index
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
           4) 'cell' :: selects the unit cell to constrain
        k: float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length. Units are eV Å^-2.
        rt: float
           Threshold length above which no Hookean force is applied.
           This argument is not supplied in case 3. Units of Å.

        Notes
        -----
        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).

        References
        ----------
           Andrew A. Peterson,  Topics in Catalysis volume 57, pages40-53 (2014)
           https://link.springer.com/article/10.1007%2Fs11244-013-0161-8
        """
        if isinstance(a2, int):
            self._type = "two atoms"
            self.indices = [a1, a2]
        elif len(a2) == 3:
            self._type = "point"
            self.index = a1
            self.origin = np.array(a2)
        elif len(a2) == 4:
            self._type = "plane"
            self.index = a1
            self.plane = a2

        else:
            raise RuntimeError("Unknown type for a2")
        self.threshold = rt
        self.spring = k
        self.used = False

    def get_removed_dof(self, atoms):
        """Get number of removed degrees of freedom due to constraint."""
        return 0

    def todict(self):
        """Convert constraint to dictionary."""
        dct = {"name": "Hookean"}
        dct["kwargs"] = {"rt": self.threshold, "k": self.spring}
        if self._type == "two atoms":
            dct["kwargs"]["a1"] = self.indices[0]
            dct["kwargs"]["a2"] = self.indices[1]
        elif self._type == "point":
            dct["kwargs"]["a1"] = self.index
            dct["kwargs"]["a2"] = self.origin
        elif self._type == "plane":
            dct["kwargs"]["a1"] = self.index
            dct["kwargs"]["a2"] = self.plane
        else:
            raise NotImplementedError(f"Bad type: {self._type}")
        return dct

    def adjust_positions(self, atoms, newpositions):
        """Adjust positions to match the constraints.

        Do nothing for this constraint.
        """

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta to match the constraints.

        Do nothing for this constraint.
        """

    def adjust_forces(self, atoms, forces):
        """Adjust forces on the atoms to match the constraints."""
        positions = atoms.positions
        if self._type == "plane":
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = (A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
            if d < 0:
                return
            magnitude = self.spring * d
            direction = -np.array((A, B, C)) / np.linalg.norm((A, B, C))
            forces[self.index] += direction * magnitude
            return
        if self._type == "two atoms":
            p1, p2 = positions[self.indices]
        elif self._type == "point":
            p1 = positions[self.index]
            p2 = self.origin

        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)

        if bondlength < self.threshold:
            print(
                "Hookean adjusting forces, bondlength: ",
                bondlength,
                " < ",
                self.threshold,
            )
            self.used = True
            magnitude = self.spring * (self.threshold - bondlength)
            direction = displace / np.linalg.norm(displace)
            if self._type == "two atoms":
                forces[self.indices[0]] -= direction * magnitude
                forces[self.indices[1]] += direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, atoms):
        """Return the difference to the potential energy due to an active constraint.

        (the quantity returned is to be added to the potential energy).
        """
        positions = atoms.positions
        if self._type == "plane":
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = (A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
            if d > 0:
                return 0.5 * self.spring * d**2
            return 0.0
        if self._type == "two atoms":
            p1, p2 = positions[self.indices]
        elif self._type == "point":
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength < self.threshold:
            return 0.5 * self.spring * (bondlength - self.threshold) ** 2
        return 0.0

    def get_indices(self):
        """Get the indices."""
        if self._type == "two atoms":
            return self.indices
        if self._type == "point":
            return self.index
        if self._type == "plane":
            return self.index
        return None

    def index_shuffle(self, atoms, ind):
        """Change the indices."""
        if self._type == "two atoms":
            newa = [-1, -1]  # Error condition
            for new, old in slice2enlist(ind, len(atoms)):
                for i, a in enumerate(self.indices):
                    if old == a:
                        newa[i] = new
            if newa[0] == -1 or newa[1] == -1:
                raise IndexError("Constraint not part of slice")
            self.indices = newa
        elif (self._type == "point") or (self._type == "plane"):
            new_a = -1  # Error condition
            for new, old in slice2enlist(ind, len(atoms)):
                if old == self.index:
                    new_a = new
                    break
            if new_a == -1:
                raise IndexError("Constraint not part of slice")
            self.index = new_a

    def __repr__(self):
        """Return a representation of the constraint."""
        if self._type == "two atoms":
            return f"Hookean({self.indices[0]}, {self.indices[1]})"
        if self._type == "point":
            return f"Hookean({self.index}) to cartesian"
        return f"Hookean({self.index}) to plane"


def process_rss(
    atom: Atoms,
    mlip_type: str,
    mlip_path: str | list[str],
    output_file_name: str = "RSS_relax_results",
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 1000,
    force_tol: float = 0.01,
    stress_tol: float = 0.01,
    hookean_repul: bool = False,
    hookean_paras: dict | None = None,
    write_traj: bool = True,
    device: str = "cpu",
    isolated_atom_energies: dict[int, float] | None = None,
    config_type: str = "traj",
    keep_symmetry: bool = True,
) -> str | None:
    """Run RSS on a single thread using MLIPs.

    Parameters
    ----------
    atom: Atoms
        ASE Atoms object representing the atomic configuration.
    mlip_type: str
        Choose one specific MLIP type:
        'GAP' | 'J-ACE' | 'NequIP' | 'M3GNet' | 'MACE'.
    mlip_path: str | list[str]
        Path to the MLIP model or List of Path to the MLIP model.
    output_file_name: str
        Prefix for the trajectory/log file name. The actual output file name
        may be composed of this prefix, an index, and file types.
    scalar_pressure_method: str
        Method for adding external pressures. Default is 'exp'.
    scalar_exp_pressure: float
        Scalar exponential pressure. Default is 100.
    scalar_pressure_exponential_width: float
        Width for scalar pressure exponential. Default is 0.2.
    scalar_pressure_low: float
        Low limit for scalar pressure. Default is 0.
    scalar_pressure_high: float
        High limit for scalar pressure. Default is 50.
    max_steps: int
        Maximum number of steps for relaxation. Default is 1000.
    force_tol: float
        Force residual tolerance for relaxation. Default is 0.01.
    stress_tol: float
        Stress residual tolerance for relaxation. Default is 0.01.
    hookean_repul: bool
        If true, apply Hookean repulsion. Default is False.
    hookean_paras: dict[tuple[int, int], tuple[float, float]]
        Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
    write_traj: bool
        If true, write trajectory of RSS. Default is True.
    device: str
        Specify the device to use: "cuda" or "cpu".
    isolated_atom_energies: dict
        Dictionary of isolated atoms energies.
    config_type: str
        Specify the type of configurations generated from RSS.
    keep_symmetry: bool
        If true, preserve symmetry during relaxation.

    Returns
    -------
    str | None
        Output string containing path for the results of the RSS relaxation.
    """
    if hookean_paras is not None:
        hookean_paras = {
            ast.literal_eval(k) if isinstance(k, str) else k: v
            for k, v in hookean_paras.items()
        }

    mlip_path = mlip_path[0] if isinstance(mlip_path, list) else mlip_path

    if mlip_type == "GAP":
        gap_label = os.path.join(mlip_path, "gap_file.xml")
        gap_control = "Potential xml_label=" + extract_gap_label(gap_label)
        pot = CustomPotential(args_str=gap_control, param_filename=gap_label)

    elif mlip_type == "J-ACE":
        from ase.calculators.lammpslib import LAMMPSlib

        try:
            from lammps import lammps

            lmp = lammps()
            if "ML-PACE" not in lmp.installed_packages:
                raise RuntimeError(
                    "To use RSS flow with J-ACE potential, it needs LAMMPS compiled with"
                    "lammps-user-pace library. ML-PACE is not found in the current LAMMPS binary installation."
                    "Please follow the instructions in the autoplex documentation to install"
                    "LAMMPS with the required packages. Link to the documentation: "
                    " https://autoatml.github.io/autoplex/user/index.html#lammps-installation"
                )
        except Exception as exc:
            raise RuntimeError(
                "To use RSS flow with J-ACE potential, it needs LAMMPS compiled with"
                "python bindings and lammps-user-pace. Please follow the instructions in"
                "the autoplex documentation to install LAMMPS with the required packages. "
                "Link to the documentation:"
                " https://autoatml.github.io/autoplex/user/index.html#lammps-installation"
            ) from exc

        ace_label = os.path.join(mlip_path, "acemodel.yace")
        ace_json = os.path.join(mlip_path, "acemodel.json")
        ace_table = os.path.join(mlip_path, "acemodel_pairpot.table")

        atom_types, cmds = extract_pairstyle(ace_label, ace_json, ace_table)

        pot = LAMMPSlib(
            lmpcmds=cmds, atom_types=atom_types, log_file="test.log", keep_alive=True
        )

    elif mlip_type == "NEQUIP":
        nequip_label = os.path.join(mlip_path, "deployed_nequip_model.pth")
        if isolated_atom_energies:
            ele_syms = [
                chemical_symbols[int(e_num)] for e_num in isolated_atom_energies
            ]

        else:
            raise ValueError("isol_es is empty or not defined!")
        pot = NequIPCalculator.from_deployed_model(
            model_path=nequip_label,
            device=device,
            species_to_type_name={s: s for s in ele_syms},
            set_global_options=False,
        )

    elif mlip_type == "M3GNET":
        pot_file = matgl.load_model(path=mlip_path)
        pot = M3GNetCalculator(potential=pot_file)

    elif mlip_type == "MACE":
        mace_label = os.path.join(mlip_path, "checkpoints/MACE_model_run-123.model")
        pot = MACECalculator(model_paths=mace_label, device=device)

    unique_starting_index = atom.info["unique_starting_index"]
    log_file = output_file_name + "_" + str(unique_starting_index) + ".log"
    constraint_list = []
    if hookean_repul and hookean_paras:
        atom_num = atom.get_atomic_numbers()
        for i in range(len(atom_num)):
            for j in range(i + 1, len(atom_num)):
                if (
                    (atom_num[i], atom_num[j]) in hookean_paras
                    and hookean_paras[(atom_num[i], atom_num[j])][0] != 0
                    and hookean_paras[(atom_num[i], atom_num[j])][1] != 0
                ):
                    # print(f"Hookean repulsion is used for {atom_num[i]}-{atom_num[j]}!")
                    constraint_list.append(
                        HookeanRepulsion(
                            i, j, *hookean_paras[(atom_num[i], atom_num[j])]
                        )
                    )
                elif (
                    (atom_num[j], atom_num[i]) in hookean_paras
                    and hookean_paras[(atom_num[j], atom_num[i])][0] != 0
                    and hookean_paras[(atom_num[j], atom_num[i])][1] != 0
                ):
                    # print(f"Hookean repulsion is used for {atom_num[j]}-{atom_num[i]}!")
                    constraint_list.append(
                        HookeanRepulsion(
                            i, j, *hookean_paras[(atom_num[j], atom_num[i])]
                        )
                    )

    if keep_symmetry:
        print("Creating FixSymmetry calculator and maintaining initial symmetry!")
        constraint_list.append(FixSymmetry(atom, symprec=1.0e-4))

    if constraint_list:
        atom.set_constraint(constraint_list)

    atom.calc = pot

    if scalar_pressure_method == "exp":
        scalar_pressure_tmp = scalar_exp_pressure * GPa
        if scalar_pressure_exponential_width > 0.0:
            scalar_pressure_tmp *= np.random.exponential(
                scalar_pressure_exponential_width
            )
    elif scalar_pressure_method == "uniform":
        scalar_pressure_tmp = (
            np.random.uniform(low=scalar_pressure_low, high=scalar_pressure_high) * GPa
        )
    atom.info["RSS_applied_pressure"] = scalar_pressure_tmp / GPa
    atom = UnitCellFilter(atom, scalar_pressure=scalar_pressure_tmp)

    try:
        optimizer = PreconLBFGS(
            atom, precon=Exp(3), use_armijo=True, logfile=log_file, master=True
        )
        traj = []

        def build_traj():
            atom_copy = atom.copy()
            atom_copy.info["energy"] = atom.atoms.get_potential_energy()
            atom_copy.info["enthalpy"] = atom.get_potential_energy().copy()
            traj.append(atom_copy)

        optimizer.attach(build_traj)
        optimizer.run(fmax=force_tol, smax=stress_tol, steps=max_steps)

        minim_stat = "converged" if optimizer.converged() else "unconverged"

        for traj_at_i, traj_at in enumerate(traj):
            traj_at.info["RSS_minim_iter"] = traj_at_i
            traj_at.info["config_type"] = config_type
            traj_at.info["minim_stat"] = minim_stat

        if write_traj:
            traj_file_name = (
                output_file_name + "_traj_" + str(unique_starting_index) + ".extxyz"
            )
            ase.io.write(traj_file_name, traj, parallel=False)
        del traj[-1].info["minim_stat"]
        traj[-1].info["config_type"] = minim_stat + "_minimum"

        local_minima = traj[-1]

        if local_minima.info["config_type"] == "converged_minimum":
            dir_path = Path.cwd()
            return os.path.join(dir_path, traj_file_name)
        return None

    except RuntimeError:
        print("RuntimeError occurred during optimization! Return none!")
        return None


def minimize_structures(
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"],
    mlip_path: list[str],
    iteration_index: str,
    structures: list[Structure],
    output_file_name: str = "RSS_relax_results",
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 1000,
    force_tol: float = 0.01,
    stress_tol: float = 0.01,
    hookean_repul: bool = False,
    hookean_paras: dict[tuple[int, int], tuple[float, float]] | None = None,
    write_traj: bool = True,
    num_processes_rss: int = 1,
    device: str = "cpu",
    isolated_atom_energies: dict[int, float] | None = None,
    config_type: str = "traj",
    struct_start_index: int = 0,
    keep_symmetry: bool = True,
) -> list[str | None]:
    """Run RSS in parallel.

    Parameters
    ----------
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted.
    mlip_path: list[str]
        List of path to the MLIP model.
    iteration_index: str
        Index for the current iteration.
    structures: list[Structure]
        List of structures to be relaxed.
    output_file_name: str
        Prefix for the trajectory/log file name. The actual output file name
        may be composed of this prefix, an index, and file types.
    scalar_pressure_method: str
        Method for adding external pressures. Default is 'exp'.
    scalar_exp_pressure: float
        Scalar exponential pressure. Default is 100.
    scalar_pressure_exponential_width: float
        Width for scalar pressure exponential. Default is 0.2.
    scalar_pressure_low: float
        Low limit for scalar pressure. Default is 0.
    scalar_pressure_high: float
        High limit for scalar pressure. Default is 50.
    max_steps: int
        Maximum number of steps for relaxation. Default is 1000.
    force_tol: float
        Force residual tolerance for relaxation. Default is 0.01.
    stress_tol: float
        Stress residual tolerance for relaxation. Default is 0.01.
    hookean_repul: bool
        If true, apply Hookean repulsion. Default is False.
    hookean_paras: dict
        Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
    write_traj: bool
        If true, write trajectory of RSS. Default is True.
    num_processes_rss: int
        Number of processes used for running RSS.
    device: str
        Specify device to use "cuda" or "cpu".
    isolated_atom_energies: dict
        Dictionary of isolated atoms energies.
    config_type: str
        Specify the type of configurations generated from RSS
    struct_start_index: int
        Specify the starting index within a list
    keep_symmetry: bool
        If true, preserve symmetry during relaxation.

    Returns
    -------
    list
        Output list[str] containing paths for the results of the RSS relaxation.
    """
    atoms = [AseAtomsAdaptor().get_atoms(structure) for structure in structures]

    if hookean_repul:
        print("Hookean repulsion is used!")

    for i, atom in enumerate(atoms):
        atom.info["unique_starting_index"] = iteration_index + f"{i+struct_start_index}"

    args = [
        (
            atom,
            mlip_type,
            mlip_path,
            output_file_name,
            scalar_pressure_method,
            scalar_exp_pressure,
            scalar_pressure_exponential_width,
            scalar_pressure_low,
            scalar_pressure_high,
            max_steps,
            force_tol,
            stress_tol,
            hookean_repul,
            hookean_paras,
            write_traj,
            device,
            isolated_atom_energies,
            config_type,
            keep_symmetry,
        )
        for atom in atoms
    ]

    with Pool(processes=num_processes_rss) as pool:
        results = pool.starmap(process_rss, args)

    return list(results)


def split_structure_into_groups(structures: list, num_groups: int) -> list[list]:
    """
    Split a list of structures into several groups, with each group being its own list.

    Parameters
    ----------
    structures: list
        List of structures
    num_groups: int
        Number of structure groups, used for assigning tasks across multiple nodes,
        with each node handling one group.
    """
    base_size = len(structures) // num_groups
    remainder = len(structures) % num_groups

    structure_groups = []
    start_index = 0

    for i in range(num_groups):
        extra = 1 if i < remainder else 0
        group_size = base_size + extra

        structure_groups.append(structures[start_index : start_index + group_size])
        start_index += group_size

    return structure_groups
