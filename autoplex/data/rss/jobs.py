"""Jobs for running RSS."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

import ase.io
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from jobflow import Flow, Maker, Response, job

from autoplex.data.rss.utils import minimize_structures, split_structure_into_groups


@dataclass
class RandomizedStructure(Maker):
    """
    Maker to create random structures by 'buildcell'.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    struct_number : int
        Expected number of generated randomized unit cells.
    tag : (str)
        name of the seed file for builcell.
    input_file_name: str
        input file of buildcell to set parameters
    output_file_name : str
        A file to store all generated structures.
    remove_tmp_files : bool
        Remove all temporary files raised by buildcell to save memory
    buildcell_options : dict
        Customized parameters for buildcell
    cell_seed_path : str
        Path to the custom buildcell control file, which ends with ".cell". If this file exists,
        the buildcell_options argument will no longer take effect
    num_processes: int
        number of processes to use for parallel computation.
    """

    name: str = "Build_random_cells"
    struct_number: int = 20
    tag: str = "Si"
    output_file_name: str = "random_structs.extxyz"
    remove_tmp_files: bool = True
    buildcell_options: dict | None = None
    cell_seed_path: str | None = None
    num_processes: int = 32

    @job
    def make(self):
        """Maker to create random structures by buildcell."""
        if self.cell_seed_path:
            if not os.path.isfile(self.cell_seed_path):
                raise FileNotFoundError(
                    f"No file found at the specified path: {self.cell_seed_path}"
                )
            bt_file = self.cell_seed_path

        else:
            buildcell_parameters = [
                "VARVOL=15",
                "SPECIES=Si%NUM=1",
                "NFORM=1-7",
                "SYMMOPS=1-8",
                "SLACK=0.25",
                "OVERLAP=0.1",
                "COMPACT",
                "MINSEP=1.5",
            ]

            buildcell_parameters = self._update_buildcell_options(
                self.buildcell_options, buildcell_parameters
            )

            elements = self._extract_elements(self.tag)  # {"Si":1, "O":2}

            if "SPECIES" not in self.buildcell_options:
                make_species = self._make_species(elements)  # Si%NUM=1,O%NUM=2
                buildcell_parameters = self._update_buildcell_options(
                    {"SPECIES": make_species}, buildcell_parameters
                )

            if (
                "VARVOL" not in self.buildcell_options
                or "MINSEP" not in self.buildcell_options
            ):
                r0 = {}
                varvol = {}
                num_atom_formula = 0
                total_varvol_formula = 0

                for ele in elements:
                    r0[ele] = covalent_radii[atomic_numbers[ele]]

                    if self._is_metal(ele):
                        varvol[ele] = 4.5 * np.power(r0[ele], 3)
                    else:
                        varvol[ele] = 12.0 * np.power(r0[ele], 3)

                    total_varvol_formula += varvol[ele] * elements[ele]

                    num_atom_formula += elements[ele]

                if "VARVOL" not in self.buildcell_options:
                    mean_var = total_varvol_formula / num_atom_formula * len(elements)
                    buildcell_parameters = self._update_buildcell_options(
                        {"VARVOL": mean_var}, buildcell_parameters
                    )

                if "MINSEP" not in self.buildcell_options:
                    minsep = self._make_minsep(r0)
                    buildcell_parameters = self._update_buildcell_options(
                        {
                            "MINSEP": minsep,
                        },
                        buildcell_parameters,
                    )

            self._cell_seed(buildcell_parameters, self.tag)
            bt_file = f"{self.tag}.cell"

        with Pool(processes=self.num_processes) as pool:
            args = [
                (i, bt_file, self.tag, self.remove_tmp_files)
                for i in range(self.struct_number)
            ]
            atoms_group = pool.starmap(self._parallel_process, args)

        ase.io.write(
            self.output_file_name, atoms_group, parallel=False, format="extxyz"
        )

        dir_path = Path.cwd()

        return os.path.join(dir_path, self.output_file_name)

    def _update_buildcell_options(self, updates, origin):
        """
        Update buildcell options based on a dictionary of updates.

        Parameters
        ----------
        updates : dict
            A dictionary with option as key and new value as value.
        """
        updated_keys = set()

        for i, option in enumerate(origin):
            option_key = option.split("=")[0]
            if option_key in updates:
                origin[i] = f"{option_key}={updates[option_key]}"
                updated_keys.add(option_key)

        for key, value in updates.items():
            if key not in updated_keys:
                origin.append(f"{key}={value}")

        return origin

    def _cell_seed(
        self,
        buildcell_options,
        tag,
    ):
        """
        Prepare random cells in self.directory.

        Arguments:
        buildcell options :: (list of str) e.g. ['VARVOL=20']
        """
        bc_file = f"{tag}.cell"
        contents = []
        contents.extend(["#" + i + "\n" for i in buildcell_options])

        with open(bc_file, "w") as f:
            f.writelines(contents)

    def _is_metal(self, element_symbol):
        metals = [
            "Li",
            "Be",
            "Na",
            "Mg",
            "Al",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]

        return element_symbol in metals

    def _extract_elements(self, input_str):
        elements = {}
        pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
        matches = pattern.findall(input_str)

        for match in matches:
            element, count = match
            count = int(count) if count else 1
            if element in elements:
                elements[element] += count
            else:
                elements[element] = count

        return elements

    def _make_species(self, elements):
        output = ""
        for element, count in elements.items():
            output += f"{element}%NUM={count},"
        return output[:-1]

    def _make_minsep(self, r):
        keys = list(r.keys())
        if len(keys) == 1:
            return str(1.5 * r[keys[0]])

        minsep = "1.5 "
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                el1, el2 = keys[i], keys[j]
                r1, r2 = r[el1], r[el2]
                if el1 == el2:
                    result = r1 * 2.0
                else:
                    result = (
                        (r1 + r2) / 2 * 2.0
                        if self._is_metal(el1) and self._is_metal(el2)
                        else (r1 + r2) / 2 * 1.5
                    )

                minsep += f"{el1}-{el2}={result} "

        return minsep[:-1]

    def _parallel_process(self, i, bt_file, tag, remove_tmp_files):
        tmp_file_name = "tmp." + str(i) + "." + tag + ".cell"

        with open(bt_file) as bt_file_handle, open(
            tmp_file_name, "w"
        ) as tmp_file_handle:
            run(
                "buildcell",
                stdin=bt_file_handle,
                stdout=tmp_file_handle,
                shell=True,
                check=True,
            )

        atom = ase.io.read(tmp_file_name, parallel=False)
        atom.info["unique_starting_index"] = i

        if "castep_labels" in atom.arrays:
            del atom.arrays["castep_labels"]

        if "initial_magmoms" in atom.arrays:
            del atom.arrays["initial_magmoms"]

        if remove_tmp_files:
            os.remove(tmp_file_name)

        return atom


@job
def do_rss_single_node(
    mlip_type: str | None = None,
    iteration_index: str | None = None,
    mlip_path: str | None = None,
    structure: list[Structure] | None = None,
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 1000,
    force_tol: float = 0.01,
    stress_tol: float = 0.01,
    Hookean_repul: bool = False,
    hookean_paras: dict[tuple[int, int], tuple[float, float]] | None = None,
    write_traj: bool = True,
    num_processes_rss: int = 1,
    device: str = "cpu",
    isol_es: dict[int, float] | None = None,
    struct_start_index: int = 0,
    config_type: str = "traj",
) -> list[str | None]:
    """
    Perform sandom structure searching (RSS) using a MLIP.

    Parameters
    ----------
    mlip_type : str, mandatory
        Choose one specific MLIP type:
        'GAP' | 'ACE' | 'NequIP' | 'M3GNet' | 'MACE'.
    iteration_index : str, mandatory
        Index for the current iteration.
    mlip_path : str, mandatory
        Path to the MLIP model.
    structure : list of Structure, mandatory
        List of structures to be relaxed.
    scalar_pressure_method : str
        Method for scalar pressure. Default is 'exp'.
    scalar_exp_pressure : float
        Scalar exponential pressure. Default is 100.
    scalar_pressure_exponential_width : float
        Width for scalar pressure exponential. Default is 0.2.
    scalar_pressure_low : float
        Low limit for scalar pressure. Default is 0.
    scalar_pressure_high : float
        High limit for scalar pressure. Default is 50.
    max_steps : int
        Maximum number of steps for relaxation. Default is 1000.
    force_tol : float
        Force tolerance for relaxation. Default is 0.01.
    stress_tol : float
        Stress tolerance for relaxation. Default is 0.01.
    Hookean_repul : bool
        Whether to apply Hookean repulsion. Default is False.
    hookean_paras : dict
        Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
    write_traj : bool
        Whether to write trajectory. Default is True.
    num_processes_rss: int
        Number of processes used for running RSS.
    device: str
        Specify device to use cuda or cpu.
    struct_start_index: int
        Specify the starting index within a list
    config_type: str
        Specify the type of configurations generated from RSS

    Returns
    -------
    list
        Output list[str] containing paths for the results of the RSS relaxation.
    """
    return minimize_structures(
        mlip_path=mlip_path,
        index=iteration_index,
        input_structure=structure,
        output_file_name="RSS_relax_results",
        mlip_type=mlip_type,
        scalar_pressure_method=scalar_pressure_method,
        scalar_exp_pressure=scalar_exp_pressure,
        scalar_pressure_exponential_width=scalar_pressure_exponential_width,
        scalar_pressure_low=scalar_pressure_low,
        scalar_pressure_high=scalar_pressure_high,
        max_steps=max_steps,
        force_tol=force_tol,
        stress_tol=stress_tol,
        Hookean_repul=Hookean_repul,
        hookean_paras=hookean_paras,
        write_traj=write_traj,
        num_processes_rss=num_processes_rss,
        device=device,
        isol_es=isol_es,
        struct_start_index=struct_start_index,
        config_type=config_type,
    )


@job
def do_rss_multi_node(
    mlip_type: str | None = None,
    iteration_index: str | None = None,
    mlip_path: str | None = None,
    structure: list[Structure] | None = None,
    scalar_pressure_method: str = "exp",
    scalar_exp_pressure: float = 100,
    scalar_pressure_exponential_width: float = 0.2,
    scalar_pressure_low: float = 0,
    scalar_pressure_high: float = 50,
    max_steps: int = 1000,
    force_tol: float = 0.01,
    stress_tol: float = 0.01,
    Hookean_repul: bool = False,
    hookean_paras: dict[tuple[int, int], tuple[float, float]] | None = None,
    write_traj: bool = True,
    num_processes_rss: int = 1,
    device: str = "cpu",
    isol_es: dict[int, float] | None = None,
    num_groups: int = 5,
    config_type: str = "traj",
) -> list[list | None]:
    """
    Perform sandom structure searching (RSS) using a MLIP.

    Parameters
    ----------
    mlip_type : str, mandatory
        Choose one specific MLIP type:
        'GAP' | 'ACE' | 'NequIP' | 'M3GNet' | 'MACE'.
    iteration_index : str, mandatory
        Index for the current iteration.
    mlip_path : str, mandatory
        Path to the MLIP model.
    structure : list of Structure, mandatory
        List of structures to be relaxed.
    scalar_pressure_method : str, mandatory
        Method for scalar pressure. Default is 'exp'.
    scalar_exp_pressure : float, optional
        Scalar exponential pressure. Default is 100.
    scalar_pressure_exponential_width : float, optional
        Width for scalar pressure exponential. Default is 0.2.
    scalar_pressure_low : float, optional
        Low limit for scalar pressure. Default is 0.
    scalar_pressure_high : float, optional
        High limit for scalar pressure. Default is 50.
    max_steps : int, mandatory
        Maximum number of steps for relaxation. Default is 1000.
    force_tol : float, mandatory
        Force tolerance for relaxation. Default is 0.01.
    stress_tol : float, optional
        Stress tolerance for relaxation. Default is 0.01.
    Hookean_repul : bool, optional
        Whether to apply Hookean repulsion. Default is False.
    hookean_paras : dict, optional
        Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
    write_traj : bool, mandatory
        Whether to write trajectory. Default is True.
    num_processes_rss: int, mandatory
        Number of processes used for running RSS on one node.
    device: str, mandatory
        Specify device to use cuda or cpu.
    num_groups: int, mandatory
        Number of structure groups, used for assigning tasks across multiple nodes,
        with each node handling one group.
    config_type: str
        Specify the type of configurations generated from RSS

    Returns
    -------
    list
        Output list containing paths for the results of the RSS relaxation.
    """
    job_list = []

    if structure is None:
        raise ValueError("The 'structure' cannot be None!")

    structure_groups = split_structure_into_groups(structure, num_groups)

    rss_info = []

    struct_start_index = 0

    for i in range(num_groups):
        rss = do_rss_single_node(
            mlip_type=mlip_type,
            iteration_index=iteration_index,
            mlip_path=mlip_path,
            structure=structure_groups[i],
            scalar_pressure_method=scalar_pressure_method,
            scalar_exp_pressure=scalar_exp_pressure,
            scalar_pressure_exponential_width=scalar_pressure_exponential_width,
            scalar_pressure_low=scalar_pressure_low,
            scalar_pressure_high=scalar_pressure_high,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            Hookean_repul=Hookean_repul,
            hookean_paras=hookean_paras,
            write_traj=write_traj,
            num_processes_rss=num_processes_rss,
            device=device,
            isol_es=isol_es,
            struct_start_index=struct_start_index,
            config_type=config_type,
        )

        struct_start_index += len(structure_groups[i])

        job_list.append(rss)
        rss_info.append(rss.output)

    return Response(replace=Flow(job_list), output=rss_info)
