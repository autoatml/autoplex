"""Jobs for running RSS."""

import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from shutil import which
from subprocess import run
from typing import Literal

import ase.io
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from jobflow import Flow, Maker, Response, job
from monty.dev import requires
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.data.common.utils import flatten
from autoplex.data.rss.utils import minimize_structures, split_structure_into_groups


@dataclass
class RandomizedStructure(Maker):
    """
    Maker to create random structures using the 'buildcell' tool.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    struct_number : int
        Expected number of generated randomized unit cells.
    tag: str
        Tag of systems. It can also be used for setting up elements and stoichiometry.
        For example, the tag of 'SiO2' will be recognized as a 1:2 ratio of Si to O and
        passed into the parameters of buildcell. However, note that this will be overwritten
        if the stoichiometric ratio of elements is defined in the 'cell_seed_paths' or 'buildcell_options'.
    output_file_name: str
        Name of the file to store all generated structures.
    remove_tmp_files: bool
        Remove all temporary files raised by buildcell to save memory.
    buildcell_option: dict
        Customized parameters for buildcell.
    cell_seed_path: str
        Path to the custom buildcell control file, which ends with '.cell'. If this file exists,
        the buildcell_option argument will no longer take effect.
    num_processes: int
        Number of processes to use for parallel computation.
    fragment: Atoms | list[Atoms] (optional)
        Fragment(s) for random structures, e.g. molecules, to be placed indivudally intact.
        atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
        atoms.cell must be defined (e.g. Atoms.cell = np.eye(3)*20).
    fragment_numbers: list[str] (optional)
        Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
    """

    name: str = "build_random_cells"
    struct_number: int = 20
    tag: str = "Si"
    output_file_name: str = "random_structs.extxyz"
    remove_tmp_files: bool = True
    buildcell_option: dict | None = None
    cell_seed_path: str | None = None
    num_processes: int = 32
    fragment_file: str | None = None
    fragment_numbers: list[str] | None = None

    @requires(
        which("buildcell"),
        "RSS flows requires the executable 'buildcell' to be in PATH. "
        "Please follow the instructions in the autoplex documentation to install "
        "the AIRSS library and add it to PATH. Link to the documentation:"
        " https://autoatml.github.io/autoplex/user/index.html#enabling-rss-workflows",
    )
    @job
    def make(self):
        """Maker to create random structures by buildcell."""
        if self.cell_seed_path:
            if not os.path.isfile(self.cell_seed_path):
                raise FileNotFoundError(
                    f"No file found at the specified path: {self.cell_seed_path}"
                )
            bc_file = self.cell_seed_path

        else:
            buildcell_parameters = [
                "SLACK=0.25",
                "OVERLAP=0.1",
                "COMPACT",
                "MINSEP=1.5",
            ]

            if self.buildcell_option is not None:
                buildcell_parameters = self._update_buildcell_option(
                    self.buildcell_option, buildcell_parameters
                )

            elements = self._extract_elements(self.tag)  # {"Si":1, "O":2}

            if (
                self.buildcell_option is not None
                and "SPECIES" in self.buildcell_option
                and self.fragment_file is not None
            ):
                raise ValueError(
                    "Cannot use 'SPECIES' and 'fragment' together in buildcell options.\n"
                    "Specify your fragment only and use NFORM to control their number."
                )

            if self.buildcell_option is None or (
                "SPECIES" not in self.buildcell_option and self.fragment_file is None
            ):
                make_species = self._make_species(elements)  # Si%NUM=1,O%NUM=2
                buildcell_parameters = self._update_buildcell_option(
                    {"SPECIES": make_species}, buildcell_parameters
                )

            if (
                self.buildcell_option is None
                or (
                    "VARVOL" not in self.buildcell_option
                    and "TARGVOL" not in self.buildcell_option
                )
                or "MINSEP" not in self.buildcell_option
            ):
                r0 = {}
                varvol = {}
                num_atom_formula = 0
                total_varvol_formula = 0

                for ele in elements:
                    r0[ele] = covalent_radii[atomic_numbers[ele]]

                    if Element(ele).is_metal:
                        varvol[ele] = 5.5 * np.power(r0[ele], 3)
                    else:
                        varvol[ele] = 14.5 * np.power(r0[ele], 3)

                    total_varvol_formula += varvol[ele] * elements[ele]

                    num_atom_formula += elements[ele]

                if self.buildcell_option is None or (
                    "VARVOL" not in self.buildcell_option
                    and "TARGVOL" not in self.buildcell_option
                ):
                    mean_var = total_varvol_formula / num_atom_formula * len(elements)
                    buildcell_parameters = self._update_buildcell_option(
                        {"TARGVOL": f"{mean_var*0.8}-{mean_var*1.2}"},
                        buildcell_parameters,
                    )

                if (
                    self.buildcell_option is None
                    or "MINSEP" not in self.buildcell_option
                ):
                    minsep = self._make_minsep(r0)
                    buildcell_parameters = self._update_buildcell_option(
                        {
                            "MINSEP": minsep,
                        },
                        buildcell_parameters,
                    )

            if self.fragment_file is not None:
                self.fragment = ase.io.read(self.fragment_file, index=":")

                if len(self.fragment) == 1:
                    self.fragment = self.fragment[0]

                if isinstance(self.fragment, Atoms):
                    if self.fragment_numbers is None:
                        fragment_numbers = [1 for _ in self.fragment]
                    else:
                        fragment_numbers = self.fragment_numbers
                    if "fragment_id" not in self.fragment.arrays:
                        self.fragment.arrays["fragment_id"] = [
                            f"{1}-f" for i in self.fragment
                        ]

                elif isinstance(self.fragment, list):
                    if self.fragment_numbers is None:
                        fragment_numbers = [
                            1 for _ in range(sum([len(i) for i in self.fragment]))
                        ]
                    else:
                        fragment_numbers = self.fragment_numbers
                    write_fragment = self.fragment[0]
                    for frag in self.fragment[
                        1:
                    ]:  # merge all separate fragments into one Atoms object
                        write_fragment += frag

                fragment_parameters = [
                    "%BLOCK POSITIONS_ABS",
                ]
                symbols = self.fragment.get_chemical_symbols()
                for i, val in enumerate(self.fragment.get_positions(wrap=True)):
                    if i == 0:
                        newline = (
                            f"{symbols[i]} {val[0]:.8f} {val[1]:.8f} {val[2]:.8f}"
                            f" # {self.fragment.arrays['fragment_id'][i]}"
                            f" % NUM={fragment_numbers[i]}"
                        )
                    else:
                        newline = (
                            f"{symbols[i]} {val[0]:.8f} {val[1]:.8f} {val[2]:.8f}"
                            f" # {self.fragment.arrays['fragment_id'][i]}"
                        )
                    fragment_parameters.append(newline)
                fragment_parameters.append("%ENDBLOCK POSITIONS_ABS")

                buildcell_parameters = (
                    fragment_parameters + buildcell_parameters
                )  # prepend with structural info

            self._cell_seed(buildcell_parameters, self.tag)
            bc_file = f"{self.tag}.cell"

        with Pool(processes=self.num_processes) as pool:
            args = [
                (i, bc_file, self.tag, self.remove_tmp_files)
                for i in range(self.struct_number)
            ]
            atoms_group = pool.starmap(self._parallel_process, args)

        atoms_group = [
            atom for atom in atoms_group if not np.isnan(atom.get_positions()).any()
        ]

        ase.io.write(
            self.output_file_name, atoms_group, parallel=False, format="extxyz"
        )

        # structure = [AseAtomsAdaptor().get_structure(at) for at in atoms_group]
        return os.path.join(Path.cwd(), self.output_file_name)
        # return structure

    def _update_buildcell_option(self, updates, origin) -> list:
        """
        Update buildcell parameters based on a dictionary of updates.

        Parameters
        ----------
        updates: dict
            A dictionary consisting of new values to update buildcell parameters.
        origin: list
            The default list of buildcell parameters.
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
        buildcell_parameters: list,
        tag: str,
    ):
        """
        Prepare the seed file for buildcell.

        Parameters
        ----------
        buildcell_parameters: (list of str) e.g. ['VARVOL=20']
            List of parameters for creating the seed file for buildcell.
        tag: str
            Tag of systems.
        """
        bc_file = f"{tag}.cell"
        contents = []
        flag = False  # for printing blocks correctly with '#'
        for i in buildcell_parameters:
            if i.startswith("%") or flag:
                flag = not (flag and i.startswith("%"))
                contents.append(i + "\n")
            else:
                contents.append("#" + i + "\n")

        with open(bc_file, "w") as f:
            f.writelines(contents)

    def _extract_elements(self, input_str: str) -> dict[str, int]:
        """
        Extract elements and their counts from a chemical formula string.

        Parameters
        ----------
        input_str: str
            A string representing a chemical formula (e.g., "SiO2").

        Returns
        -------
        Dict[str, int]
            A dictionary. For example, the input "SiO2" would return {"Si": 1, "O": 2}.
        """
        elements: dict[str, int] = {}
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

    def _make_species(self, elements: dict[str, int]) -> str:
        """
        Create a formatted string from a dictionary of element symbols and their counts.

        Parameters
        ----------
        elements: dict
            A dictionary of element symbols and their counts, e.g., {"Si": 1, "O": 2}.

        Returns
        -------
        str
            A formatter string. For example, the input {"Si": 1, "O": 2} would return "Si%NUM=1,O%NUM=2".
        """
        output = ""
        for element, count in elements.items():
            output += f"{element}%NUM={count},"
        return output[:-1]

    def _make_minsep(self, r: dict[str, float]) -> str:
        """
        Generate a minsep string based on the radii of the elements.

        Parameters
        ----------
        r: dict
            A dictionary of element symbols and their atomic radii.

        Returns
        -------
        str
            A formatted string. For example, the input {"Si": 1.1, "O": 0.66} would
            return "1.5 Si-Si=1.76 Si-O=1.408 O-O=1.056".

        TODO: set up robust heuristics for multi-component systems
        """
        keys = list(r.keys())
        if len(keys) == 1:
            return str(1.6 * r[keys[0]])

        minsep = "1.5 "
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                el1, el2 = keys[i], keys[j]
                r1, r2 = r[el1], r[el2]
                result = (r1 + r2) / 2 * 1.6

                minsep += f"{el1}-{el2}={result} "

        return minsep[:-1]

    def _parallel_process(
        self, i: int, bc_file: str, tag: str, remove_tmp_files: bool
    ) -> Atoms:
        """
        Run the 'buildcell' command in parallel.

        Parameters
        ----------
        i: int
            Unique index to differentiate temporary files.
        bc_file: str
            Path to the input 'buildcell' file.
        tag: str
            Tag used to differentiate temporary files.
        remove_tmp_files: bool
            If True, remove temporary files after processing.

        """
        tmp_file_name = "tmp." + str(i) + "." + tag + ".cell"
        tmp_error_file_name = "tmp_error." + str(i) + "." + tag + ".cell"

        with (
            open(bc_file) as bc_file_handle,
            open(tmp_file_name, "w") as tmp_file_handle,
            open(tmp_error_file_name, "w") as tmp_error_file_handle,
        ):
            run(
                "buildcell",
                stdin=bc_file_handle,
                stdout=tmp_file_handle,
                stderr=tmp_error_file_handle,
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
            os.remove(tmp_error_file_name)

        return atom


@job
def do_rss_single_node(
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
    num_processes_rss: int = 1,
    device: str = "cpu",
    isolated_atom_energies: dict[int, float] | None = None,
    struct_start_index: int = 0,
    config_type: str = "traj",
    keep_symmetry: bool = True,
) -> list[str | None]:
    """
    Perform sandom structure searching (RSS) on one node using a machine learning interatomic potential (MLIP).

    Parameters
    ----------
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted.
    mlip_path: list[str]
        List of Path to the MLIP model.
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
    num_processes_rss: int
        Number of processes used for running RSS.
    device: str
        Specify device to use "cuda" or "cpu".
    isolated_atom_energies: dict
        Dictionary of isolated atoms energies.
    struct_start_index: int
        Specify the starting index within a list
    config_type: str
        Specify the type of configurations generated from RSS
    keep_symmetry: bool
        If true, preserve symmetry during relaxation.

    Returns
    -------
    list
        Output list[str] containing paths for the results of the RSS relaxation.
    """
    return minimize_structures(
        mlip_type=mlip_type,
        mlip_path=mlip_path,
        iteration_index=iteration_index,
        structures=structures,
        output_file_name=output_file_name,
        scalar_pressure_method=scalar_pressure_method,
        scalar_exp_pressure=scalar_exp_pressure,
        scalar_pressure_exponential_width=scalar_pressure_exponential_width,
        scalar_pressure_low=scalar_pressure_low,
        scalar_pressure_high=scalar_pressure_high,
        max_steps=max_steps,
        force_tol=force_tol,
        stress_tol=stress_tol,
        hookean_repul=hookean_repul,
        hookean_paras=hookean_paras,
        num_processes_rss=num_processes_rss,
        device=device,
        isolated_atom_energies=isolated_atom_energies,
        struct_start_index=struct_start_index,
        config_type=config_type,
        keep_symmetry=keep_symmetry,
    )


@job
def do_rss_multi_node(
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"],
    mlip_path: list[str],
    iteration_index: str,
    structure: list[Structure] | list[list[Structure]] | None = None,
    structure_paths: str | list[str] | None = None,
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
    num_processes_rss: int = 1,
    device: str = "cpu",
    isolated_atom_energies: dict[int, float] | None = None,
    num_groups: int = 1,
    config_type: str = "traj",
    keep_symmetry: bool = True,
) -> list[list | None]:
    """
    Perform sandom structure searching (RSS) on multiple nodes using a machine learning interatomic potential (MLIP).

    Parameters
    ----------
    mlip_type: Literal["GAP", "J-ACE", "NEP", "NEQUIP", "M3GNET", "MACE"]
        Choose one specific MLIP type to be fitted.
    mlip_path: list[str]
        List of Path to the MLIP model.
    iteration_index: str
        Index for the current iteration.
    structure: list[Structure]
        List of structures to be relaxed.
    structure_paths: str | list[str]
        Path(s) to structures to be used in the RSS process.
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
    num_processes_rss: int
        Number of processes used for running RSS.
    device: str
        Specify device to use "cuda" or "cpu".
    isolated_atom_energies: dict
        Dictionary of isolated atoms energies.
    num_groups: int
        Number of structure groups, used for assigning tasks across multiple nodes,
        with each node handling one group.
    config_type: str
        Specify the type of configurations generated from RSS
    keep_symmetry: bool
        If true, preserve symmetry during relaxation.

    Returns
    -------
    list
        Output list[str] containing paths for the results of the RSS relaxation.
    """
    job_list = []

    if structure_paths is not None:
        if isinstance(structure_paths, list):
            atoms = [ase.io.read(dir, index=":") for dir in structure_paths]
            atoms = flatten(atoms, recursive=True)
        elif isinstance(structure_paths, str):
            atoms = ase.io.read(structure_paths, index=":")
        structure = [AseAtomsAdaptor().get_structure(at) for at in atoms]
    elif structure is not None and isinstance(structure[0], list):
        structure = flatten(structure, recursive=False)

    if structure is not None and isinstance(structure, list):
        structure_groups = split_structure_into_groups(structure, num_groups)
    else:
        raise ValueError("Invalid structure format. It must be a list of structures.")

    rss_info = []

    struct_start_index = 0

    for i in range(num_groups):
        rss = do_rss_single_node(
            mlip_type=mlip_type,
            mlip_path=mlip_path,
            iteration_index=iteration_index,
            structures=structure_groups[i],
            output_file_name=output_file_name,
            scalar_pressure_method=scalar_pressure_method,
            scalar_exp_pressure=scalar_exp_pressure,
            scalar_pressure_exponential_width=scalar_pressure_exponential_width,
            scalar_pressure_low=scalar_pressure_low,
            scalar_pressure_high=scalar_pressure_high,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            hookean_repul=hookean_repul,
            hookean_paras=hookean_paras,
            num_processes_rss=num_processes_rss,
            device=device,
            isolated_atom_energies=isolated_atom_energies,
            struct_start_index=struct_start_index,
            config_type=config_type,
            keep_symmetry=keep_symmetry,
        )

        struct_start_index += len(structure_groups[i])

        job_list.append(rss)
        rss_info.append(rss.output)

    return Response(replace=Flow(job_list), output=rss_info)
