"""Flows to create and check training data."""

import logging
import traceback
from dataclasses import dataclass, field

from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
    update_user_potcar_settings,
)
from atomate2.vasp.sets.core import StaticSetGenerator
from emmet.core.math import Matrix3D
from jobflow import Flow, Maker, Response, job
from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.data.common.jobs import (
    convert_to_extxyz,
    generate_randomized_structures,
    get_supercell_job,
    plot_force_distribution,
)
from autoplex.data.common.utils import (
    ElementCollection,
    flatten,
)

__all__ = ["DFTStaticLabelling", "GenerateTrainingDataForTesting"]

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


@dataclass
class GenerateTrainingDataForTesting(Maker):
    """Maker for generating training data to test it and check the forces.

    This Maker will first generate training data based on the chosen ML model (default is GAP)
    by randomizing (ase rattle) atomic displacements in supercells of the provided input structures.
    Then it will proceed with MLIP-based Phonon calculations (based on atomate2 PhononMaker), collect
    all structure data in extended xyz files and plot the forces in histograms (per rescaling cell_factor
    and total).

    Parameters
    ----------
    name: str
        Name of the flow.
    bulk_relax_maker: ForceFieldRelaxMaker | None
        Maker for the relax jobs.
    static_energy_maker: ForceFieldStaticMaker | ForceFieldRelaxMaker | None
        Maker for the static jobs.

    """

    name: str = "generate_training_data_for_testing"
    bulk_relax_maker: ForceFieldRelaxMaker | None = None
    static_energy_maker: ForceFieldStaticMaker | ForceFieldRelaxMaker | None = None

    def make(
        self,
        train_structure_list: list[Structure],
        cell_factor_sequence: list[float] | None = None,
        potential_filename: str = "gap.xml",
        n_structures: int = 50,
        rattle_std: float = 0.01,
        relax_cell: bool = True,
        steps: int = 1000,
        supercell_matrix: Matrix3D | None = None,
        config_type: str = "train",
        x_min: int = 0,
        x_max: int = 5,
        bin_width: float = 0.125,
        **relax_kwargs,
    ):
        """
        Generate ase.rattled structures from the training data and returns histogram plots of the forces.

        Parameters
        ----------
        train_structure_list: list[Structure].
            List of pymatgen structures object.
        cell_factor_sequence: list[float]
            List of factor to resize cell parameters.
        potential_filename: str
            The param_file_name for :obj:`quippy.potential.Potential()'`.
        n_structures : int.
            Total number of randomly displaced structures to be generated.
        rattle_std: float.
            Rattle amplitude (standard deviation in normal distribution).
            Default=0.01.
        relax_cell : bool
            Whether to allow the cell shape/volume to change during relaxation.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        supercell_matrix: Matrix3D | None
            The matrix to generate the supercell.
        config_type: str
            Configuration type of the data.
        x_min: int
            Minimum value for the plot x-axis.
        x_max: int
            Maximum value for the plot x-axis.
        bin_width: float
            Width of the plot bins.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.

        Returns
        -------
        Matplotlib plots "count vs. forces".
        """
        jobs = []
        if cell_factor_sequence is None:
            cell_factor_sequence = [0.975, 1.0, 1.025, 1.05]
        for structure in train_structure_list:
            if self.bulk_relax_maker is None:
                self.bulk_relax_maker = ForceFieldRelaxMaker(
                    calculator_kwargs={
                        "args_str": "IP GAP",
                        "param_filename": str(potential_filename),
                    },
                    force_field_name="GAP",
                    relax_cell=relax_cell,
                    steps=steps,
                )
            if supercell_matrix is None:
                supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

            bulk_relax = self.bulk_relax_maker.make(structure=structure)
            jobs.append(bulk_relax)
            supercell = get_supercell_job(
                structure=bulk_relax.output.structure,
                supercell_matrix=supercell_matrix,
            )
            jobs.append(supercell)

            for cell_factor in cell_factor_sequence:
                rattled_job = generate_randomized_structures(
                    structure=supercell.output,
                    n_structures=n_structures,
                    volume_custom_scale_factors=[cell_factor],
                    rattle_std=rattle_std,
                )
                jobs.append(rattled_job)
                static_conv_jobs = self.static_run_and_convert(
                    rattled_job.output,
                    cell_factor,
                    config_type,
                    potential_filename,
                    **relax_kwargs,
                )
                jobs.append(static_conv_jobs)
                plots = plot_force_distribution(
                    cell_factor, static_conv_jobs.output, x_min, x_max, bin_width
                )
                jobs.append(plots)

        return Flow(jobs=jobs, name=self.name)  # , plots.output)

    @job
    def static_run_and_convert(
        self,
        structure_list: list[Structure],
        cell_factor: float,
        config_type,
        potential_filename,
        **relax_kwargs,
    ):
        """
        Job for the static runs and the data conversion to the extxyz format.

        Parameters
        ----------
        structure_list: list[Structure].
            List of pymatgen structures object.
        cell_factor: float
            Factor to resize cell parameters.
        config_type: str
            Configuration type of the data.
        potential_filename: str
            The param_file_name for :obj:`quippy.potential.Potential()'`.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.

        """
        jobs = []
        for rattled in structure_list:
            if relax_kwargs == {}:
                relax_kwargs = {
                    "interval": 50000,
                    "fmax": 0.5,
                    "traj_file": rattled.reduced_formula
                    + "_"
                    + f"{cell_factor}".replace(".", "")
                    + ".pkl",
                }
            if self.static_energy_maker is None:
                self.static_energy_maker = ForceFieldRelaxMaker(
                    calculator_kwargs={
                        "args_str": "IP GAP",
                        "param_filename": str(potential_filename),
                    },
                    force_field_name="GAP",
                    relax_cell=False,
                    relax_kwargs=relax_kwargs,
                    steps=1,
                )
            static_run = self.static_energy_maker.make(structure=rattled)
            jobs.append(static_run)
            conv_job = convert_to_extxyz(
                static_run.output,
                rattled.reduced_formula
                + "_"
                + f"{cell_factor}".replace(".", "")
                + ".pkl",
                config_type,
                f"{cell_factor}".replace(".", ""),
            )
            jobs.append(conv_job)

        return Response(replace=Flow(jobs), output=conv_job.output)


@dataclass
class DFTStaticLabelling(Maker):
    """
    Maker to set up and run VASP static calculations for input structures, including bulk, isolated atoms, and dimers.

    It supports custom VASP input parameters and error handlers.

    Parameters
    ----------
    name: str
        Name of the flow.
    isolated_atom: bool
        If true, perform single-point calculations for isolated atoms. Default is False.
    isolated_species: list[str]
        List of species for which to perform isolated atom calculations. If None,
        species will be automatically derived from the 'structures' list. Default is None.
    e0_spin: bool
        If true, include spin polarization in isolated atom and dimer calculations.
        Default is False.
    isolatedatom_box: list[float]
        List of the lattice constants for a isolated_atom configuration.
    dimer: bool
        If true, perform single-point calculations for dimers. Default is False.
    dimer_box: list[float]
        The lattice constants of a dimer box.
    dimer_species: list[str]
        List of species for which to perform dimer calculations. If None, species
        will be derived from the 'structures' list. Default is None.
    dimer_range: list[float]
        Range of distances for dimer calculations.
    dimer_num: int
        Number of different distances to consider for dimer calculations.
    custom_incar: dict
        Dictionary of custom VASP input parameters. If provided, will update the
        default parameters. Default is None.
    custom_potcar: dict
        Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
        Default is None.

    Returns
    -------
    dict
        A dictionary containing:
        - 'dirs_of_vasp': List of directories containing VASP data.
        - 'config_type': List of configuration types corresponding to each directory.
    """

    name: str = "do_dft_labelling"
    isolated_atom: bool = False
    isolated_species: list[str] | None = None
    e0_spin: bool = False
    isolatedatom_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer: bool = False
    dimer_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer_species: list[str] | None = None
    dimer_range: list[float] | None = None
    dimer_num: int = 21
    custom_incar: dict | None = None
    custom_potcar: dict | None = None

    @job
    def make(
        self,
        structures: list,
        config_type: str | None = None,
    ):
        """
        Maker to set up and run VASP static calculations.

        Parameters
        ----------
        structures : list[Structure] | list[list[Structure]]
            List of structures for which to run the VASP static calculations. If None,
            no bulk calculations will be performed. Default is None.
        config_type : str
            Configuration types corresponding to the structures. If None, defaults
            to 'bulk'. Default is None.
        """
        job_list = []

        if isinstance(structures[0], list):
            structures = flatten(structures, recursive=False)

        dirs: dict[str, list[str]] = {"dirs_of_vasp": [], "config_type": []}

        default_custom_set = {
            "ADDGRID": "True",
            "ENCUT": 520,
            "EDIFF": 1e-06,
            "ISMEAR": 0,
            "SIGMA": 0.01,
            "PREC": "Accurate",
            "ISYM": None,
            "KSPACING": 0.2,
            "NPAR": 8,
            "LWAVE": "False",
            "LCHARG": "False",
            "ENAUG": None,
            "GGA": None,
            "ISPIN": None,
            "LAECHG": None,
            "LELF": None,
            "LORBIT": None,
            "LVTOT": None,
            "NSW": None,
            "SYMPREC": None,
            "NELM": 100,
            "LMAXMIX": None,
            "LASPH": None,
            "AMIN": None,
        }

        if self.custom_incar is not None:
            default_custom_set.update(self.custom_incar)

        custom_set = default_custom_set

        st_m = StaticMaker(
            input_set_generator=StaticSetGenerator(user_incar_settings=custom_set),
            run_vasp_kwargs={"handlers": ()},
        )

        if self.custom_potcar is not None:
            st_m = update_user_potcar_settings(st_m, potcar_updates=self.custom_potcar)

        if structures:
            for idx, struct in enumerate(structures):
                static_job = st_m.make(structure=struct)
                static_job.name = f"static_bulk_{idx}"
                dirs["dirs_of_vasp"].append(static_job.output.dir_name)
                if config_type:
                    dirs["config_type"].append(config_type)
                else:
                    dirs["config_type"].append("bulk")
                job_list.append(static_job)

        if self.isolated_atom:
            try:
                if self.isolated_species is not None:
                    syms = self.isolated_species

                elif (self.isolated_species is None) and (structures is not None):
                    # Get the species from the database
                    atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                    syms = ElementCollection(atoms).get_species()

                for idx, sym in enumerate(syms):
                    lattice = Lattice.orthorhombic(
                        self.isolatedatom_box[0],
                        self.isolatedatom_box[1],
                        self.isolatedatom_box[2],
                    )
                    isolated_atom_struct = Structure(lattice, [sym], [[0.0, 0.0, 0.0]])
                    static_job = st_m.make(structure=isolated_atom_struct)
                    static_job.name = f"static_isolated_{idx}"
                    static_job = update_user_incar_settings(
                        static_job,
                        {"KSPACING": 100.0, "ALGO": "All", "KPAR": 1},
                    )

                    if self.e0_spin:
                        static_job = update_user_incar_settings(
                            static_job, {"ISPIN": 2}
                        )

                    dirs["dirs_of_vasp"].append(static_job.output.dir_name)
                    dirs["config_type"].append("IsolatedAtom")
                    job_list.append(static_job)

            except ValueError as e:
                logging.error(f"Unknown species of isolated atoms! Exception: {e}")
                traceback.print_exc()

        if self.dimer:
            try:
                atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                if self.dimer_species is not None:
                    dimer_syms = self.dimer_species
                elif (self.dimer_species is None) and (structures is not None):
                    # Get the species from the database
                    dimer_syms = ElementCollection(atoms).get_species()
                pairs_list = ElementCollection(atoms).find_element_pairs(dimer_syms)
                for pair in pairs_list:
                    for dimer_i in range(self.dimer_num):
                        if self.dimer_range is not None:
                            dimer_distance = self.dimer_range[0] + (
                                self.dimer_range[1] - self.dimer_range[0]
                            ) * float(dimer_i) / float(
                                self.dimer_num - 1 + 0.000000000001
                            )

                        lattice = Lattice.orthorhombic(
                            self.dimer_box[0],
                            self.dimer_box[1],
                            self.dimer_box[2],
                        )
                        dimer_struct = Structure(
                            lattice,
                            [pair[0], pair[1]],
                            [[0.0, 0.0, 0.0], [dimer_distance, 0.0, 0.0]],
                            coords_are_cartesian=True,
                        )

                        static_job = st_m.make(structure=dimer_struct)
                        static_job.name = f"static_dimer_{dimer_i}"
                        static_job = update_user_incar_settings(
                            static_job,
                            {"KSPACING": 100.0, "ALGO": "All", "KPAR": 1},
                        )

                        if self.e0_spin:
                            static_job = update_user_incar_settings(
                                static_job, {"ISPIN": 2}
                            )

                        dirs["dirs_of_vasp"].append(static_job.output.dir_name)
                        dirs["config_type"].append("dimer")
                        job_list.append(static_job)

            except ValueError:
                logging.error("Unknown atom types in dimers!")
                traceback.print_exc()

        return Response(replace=Flow(job_list), output=dirs)
