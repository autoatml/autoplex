"""Flows to create and check training data."""

import logging
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

from atomate2 import SETTINGS
from atomate2.common.jobs.phonons import get_supercell_size
from atomate2.common.jobs.utils import (
    structure_to_conventional,
    structure_to_primitive,
)
from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator
from emmet.core.math import Matrix3D
from jobflow import Flow, Maker, Response, job
from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.data.common.jobs import (
    collect_dft_data,
    convert_to_extxyz,
    generate_randomized_structures,
    get_supercell_job,
    plot_force_distribution,
)
from autoplex.data.common.utils import (
    ElementCollection,
    flatten,
)
from autoplex.misc.castep.jobs import BaseCastepMaker

__all__ = [
    "DFTStaticLabelling",
    "GenerateTrainingDataForTesting",
    "RattledTrainingDataMaker",
]


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


_DEFAULT_STATIC_ENERGY_MAKER = StaticMaker(
    input_set_generator=StaticSetGenerator(
        user_incar_settings={
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
    ),
    run_vasp_kwargs={"handlers": ()},
)


_DEFAULT_RELAXATION_MAKER = DoubleRelaxMaker.from_relax_maker(
    TightRelaxMaker(
        input_set_generator=TightRelaxSetGenerator(
            user_incar_settings={
                "ALGO": "Normal",
                "ISPIN": 1,
                "LAECHG": False,
                "ISMEAR": 0,
                "ENCUT": 700,
                "ISYM": 0,
                "SIGMA": 0.05,
                "LCHARG": False,  # Do not write the CHGCAR file
                "LWAVE": False,  # Do not write the WAVECAR file
                "LVTOT": False,  # Do not write LOCPOT file
                "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                "LOPTICS": False,  # No PCDAT file
                "NCORE": 4,
            }
        ),
        run_vasp_kwargs={"handlers": {}},
    )
)


@dataclass
class DFTStaticLabelling(Maker):
    """
    Maker to set up and run VASP static calculations for input structures, including bulk, isolated atoms, and dimers.

    Parameters
    ----------
    name: str
        Name of the flow.
    include_isolated_atom: bool
        If true, perform single-point calculations for isolated atoms. Default is False.
    isolated_species: list[str] | None
        List of species for which to perform isolated atom calculations. If None,
        species will be automatically derived from the 'structures' list. Default is None.
    e0_spin: bool
        If true, include spin polarization in isolated atom and dimer calculations.
        Default is False.
    isolatedatom_box: list[float]
        List of the lattice constants for a isolated_atom configuration.
    include_dimer: bool
        If true, perform single-point calculations for dimers. Default is False.
    dimer_box: list[float]
        The lattice constants of a dimer box.
    dimer_species: list[str] | None
        List of species for which to perform dimer calculations. If None, species
        will be derived from the 'structures' list. Default is None.
    dimer_range: list[float] | None
        Range of distances for dimer calculations.
    dimer_num: int
        Number of different distances to consider for dimer calculations.
    custom_incar: dict | None
        Dictionary of custom VASP input parameters. If provided, will update the
        default parameters. Default is None.
    custom_potcar: dict | None
        Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
        Default is None.
    static_energy_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker
        Maker for static energy jobs: either BaseVaspMaker (VASP-based) or BaseCastepMaker (CASTEP-based) or
        ForceFieldStaticMaker (force field-based). Defaults to StaticMaker (VASP-based).
    static_energy_maker_isolated_atoms: BaseVaspMaker | ForceFieldStaticMaker | None
        Maker for static energy jobs of isolated atoms: either BaseVaspMaker (VASP-based) or
        ForceFieldStaticMaker (force field-based) or None. If set to `None`, the parameters
        from `static_energy_maker` will be used as the default for isolated atoms. In this case,
        if `static_energy_maker` is a `StaticMaker`, all major settings will be inherited,
        except that `kspacing` will be automatically set to 100 to enforce a Gamma-point-only calculation.
        This is typically suitable for single-atom systems. Default is None. If a non-`StaticMaker` maker
        is used here, its output must include a `dir_name` field to ensure compatibility with downstream workflows.
    config_type : str
        Configuration types corresponding to the structures. If None, defaults
        to 'bulk'. Default is None.

    Returns
    -------
    dict
        A dictionary containing:
        - 'dirs_of_dft': List of directories containing DFT data.
        - 'config_type': List of configuration types corresponding to each directory.
    """

    name: str = "do_dft_labelling"
    include_isolated_atom: bool = False
    isolated_species: list[str] | None = None
    e0_spin: bool = False
    isolatedatom_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    include_dimer: bool = False
    dimer_box: list[float] = field(default_factory=lambda: [20, 20, 20])
    dimer_species: list[str] | None = None
    dimer_range: list[float] | None = None
    dimer_num: int = 21
    custom_incar: dict | None = None
    custom_potcar: dict | None = None
    static_energy_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker = (
        _DEFAULT_STATIC_ENERGY_MAKER
    )
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker | None
    ) = None
    config_type: str | None = (None,)

    @job
    def make(
        self,
        structures: list,
    ):
        """
        Maker to set up and run VASP static calculations.

        Parameters
        ----------
        structures : list[Structure] | list[list[Structure]]
            List of structures for which to run the VASP static calculations. If None,
            no bulk calculations will be performed. Default is None.
        """
        job_list = []

        if isinstance(structures[0], list):
            structures = flatten(structures, recursive=False)

        dirs: dict[str, list[str]] = {"dirs_of_dft": [], "config_type": []}

        if isinstance(self.static_energy_maker, StaticMaker):

            if self.custom_incar is not None:
                self.static_energy_maker.input_set_generator.user_incar_settings.update(
                    self.custom_incar
                )

            if self.custom_potcar is not None:
                self.static_energy_maker.input_set_generator.user_potcar_settings.update(
                    self.custom_potcar
                )

        st_m = self.static_energy_maker

        if structures:
            for idx, struct in enumerate(structures):
                static_job = st_m.make(structure=struct)
                static_job.name = f"static_bulk_{idx}"
                dirs["dirs_of_dft"].append(static_job.output.dir_name)
                if self.config_type:
                    dirs["config_type"].append(self.config_type)
                else:
                    dirs["config_type"].append("bulk")
                job_list.append(static_job)

        if self.include_isolated_atom:
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

                    if self.static_energy_maker_isolated_atoms is None:
                        static_job = st_m.make(structure=isolated_atom_struct)
                        if isinstance(self.static_energy_maker, StaticMaker):
                            static_job = update_user_incar_settings(
                                static_job,
                                {"KSPACING": 100.0, "KPAR": 1},
                            )

                            if self.e0_spin:
                                static_job = update_user_incar_settings(
                                    static_job, {"ISPIN": 2}
                                )
                    else:
                        static_job = self.static_energy_maker_isolated_atoms.make(
                            structure=isolated_atom_struct
                        )

                    static_job.name = f"static_isolated_{idx}"
                    dirs["dirs_of_dft"].append(static_job.output.dir_name)
                    dirs["config_type"].append("IsolatedAtom")
                    job_list.append(static_job)

            except ValueError as e:
                logging.error(f"Unknown species of isolated atoms! Exception: {e}")
                traceback.print_exc()

        if self.include_dimer:
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
                        if isinstance(self.static_energy_maker, StaticMaker):
                            static_job = update_user_incar_settings(
                                static_job,
                                {"KSPACING": 100.0, "KPAR": 1},
                            )
                            if self.e0_spin:
                                static_job = update_user_incar_settings(
                                    static_job, {"ISPIN": 2}
                                )

                        static_job.name = f"static_dimer_{dimer_i}"
                        dirs["dirs_of_dft"].append(static_job.output.dir_name)
                        dirs["config_type"].append("dimer")
                        job_list.append(static_job)

            except ValueError:
                logging.error("Unknown atom types in dimers!")
                traceback.print_exc()

        return Response(replace=Flow(job_list), output=dirs)


@dataclass
class RattledTrainingDataMaker(DFTStaticLabelling):
    """
    Build a DFT-labeled dataset of rattled or distorted atomic structures.

    Starting from a relaxed bulk structure generated by a relaxation maker, this class applies controlled
    perturbations such as atomic position rattling, volume scaling, and geometric distortions. The resulting
    structures are then labeled with DFT, producing a dataset suitable for training or benchmarking atomistic models.

    Parameters
    ----------
    name: str
        Name of the flow.
    bulk_relax_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldRelaxMaker
        Maker used to produce the relaxed structure that will be
        perturbed. Defaults to _DEFAULT_RELAXATION_MAKER.
    uc: bool
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    distort_type: int
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures: int.
        Target total number of structures to generate (after rattling). Default=10.
        - If `volume_custom_scale_factors` is None:
            The code generates `n_structures` different volume or angle distortions.
            Each is rattled once.
        - If `volume_custom_scale_factors` is defined:
            Given that the list length equals m, the total `n_structures` is distributed
            over these m scale factors:
                base = n_structures // m
                rem = n_structures % m
            Note that the last `rem` factors get one extra rattled structure.
            Example: volume_custom_scale_factors=[0.95,0.97,0.99], n_structures=10 -> counts=[3,3,4].
    volume_scale_factor_range: list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume -+10%.
    volume_custom_scale_factors: list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
    rattle_type: int.
        0- standard rattling, 1- Monte-Carlo rattling. Default=0.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
        Note that for MC rattling, displacements generated will roughly be
        rattle_mc_n_iter**0.5 * rattle_std for small values of n_iter.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
        Default=42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.
    min_distance: float
        Minimum separation allowed between any two atoms.
        Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    symprec: float
        Precision to determine symmetry.
    use_symmetrized_structure: str or None
        Primitive, conventional or None
    supercell_matrix : Matrix3D | None
        Explicit supercell matrix to use when expanding the reference cell.
        Default None.
    supercell_settings : dict
        Settings used to construct a supercell when supercell_matrix is not
        provided. Expected keys include min_length, max_length,
        prefer_90_degrees, and allow_orthorhombic.
    dft_ref_file : str
        Output filename to store generated structures in extxyz format.
        Default "dft_ref.extxyz".
    config_type : str
        Type identifier that will be attached to generated structures.
        Default "rattled_structures".
    """

    name: str = "do_rattling"
    bulk_relax_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldRelaxMaker = (
        _DEFAULT_RELAXATION_MAKER
    )
    uc: bool = False
    distort_type: int = 0
    n_structures: int = 10
    volume_custom_scale_factors: list[float] | None = None
    volume_scale_factor_range: list[float] | None = None
    rattle_type: int = 0
    rattle_std: float = 0.01
    rattle_seed: int = 42
    rattle_mc_n_iter: int = 10
    min_distance: float = 1.5
    angle_percentage_scale: float = 10
    angle_max_attempts: int = 1000
    w_angle: list[int] | None = None
    symprec: float = SETTINGS.SYMPREC
    use_symmetrized_structure: str | None = None
    supercell_matrix: Matrix3D | None = None
    supercell_settings: dict = field(
        default_factory=lambda: {
            "min_length": 15,
            "max_length": 20,
            "prefer_90_degrees": False,
            "allow_orthorhombic": False,
        }
    )
    dft_ref_file: str = "dft_ref.extxyz"
    config_type: str = "rattled_structures"

    @job
    def make(
        self,
        structure: Structure,
    ):
        """
        Generate and label a set of rattled or distorted structures based on a reference structure.

        Parameters
        ----------
        structure : Structure
            Input crystal structure to serve as the starting point. Typically
            the primitive or conventional cell of a bulk material.

        Returns
        -------
        dict
            A dictionary containing:
            - `pre_database_dir`: Path to the directory with collected DFT data.
            - `isolated_atom_energies`: Mapping of element symbols to their isolated-atom reference
              energies.
        """
        job_list = []
        final_structures = []

        if self.bulk_relax_maker is not None:
            relaxed = self.bulk_relax_maker.make(structure)
            job_list.append(relaxed)
            structure = relaxed.output.structure

        if self.use_symmetrized_structure == "primitive":
            prim_job = structure_to_primitive(structure, self.symprec)
            job_list.append(prim_job)
            structure = prim_job.output
        elif self.use_symmetrized_structure == "conventional":
            conv_job = structure_to_conventional(structure, self.symprec)
            job_list.append(conv_job)
            structure = conv_job.output

        if self.supercell_matrix is None:
            supercell_job = get_supercell_size(
                structure=structure,
                min_length=self.supercell_settings.get("min_length", 12),
                max_length=self.supercell_settings.get("max_length", 20),
                prefer_90_degrees=self.supercell_settings.get(
                    "prefer_90_degrees", False
                ),
                allow_orthorhombic=self.supercell_settings.get(
                    "allow_orthorhombic", False
                ),
            )
            job_list.append(supercell_job)
            supercell_matrix = supercell_job.output
        else:
            supercell_matrix = self.supercell_matrix

        rattle_job = generate_randomized_structures(
            structure=structure,
            supercell_matrix=supercell_matrix,
            distort_type=self.distort_type,
            n_structures=self.n_structures,
            volume_custom_scale_factors=self.volume_custom_scale_factors,
            volume_scale_factor_range=self.volume_scale_factor_range,
            rattle_std=self.rattle_std,
            min_distance=self.min_distance,
            angle_percentage_scale=self.angle_percentage_scale,
            angle_max_attempts=self.angle_max_attempts,
            rattle_type=self.rattle_type,
            rattle_seed=self.rattle_seed,
            rattle_mc_n_iter=self.rattle_mc_n_iter,
            w_angle=self.w_angle,
        )
        job_list.append(rattle_job)
        final_structures.append(rattle_job.output)

        if self.uc:
            rattle_uc_job = generate_randomized_structures(
                structure=structure,
                supercell_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                distort_type=self.distort_type,
                n_structures=self.n_structures,
                volume_custom_scale_factors=self.volume_custom_scale_factors,
                volume_scale_factor_range=self.volume_scale_factor_range,
                rattle_std=self.rattle_std,
                min_distance=self.min_distance,
                angle_percentage_scale=self.angle_percentage_scale,
                angle_max_attempts=self.angle_max_attempts,
                rattle_type=self.rattle_type,
                rattle_seed=self.rattle_seed,
                rattle_mc_n_iter=self.rattle_mc_n_iter,
                w_angle=self.w_angle,
            )
            job_list.append(rattle_uc_job)
            final_structures.append(rattle_uc_job.output)

        do_dft_static = DFTStaticLabelling.make(self, structures=final_structures)
        job_list.append(do_dft_static)

        do_data_collection = collect_dft_data(
            dft_ref_file=self.dft_ref_file,
            dft_dirs=do_dft_static.output,
        )
        job_list.append(do_data_collection)

        return Response(
            replace=Flow(job_list),
            output={
                "pre_database_dir": do_data_collection.output,
                "isolated_atom_energies": do_data_collection.output[
                    "isolated_atom_energies"
                ],
            },
        )


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
