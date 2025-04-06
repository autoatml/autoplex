"""Flows to create training data for ML potentials."""

from dataclasses import dataclass, field

from atomate2.common.jobs.phonons import run_phonon_displacements
from atomate2.forcefields.flows.phonons import PhononMaker as FFPhononMaker
from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator
from jobflow import Flow, Maker, Response, job
from pymatgen.core import Molecule, Site
from pymatgen.core.structure import Species, Structure

from autoplex.data.common.jobs import generate_randomized_structures
from autoplex.data.phonons.jobs import reduce_supercell_size_job
from autoplex.data.phonons.utils import (
    ml_phonon_maker_preparation,
    reduce_supercell_size,
)

__all__ = [
    "DFTPhononMaker",
    "IsoAtomMaker",
    "IsoAtomStaticMaker",
    "MLPhononMaker",
    "RandomStructuresDataGenerator",
    "TightDFTStaticMaker",
]


@dataclass
class TightDFTStaticMaker(PhononDisplacementMaker):
    """Adapted phonon displacement maker for static calculation.

    The input set used is same as PhononDisplacementMaker.
    Only difference is Spin polarization is switched off and Gaussian smearing is used

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        Generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "dft static"
    run_vasp_kwargs: dict = field(default_factory=lambda: {"handlers": ()})

    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_incar_settings={
                "ALGO": "Normal",  # not switching to Fast because it's not precise enough for the fit
                "IBRION": -1,
                "ISPIN": 1,
                "ISMEAR": 0,
                "ISIF": 3,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "NSW": 0,
                "LCHARG": False,  # Do not write the CHGCAR file
                "LWAVE": False,  # Do not write the WAVECAR file
                "LVTOT": False,  # Do not write LOCPOT file
                "LORBIT": None,  # remove Lorbit
                "LOPTICS": False,  # No PCDAT file
                "SIGMA": 0.05,
                "ISYM": 0,
                "SYMPREC": 1e-9,
                "KSPACING": 0.2,
                # To be removed
                "NPAR": 4,
            },
            auto_ispin=False,
        )
    )


@dataclass
class DFTPhononMaker(PhononMaker):
    """
    Adapted PhononMaker to calculate harmonic phonons with VASP and Phonopy.

    The input set used is same as PhononMaker from atomate2.
    Only difference is Spin polarization is switched off and Gaussian smearing is used

    Parameters
    ----------
    name : str = "phonon"
        Name of the flows produced by this maker.
    sym_reduce : bool = True
        Whether to reduce the number of deformations using symmetry.
    symprec : float = 1e-4
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float = 0.01
        Displacement distance for phonons
    min_length: float = 20.0
        Minimum length of the supercell that will be built
    prefer_90_degrees: bool = True
        If set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict = {}
        Keyword arguments that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str or None = None
        Allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        Maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        Maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .BaseVaspMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Arg that determines if thermal_displacement_matrices are computed
    kpath_scheme: str = "seekpath"
        Scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str = "vasp"
        Determines the DFT code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        If True, force constants will be stored
    """

    name: str = "dft phonon"
    sym_reduce: bool = True
    symprec: float = 1e-4
    displacement: float = 0.01
    min_length: float | None = 20.0
    max_length: float | None = 30.0
    prefer_90_degrees: bool = True
    allow_orthorhombic: bool = False
    get_supercell_size_kwargs: dict = field(
        default_factory=lambda: {"max_atoms": 800, "step_size": 1.0}
    )
    use_symmetrized_structure: str | None = None
    create_thermal_displacements: bool = False
    store_force_constants: bool = False
    generate_frequencies_eigenvectors_kwargs: dict = field(
        default_factory=lambda: {"tol_imaginary_modes": 1e-1}
    )
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(
            TightRelaxMaker(
                run_vasp_kwargs={"handlers": {}},
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
                        # to be removed
                        "NPAR": 4,
                    }
                ),
            )
        ),
    )
    static_energy_maker: BaseVaspMaker | None = None

    phonon_displacement_maker: BaseVaspMaker | None = field(
        default_factory=TightDFTStaticMaker
    )


@dataclass
class RandomStructuresDataGenerator(Maker):
    """
    Maker to generate DFT labelled training data for ML potential fitting based on random atomic displacements.

    This Maker performs the two following steps:
    1. Generates supercells from the provided structure and randomly displaces the atomic positions using ase rattle.
    (randomized unit cells can be generated additionally).
    2. Performs the static DFT (VASP) calculations on the randomized cells.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    displacement_maker: .BaseVaspMaker or None
        Maker used for a static calculation for a supercell.
    code: str
        Determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    uc: bool.
        If True, will use the unit cells of initial randomly displaced
        structures and add phonon static computation jobs to the flow
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
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
    supercell_settings: dict
        Settings for supercells.
    """

    name: str = "RandomStruturesDataGeneratorForML"
    displacement_maker: BaseVaspMaker | None = field(
        default_factory=TightDFTStaticMaker
    )
    bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: TightRelaxMaker(
            run_vasp_kwargs={"handlers": {}},
            input_set_generator=TightRelaxSetGenerator(
                user_incar_settings={
                    "ALGO": "Normal",
                    "ISPIN": 1,
                    "LAECHG": False,
                    "ISYM": 0,  # to be changed
                    "ISMEAR": 0,
                    "SIGMA": 0.05,  # to be changed back
                    "LCHARG": False,  # Do not write the CHGCAR file
                    "LWAVE": False,  # Do not write the WAVECAR file
                    "LVTOT": False,  # Do not write LOCPOT file
                    "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    # to be removed
                    "NPAR": 4,
                }
            ),
        )
    )
    code: str = "vasp"
    uc: bool = False
    distort_type: int = 0
    n_structures: int = 10
    min_distance: float = 1.5
    angle_percentage_scale: float = 10
    angle_max_attempts: int = 1000
    rattle_type: int = 0
    rattle_std: float = 0.01
    rattle_seed: int = 42
    rattle_mc_n_iter: int = 10
    w_angle: list[float] | None = None
    supercell_settings: dict | None = field(
        default_factory=lambda: {"min_length": 15, "max_length": 20}
    )

    def make(
        self,
        structure: Structure,
        mp_id: str,
        volume_custom_scale_factors: list[float] | None = None,
        volume_scale_factor_range: list[float] | None = None,
    ):
        """
        Make a flow to generate rattled structures reference DFT data.

        Parameters
        ----------
        structure :
            The pymatgen structures drawn from the Materials Project.
        mp_id: str
            Materials Project IDs
        volume_scale_factor_range : list[float]
            [min, max] of volume scale factors.
            e.g. [0.90, 1.10] will distort volume +-10%.
        volume_custom_scale_factors : list[float]
            Specify explicit scale factors (if range is not specified).
            If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
        """
        if self.supercell_settings is None:
            self.supercell_settings = field(
                default_factory=lambda: {"min_length": 15, "max_length": 20}
            )
        jobs = []  # initializing empty job list
        outputs = []
        if self.bulk_relax_maker is not None:
            relaxed = self.bulk_relax_maker.make(structure)
            jobs.append(relaxed)
            structure = relaxed.output.structure

        supercell_matrix = self.supercell_settings.get(mp_id, {}).get(
            "supercell_matrix"
        )
        if not supercell_matrix:
            supercell_matrix_job = reduce_supercell_size_job(
                structure=structure,
                min_length=self.supercell_settings.get("min_length", 12),
                max_length=self.supercell_settings.get("max_length", 20),
                fallback_min_length=self.supercell_settings.get(
                    "fallback_min_length", 10
                ),
                max_atoms=self.supercell_settings.get("max_atoms", 500),
                min_atoms=self.supercell_settings.get("min_atoms", 50),
                step_size=self.supercell_settings.get("step_size", 1.0),
            )
            jobs.append(supercell_matrix_job)
            supercell_matrix = supercell_matrix_job.output

        random_rattle_sc = generate_randomized_structures(
            structure=structure,
            supercell_matrix=supercell_matrix,
            distort_type=self.distort_type,
            n_structures=self.n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
            volume_scale_factor_range=volume_scale_factor_range,
            rattle_std=self.rattle_std,
            min_distance=self.min_distance,
            angle_percentage_scale=self.angle_percentage_scale,
            angle_max_attempts=self.angle_max_attempts,
            rattle_type=self.rattle_type,
            rattle_seed=self.rattle_seed,
            rattle_mc_n_iter=self.rattle_mc_n_iter,
            w_angle=self.w_angle,
        )
        jobs.append(random_rattle_sc)
        # perform the phonon displaced calculations for randomized displaced structures.
        #  The original structure is only needed to keep track of initial structure.
        vasp_random_sc_displacement_calcs = run_phonon_displacements(
            displacements=random_rattle_sc.output,  # pylint: disable=E1101
            structure=structure,
            supercell_matrix=None,
            phonon_maker=self.displacement_maker,
        )

        jobs.append(vasp_random_sc_displacement_calcs)
        outputs.append(vasp_random_sc_displacement_calcs.output["dirs"])

        if self.uc is True:
            random_rattle = generate_randomized_structures(
                structure=structure,
                supercell_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                distort_type=self.distort_type,
                n_structures=self.n_structures,
                volume_custom_scale_factors=volume_custom_scale_factors,
                volume_scale_factor_range=volume_scale_factor_range,
                rattle_std=self.rattle_std,
                min_distance=self.min_distance,
                angle_percentage_scale=self.angle_percentage_scale,
                angle_max_attempts=self.angle_max_attempts,
                rattle_type=self.rattle_type,
                rattle_seed=self.rattle_seed,
                rattle_mc_n_iter=self.rattle_mc_n_iter,
                w_angle=self.w_angle,
            )
            jobs.append(random_rattle)
            vasp_random_displacement_calcs = run_phonon_displacements(
                displacements=random_rattle.output,  # pylint: disable=E1101
                structure=structure,
                supercell_matrix=None,
                phonon_maker=self.displacement_maker,
            )

            jobs.append(vasp_random_displacement_calcs)
            outputs.append(vasp_random_displacement_calcs.output["dirs"])

        # create a flow including all jobs
        return Flow(jobs=jobs, output=outputs, name=self.name)


@dataclass
class MLPhononMaker(FFPhononMaker):
    """
    Maker to calculate harmonic phonons with a force field.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization
    effects, a correction of the dynamical matrix based on BORN charges can
    be performed. The BORN charges can be supplied manually.
    Finally, phonon densities of states, phonon band structures
    and thermodynamic properties are computed.

    Notes
    -----
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too
        many displacement calculations will be generated.
        It is recommended to check the convergence parameters here and
        adjust them if necessary. The default might not be strict enough
        for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        Displacement distance for phonons
    min_length: float
        Minimum length of the supercell that will be built
    prefer_90_degrees: bool
        If set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        Keyword arguments that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
        Allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will, however, use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker: .ForceFieldRelaxMaker or None
        Maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker: .ForceFieldStaticMaker or None
        Maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    phonon_displacement_maker: .ForceFieldStaticMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Arg that determines if thermal_displacement_matrices are computed
    kpath_scheme: str
        Scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str
        Determines the DFT code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        If True, force constants will be stored
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    """

    name: str = "ml phonon"
    min_length: float | None = 20.0
    displacement: float = 0.01
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: ForceFieldRelaxMaker(
            relax_cell=True,
            relax_kwargs={"interval": 500},
            force_field_name="GAP",
        )
    )
    phonon_displacement_maker: ForceFieldStaticMaker | None = field(
        default_factory=lambda: ForceFieldStaticMaker(
            name="gap phonon static",
            force_field_name="GAP",
        )
    )
    static_energy_maker: ForceFieldStaticMaker | None = None
    store_force_constants: bool = False
    get_supercell_size_kwargs: dict = field(
        default_factory=lambda: {"max_atoms": 20000, "step_size": 0.1}
    )
    generate_frequencies_eigenvectors_kwargs: dict = field(
        default_factory=lambda: {"units": "THz", "tol_imaginary_modes": 1e-1}
    )
    relax_maker_kwargs: dict | None = field(default_factory=dict)
    static_maker_kwargs: dict | None = field(default_factory=dict)

    @job
    def make_from_ml_model(
        self,
        structure,
        potential_file,
        ml_model: str = "GAP",
        calculator_kwargs: dict | None = None,
        supercell_settings: dict | None = None,
        **make_kwargs,
    ):
        """
        Maker for GAP phonon jobs.

        Parameters
        ----------
        structure : .Structure
            The pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        ml_model: str
            ML model to be used. Default is GAP.
        potential_file :
            Complete path to MLIP file(s) (train, test and MLIP files)
        calculator_kwargs :
            Keyword arguments for the ASE Calculator.
        supercell_settings:
            Dict with supercell settings.
        make_kwargs :
            Keyword arguments for the PhononMaker.

        Returns
        -------
        PhononMaker jobs.

        """
        if supercell_settings is None:
            supercell_settings = field(default_factory=lambda: {"min_length": 15})
        if ml_model == "GAP":
            if calculator_kwargs is None:
                calculator_kwargs = {
                    "args_str": "IP GAP",
                    "param_filename": str(potential_file),
                }

            ml_prep = ml_phonon_maker_preparation(
                bulk_relax_maker=self.bulk_relax_maker,
                phonon_displacement_maker=self.phonon_displacement_maker,
                static_energy_maker=self.static_energy_maker,
                calculator_kwargs=calculator_kwargs,
                relax_maker_kwargs=self.relax_maker_kwargs,
                static_maker_kwargs=self.static_maker_kwargs,
            )

        elif ml_model == "J-ACE":
            raise UserWarning("No atomate2 ACE.jl PhononMaker implemented.")

        elif ml_model == "NEP":
            if calculator_kwargs is None:
                calculator_kwargs = {"model_filename": str(potential_file)}

            ml_prep = ml_phonon_maker_preparation(
                bulk_relax_maker=ForceFieldRelaxMaker(
                    relax_cell=True,
                    relax_kwargs={"interval": 500},
                    force_field_name="NEP",
                ),
                phonon_displacement_maker=ForceFieldStaticMaker(
                    name="nep phonon static",
                    force_field_name="NEP",
                ),
                static_energy_maker=None,
                calculator_kwargs=calculator_kwargs,
                relax_maker_kwargs=self.relax_maker_kwargs,
                static_maker_kwargs=self.static_maker_kwargs,
            )
        elif ml_model == "NEQUIP":
            if calculator_kwargs is None:
                calculator_kwargs = {
                    "model_path": str(potential_file),
                    "device": "cuda",
                }
            else:
                calculator_kwargs.update({"model_path": str(potential_file)})

            ml_prep = ml_phonon_maker_preparation(
                bulk_relax_maker=ForceFieldRelaxMaker(
                    relax_cell=True,
                    relax_kwargs={"interval": 500},
                    force_field_name="Nequip",
                ),
                phonon_displacement_maker=ForceFieldStaticMaker(
                    name="nequip phonon static",
                    force_field_name="Nequip",
                ),
                static_energy_maker=None,
                calculator_kwargs=calculator_kwargs,
                relax_maker_kwargs=self.relax_maker_kwargs,
                static_maker_kwargs=self.static_maker_kwargs,
            )

        elif ml_model == "M3GNET":
            if calculator_kwargs is None:
                calculator_kwargs = {"path": str(potential_file)}

            ml_prep = ml_phonon_maker_preparation(
                bulk_relax_maker=ForceFieldRelaxMaker(
                    relax_cell=True,
                    relax_kwargs={"interval": 500},
                    force_field_name="M3GNet",
                ),
                phonon_displacement_maker=ForceFieldStaticMaker(
                    name="m3gnet phonon static",
                    force_field_name="M3GNet",
                ),
                static_energy_maker=None,
                calculator_kwargs=calculator_kwargs,
                relax_maker_kwargs=self.relax_maker_kwargs,
                static_maker_kwargs=self.static_maker_kwargs,
            )

        else:  # MACE
            if calculator_kwargs is None:
                calculator_kwargs = {"model": str(potential_file), "device": "cuda"}
            elif "model" in calculator_kwargs:
                calculator_kwargs.update(
                    {"default_dtype": "float64"}
                )  # Use float64 for geometry optimization.
            else:
                calculator_kwargs.update(
                    {"model": str(potential_file), "default_dtype": "float64"}
                )

            ml_prep = ml_phonon_maker_preparation(
                bulk_relax_maker=ForceFieldRelaxMaker(
                    relax_cell=True,
                    relax_kwargs={"interval": 500},
                    force_field_name="MACE",
                ),
                phonon_displacement_maker=ForceFieldStaticMaker(
                    name="mace phonon static",
                    force_field_name="MACE",
                ),
                static_energy_maker=None,
                calculator_kwargs=calculator_kwargs,
                relax_maker_kwargs=self.relax_maker_kwargs,
                static_maker_kwargs=self.static_maker_kwargs,
            )

        (
            self.bulk_relax_maker,
            self.phonon_displacement_maker,
            self.static_energy_maker,
        ) = ml_prep
        filtered_settings = {
            key: value
            for key, value in supercell_settings.items()
            if key
            in [
                "min_length",
                "max_length",
                "fallback_min_length",
                "max_atoms",
                "min_atoms",
                "step_size",
            ]
        }
        supercell_matrix = reduce_supercell_size(
            structure=structure, **filtered_settings
        )

        flow = self.make(
            structure=structure, supercell_matrix=supercell_matrix, **make_kwargs
        )
        return Response(replace=flow, output=flow.output)


@dataclass
class IsoAtomStaticMaker(StaticMaker):
    """
    Maker to create Isolated atoms static (VASP) jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        Generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 1},
            user_incar_settings={
                "ALGO": "Normal",
                "ISPIN": 1,
                "LAECHG": False,
                "ISMEAR": 0,
                "LCHARG": False,  # Do not write the CHGCAR file
                "LWAVE": False,  # Do not write the WAVECAR file
                "LVTOT": False,  # Do not write LOCPOT file
                "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                "LOPTICS": False,  # No PCDAT file
                # to be removed
                "NPAR": 4,
            },
        )
    )


@dataclass
class IsoAtomMaker(Maker):
    """
    Maker to generate DFT data for ML potential fitting from isolated atoms.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "IsolatedAtomEnergyMaker"

    def make(
        self,
        all_species: list[Species],
        isolated_atom_maker: IsoAtomStaticMaker = None,
    ):
        """
        Make a flow to calculate the isolated atom's energy.

        Parameters
        ----------
        all_species : list[Species]
            List of pymatgen specie object.
        isolated_atom_maker: IsoAtomMaker
            VASP input set for the isolated atom calculation.
        """
        jobs = []
        isoatoms_energy = []
        isoatoms_dirs = []
        if isolated_atom_maker is None:
            isolated_atom_static_input_set = StaticSetGenerator(
                user_kpoints_settings={"grid_density": 1},
                user_incar_settings={
                    "ALGO": "Normal",
                    "ISPIN": 1,
                    "LAECHG": False,
                    "ISMEAR": 0,
                    "LCHARG": False,  # Do not write the CHGCAR file
                    "LWAVE": False,  # Do not write the WAVECAR file
                    "LVTOT": False,  # Do not write LOCPOT file
                    "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                    "LOPTICS": False,  # No PCDAT file
                    # to be removed
                    "NPAR": 4,
                    # TODO: locpot, chgcar, chg can be deactivated!
                    # TODO: why don't we use the IsoAtomMaker and adapt it?
                },
            )
            isolated_atom_maker = StaticMaker(
                input_set_generator=isolated_atom_static_input_set, name="stat_iso_atom"
            )
        for species in all_species:
            site = Site(species=species, coords=[0, 0, 0])
            mol = Molecule.from_sites([site])
            iso_atom = mol.get_boxed_structure(a=20, b=20, c=20)
            isolated_atom_maker.name = f"{species}-stat_iso_atom"
            isolated_atom_maker.run_vasp_kwargs = {"handlers": ()}
            isoatom_calcs = isolated_atom_maker.make(iso_atom)

            jobs.append(isoatom_calcs)
            isoatoms_energy.append(isoatom_calcs.output.output.energy_per_atom)
            isoatoms_dirs.append(isoatom_calcs.output.dir_name)

        # create a flow including all jobs
        return Flow(
            jobs=jobs,
            output={"energies": isoatoms_energy, "dirs": isoatoms_dirs},
            name=self.name,
        )
