"""Settings for autoplex."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

from monty.serialization import loadfn
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from autoplex._basemodel import AutoplexBaseModel
from autoplex.fitting.mlip_hypers import MLIPHypers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

__all__ = [
    "AutoplexSettings",
    "MLIPHypers",
    "RssConfig",
]

_DEFAULT_CONFIG_FILE_PATH = "~/.autoplex.yaml"
_ENV_PREFIX = "autoplex_"


class AutoplexSettings(BaseSettings):
    """Model describing the autoplex-related commands.

    The following code has been taken and modified from
    https://github.com/materialsproject/atomate2/blob/main/src/atomate2/settings.py
    The code has been released under BSD 3-Clause License
    and the following copyright applies:
    atomate2 Copyright (c) 2015, The Regents of the University of
    California, through Lawrence Berkeley National Laboratory (subject
    to receipt of any required approvals from the U.S. Dept. of Energy).
    All rights reserved.
    """

    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from."
    )
    CASTEP_CMD: str = Field(default="castep", description="command to run castep.")
    CASTEP_KEYWORDS: Path = Field(
        default=Path(__file__).parent / "misc" / "castep" / "castep_keywords.json"
    )

    model_config = SettingsConfigDict(env_prefix=_ENV_PREFIX)

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        """
        config_file_path = values.get(key := "CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)
        env_var_name = f"{_ENV_PREFIX.upper()}{key}"
        config_file_path = Path(config_file_path).expanduser()

        new_values = {}
        if config_file_path.exists():
            if config_file_path.stat().st_size == 0:
                warnings.warn(
                    f"Using {env_var_name} at {config_file_path} but it's empty",
                    stacklevel=2,
                )
            else:
                try:
                    new_values.update(loadfn(config_file_path))
                except ValueError:
                    raise SyntaxError(
                        f"{env_var_name} at {config_file_path} is unparsable"
                    ) from None
        # warn if config path is not the default but file doesn't exist
        elif config_file_path != Path(_DEFAULT_CONFIG_FILE_PATH).expanduser():
            warnings.warn(
                f"{env_var_name} at {config_file_path} does not exist", stacklevel=2
            )

        return new_values | values


# RSS Configuration


class ResumeFromPreviousState(AutoplexBaseModel):
    """
    A model describing the state information.

    Useful to resume a previously interrupted or saved RSS workflow.
    When 'train_from_scratch' is set to False, this parameter is mandatory
    for the workflow to pick up from a saved state.
    """

    test_error: float | None = Field(
        default=None,
        description="The test error from the last completed training step.",
    )
    pre_database_dir: str | None = Field(
        default=None,
        description="Path to the directory containing the pre-existing database for resuming",
    )
    mlip_path: str | None = Field(
        default=None, description="Path to the file of a previous MLIP model."
    )
    isolated_atom_energies: dict | None = Field(
        default=None,
        description="A dictionary with isolated atom energy values mapped to atomic numbers",
    )


class SoapParas(AutoplexBaseModel):
    """A model describing the SOAP parameters."""

    l_max: int = Field(default=12, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.0875, description="idth of Gaussian smearing")
    cutoff: float = Field(default=10.5, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    zeta: float = Field(default=4.0, description="Exponent for dot-product SOAP kernel")
    average: bool = Field(
        default=True, description="Whether to average the SOAP vectors"
    )
    species: bool = Field(
        default=True, description="Whether to consider species information"
    )


class BcurParams(AutoplexBaseModel):
    """A model describing the parameters for the BCUR method."""

    soap_paras: SoapParas = Field(default_factory=SoapParas)
    frac_of_bcur: float = Field(
        default=0.8, description="Fraction of Boltzmann CUR selections"
    )
    bolt_max_num: int = Field(
        default=3000, description="Maximum number of Boltzmann selections"
    )


class BuildcellOptions(AutoplexBaseModel):
    """A model describing the parameters for buildcell."""

    ABFIX: bool = Field(default=False, description="Whether to fix the lattice vectors")
    NFORM: str | None = Field(default=None, description="The number of formula units")
    SYMMOPS: str | None = Field(
        default=None,
        description="	Build structures having a specified "
        "number of symmetry operations. For crystals, "
        "the allowed values are (1,2,3,4,6,8,12,16,24,48). "
        "For clusters (indicated with #CLUSTER), the allowed "
        "values are (1,2,3,5,4,6,7,8,9,10,11,12,24). "
        "Ranges are allowed (e.g., #SYMMOPS=1-4).",
    )
    SYSTEM: (
        None
        | Literal["Rhom", "Tric", "Mono", "Cubi", "Hexa", "Orth", "Tetra"]
        | set[Literal["Rhom", "Tric", "Mono", "Cubi", "Hexa", "Orth", "Tetra"]]
    ) = Field(default=None, description="Enforce a crystal system")
    SLACK: float | None = Field(default=None, description="The slack factor")
    OCTET: bool = Field(
        default=False,
        description="Check number of valence electrons is a multiple of eight",
    )
    OVERLAP: float | None = Field(default=None, description="The overlap factor")
    MINSEP: str | None = Field(default=None, description="The minimum separation")


class CustomIncar(AutoplexBaseModel):
    """A model describing the INCAR parameters."""

    ISMEAR: int = 0
    SIGMA: float = 0.05
    PREC: str = "Accurate"
    ADDGRID: str = ".TRUE."
    EDIFF: float = 1e-07
    NELM: int = 250
    LWAVE: str = ".FALSE."
    LCHARG: str = ".FALSE."
    ALGO: str = "Normal"
    AMIX: float | None = None
    LREAL: str = ".FALSE."
    ISYM: int = 0
    ENCUT: float = 520.0
    KSPACING: float = 0.2
    GGA: str | None = None
    KPAR: int = 8
    NCORE: int = 16
    LSCALAPACK: str = ".FALSE."
    LPLANE: str = ".FALSE."


class RssConfig(AutoplexBaseModel):
    """A model describing the complete RSS configuration."""

    tag: str | None = Field(
        default=None,
        description="Tag of systems. It can also be used for setting up elements "
        "and stoichiometry. For example, the tag of 'SiO2' will be recognized "
        "as a 1:2 ratio of Si to O and passed into the parameters of buildcell. "
        "However, note that this will be overwritten if the stoichiometric ratio "
        "of elements is defined in the 'cell_seed_paths' or 'buildcell_options'",
    )
    train_from_scratch: bool = Field(
        default=True,
        description="If True, it starts the workflow from scratch "
        "If False, it resumes from a previous state.",
    )
    resume_from_previous_state: ResumeFromPreviousState = Field(
        default_factory=ResumeFromPreviousState
    )
    generated_struct_numbers: list[int, int] = Field(
        default_factory=lambda: [10000],
        description="Expected number of generated "
        "randomized unit cells by buildcell.",
    )
    cell_seed_paths: list[str] | None = Field(
        default=None, description="Custom buildcell control files."
    )
    buildcell_options: list[BuildcellOptions] | None = Field(
        default=None, description="Customized parameters for buildcell."
    )
    fragment_file: str | None = Field(default=None, description="")
    fragment_numbers: list[int] | None = Field(
        default=None,
        description="Numbers of each fragment to be included in the random structures. "
        "Defaults to 1 for all specified.",
    )
    num_processes_buildcell: int = Field(
        default=128, description="Number of processes for buildcell."
    )
    num_of_initial_selected_structs: list[int, int] = Field(
        default_factory=lambda: [100],
        description="Number of structures to be sampled directly "
        "from the buildcell-generated randomized cells.",
    )
    num_of_rss_selected_structs: int = Field(
        default=100,
        description="Number of structures to be selected from each RSS iteration.",
    )
    initial_selection_enabled: bool = Field(
        default=True,
        description="If true, sample structures from initially generated "
        "randomized cells using CUR.",
    )
    rss_selection_method: Literal["bcur1s", "bcur2i"] | None = Field(
        default="bcur2i",
        description="Method for selecting samples from the RSS trajectories: "
        "Boltzmann flat histogram in enthalpy first, then CUR. Options are as follows",
    )
    bcur_params: BcurParams = Field(
        default_factory=BcurParams, description="Parameters for the BCUR method."
    )
    random_seed: int | None = Field(
        default=None, description="A seed to ensure reproducibility of CUR selection."
    )
    include_isolated_atom: bool = Field(
        default=True,
        description="Perform single-point calculations for isolated atoms.",
    )
    isolatedatom_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="List of the lattice constants for an "
        "isolated atom configuration.",
    )
    e0_spin: bool = Field(
        default=False,
        description="Include spin polarization in isolated atom and dimer calculations",
    )
    include_dimer: bool = Field(
        default=True,
        description="Perform single-point calculations for dimers only once",
    )
    dimer_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="The lattice constants of a dimer box.",
    )
    dimer_range: list[float, float] = Field(
        default_factory=lambda: [1.0, 5.0],
        description="The range of the dimer distance.",
    )
    dimer_num: int = Field(
        default=21,
        description="Number of different distances to consider for dimer calculations.",
    )
    custom_incar: CustomIncar | None = Field(
        default_factory=CustomIncar,
        description="Custom VASP input parameters. "
        "If provided, will update the default parameters",
    )
    custom_potcar: str | None = Field(
        default=None,
        description="POTCAR settings to update. Keys are element symbols, "
        "values are the desired POTCAR labels.",
    )
    dft_ref_file: str = Field(
        default="dft_ref.extxyz", description="Reference file for VASP data"
    )
    config_types: list[str] = Field(
        default_factory=lambda: ["initial", "traj_early", "traj"],
        description="Configuration types for the VASP calculations",
    )
    rss_group: list[str] | str = Field(
        default_factory=lambda: ["traj"],
        description="Group of configurations for the RSS calculations",
    )
    test_ratio: float = Field(
        default=0.1,
        description="The proportion of the test set after splitting the data",
    )
    disable_testing: bool = Field(
        default=False,
        description="Whether to disable running the model on test set for the run.",
    )
    regularization: bool = Field(
        default=True,
        description="Whether to apply regularization. This only works for GAP to date.",
    )
    retain_existing_sigma: bool = Field(
        default=False,
        description="Whether to retain the existing sigma values for specific configuration types."
        "If True, existing sigma values for specific configurations will remain unchanged",
    )
    scheme: Literal["linear-hull", "volume-stoichiometry"] | None = Field(
        default="linear-hull", description="Method to use for regularization"
    )
    reg_minmax: list[list[float]] = Field(
        default_factory=lambda: [
            [0.1, 1],
            [0.001, 0.1],
            [0.0316, 0.316],
            [0.0632, 0.632],
        ],
        description="List of tuples of (min, max) values for energy, force, "
        "virial sigmas for regularization",
    )
    distillation: bool = Field(
        default=False, description="Whether to apply data distillation"
    )
    force_max: float | None = Field(
        default=None, description="Maximum force value to exclude structures"
    )
    force_label: str | None = Field(
        default=None, description="The label of force values to use for distillation"
    )
    pre_database_dir: str | None = Field(
        default=None, description="Directory where the previous database was saved."
    )
    mlip_type: Literal["GAP", "J-ACE", "P-ACE", "NEQUIP", "M3GNET", "MACE"] = Field(
        default="GAP", description="MLIP to be fitted"
    )
    ref_energy_name: str = Field(
        default="REF_energy", description="Reference energy name."
    )
    ref_force_name: str = Field(
        default="REF_forces", description="Reference force name."
    )
    ref_virial_name: str = Field(
        default="REF_virial", description="Reference virial name."
    )
    auto_delta: bool = Field(
        default=True,
        description="Whether to automatically calculate the delta value for GAP terms.",
    )
    num_processes_fit: int = Field(
        default=32, description="Number of processes used for fitting"
    )
    device_for_fitting: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for model fitting"
    )
    scalar_pressure_method: Literal["exp", "uniform"] = Field(
        default="uniform", description="Method for adding external pressures."
    )
    scalar_exp_pressure: int = Field(
        default=1, description="Scalar exponential pressure"
    )
    scalar_pressure_exponential_width: float = Field(
        default=0.2, description="Width for scalar pressure exponential"
    )
    scalar_pressure_low: int = Field(
        default=0, description="Lower limit for scalar pressure"
    )
    scalar_pressure_high: int = Field(
        default=25, description="Upper limit for scalar pressure"
    )
    max_steps: int = Field(
        default=300, description="Maximum number of steps for the GAP optimization"
    )
    force_tol: float = Field(
        default=0.01, description="Force residual tolerance for relaxation"
    )
    stress_tol: float = Field(
        default=0.01, description="Stress residual tolerance for relaxation."
    )
    stop_criterion: float = Field(
        default=0.01, description="Convergence criterion for stopping RSS iterations."
    )
    max_iteration_number: int = Field(
        default=25, description="Maximum number of RSS iterations to perform."
    )
    num_groups: int = Field(
        default=6,
        description="Number of structure groups, used for assigning tasks across multiple nodes."
        "For example, if there are 10,000 trajectories to relax and 'num_groups=10',"
        "the trajectories will be divided into 10 groups and 10 independent jobs will be created,"
        "with each job handling 1,000 trajectories.",
    )
    initial_kt: float = Field(
        default=0.3, description="Initial temperature (in eV) for Boltzmann sampling."
    )
    current_iter_index: int = Field(
        default=1, description="Current iteration index for the RSS."
    )
    hookean_repul: bool = Field(
        default=False, description="Whether to apply Hookean repulsion"
    )
    hookean_paras: dict | None = Field(
        default=None,
        description="Parameters for the Hookean repulsion as a "
        "dictionary of tuples.",
    )
    keep_symmetry: bool = Field(
        default=False, description="Whether to preserve symmetry during relaxations."
    )
    remove_traj_files: bool = Field(
        default=False,
        description="Bool indicating whether to remove the RSS trajectory files.",
    )
    num_processes_rss: int = Field(
        default=128, description="Number of processes used for running RSS."
    )
    device_for_rss: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for RSS calculations."
    )
    mlip_hypers: MLIPHypers = Field(
        default_factory=MLIPHypers, description="MLIP hyperparameters"
    )

    @classmethod
    def from_file(cls, filename: str):
        """Create RSS configuration object from a file."""
        config_params = loadfn(filename)

        # check if config file has the required keys when train_from_scratch is False
        train_from_scratch = config_params.get("train_from_scratch")
        resume_from_previous_state = config_params.get("resume_from_previous_state")

        if not train_from_scratch:
            for key, value in resume_from_previous_state.items():
                if value is None:
                    raise ValueError(
                        f"Value for {key} in `resume_from_previous_state` cannot be None when "
                        f"`train_from_scratch` is set to False"
                    )

        # check if mlip arg is in the config file
        # Needed for backward compatibility with older config files of RSS workflow
        mlip_type = config_params["mlip_type"].replace("-", "_")
        mlip_hypers = MLIPHypers().__getattribute__(mlip_type)

        if "mlip_hypers" not in config_params:
            config_params["mlip_hypers"] = {config_params["mlip_type"]: {}}

        old_config_keys = []
        for arg in config_params:
            mlip_type = config_params["mlip_type"].replace("-", "_")
            if arg in mlip_hypers.model_fields:
                config_params["mlip_hypers"][mlip_type].update(
                    {arg: config_params[arg]}
                )
                old_config_keys.append(arg)

        for key in old_config_keys:
            del config_params[key]

        return cls(**config_params)


SETTINGS = AutoplexSettings()
