"""RSS workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from jobflow import Flow, Maker, Response, job
from ruamel.yaml import YAML

from autoplex.auto.rss.jobs import do_rss_iterations, initial_rss

current_dir = Path(__file__).absolute().parent
RSS_CONFIGURATION_DEFAULTS_FILE_PATH = current_dir / "rss_default_configuration.yaml"


@dataclass
class RssMaker(Maker):
    """
    Maker to set up and run RSS for exploring and learning potential-energy surfaces (from scratch).

    Parameters
    ----------
    name: str
        Name of the flow.
    path_to_default_config_parameters: Path | str
        Path to the default RSS configuration file 'rss_default_configuration.yaml'
    """

    name: str = "ml-driven rss"
    path_to_default_config_parameters: Path | str = RSS_CONFIGURATION_DEFAULTS_FILE_PATH

    @job
    def make(self, config_file: str | None = None, **kwargs):
        """
        Make a rss workflow using the specified configuration file and additional keyword arguments.

        Parameters
        ----------
        config_file: str | None
            Path to the configuration file that defines the setup parameters for the whole RSS workflow.
            If not provided, the default file 'rss_default_configuration.yaml' will be used.
        kwargs: dict, optional
            Additional optional keyword arguments to customize the job execution.

        Keyword Arguments
        -----------------
        - tag: str
            Tag of systems. It can also be used for setting up elements and stoichiometry.
            For example, the tag of 'SiO2' will be recognized as a 1:2 ratio of Si to O and
            passed into the parameters of buildcell. However, note that this will be overwritten
            if the stoichiometric ratio of elements is defined in the 'buildcell_options'.
        - train_from_scratch: bool
            If True, it starts the workflow from scratch.
            If False, it resumes from a previous state.
        - resume_from_previous_state: dict | None
            A dictionary containing the state information required to resume a previously interrupted
            or saved RSS workflow. When 'train_from_scratch' is set to False, this parameter is mandatory
            for the workflow to pick up from a saved state.
            Expected keys within this dictionary:
            - test_error: float
                The test error from the last completed training step.
            - pre_database_dir: str
                Path to the directory containing the pre-existing database for resuming.
            - mlip_path: str
                Path to the file of a previous MLIP model.
            - isolated_atom_energies: dict
                A dictionary of isolated atom energy values, with atomic numbers as keys
                and their energies as valuables.
        - generated_struct_numbers: list[int]
            Expected number of generated randomized unit cells by buildcell.
        - buildcell_options: list[dict] | None
            Customized parameters for buildcell. Default is None.
        - fragment: Atoms | list[Atoms] | None
            Fragment(s) for random structures, e.g., molecules, to be placed indivudally intact.
            atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
            atoms.cell must be defined (e.g., Atoms.cell = np.eye(3)*20).
        - fragment_numbers: list[str] | None
            Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
        - num_processes_buildcell: int
            Number of processes to use for parallel computation during buildcell generation.
        - num_of_initial_selected_structs: list[int] | None
            Number of structures to be sampled directly from the buildcell-generated randomized cells.
        - num_of_rss_selected_structs: int
            Number of structures to be selected from each RSS iteration.
        - initial_selection_enabled: bool
            If true, sample structures from initially generated randomized cells using CUR.
        - rss_selection_method: str
            Method for selecting samples from the RSS trajectories:
            Boltzmann flat histogram in enthalpy first, then CUR.
            Options include:
            - 'bcur1s': Execute bcur with one shot (1s)
            - 'bcur2i': Execute bcur with two iterations (2i)
        - bcur_params: dict | None
            Parameters for Boltzmann CUR selection. The default dictionary includes:
            - soap_paras: dict
                SOAP descriptor parameters:
                - l_max: int
                    Maximum degree of spherical harmonics (default 12).
                - n_max: int
                    Maximum number of radial basis functions (default 12).
                - atom_sigma: float
                    Width of Gaussian smearing (default 0.0875).
                - cutoff: float
                    Radial cutoff distance (default 10.5).
                - cutoff_transition_width: float
                    Width of the transition region (default 1.0).
                - zeta: float
                    Exponent for dot-product SOAP kernel (default 4.0).
                - average: bool
                    Whether to average the SOAP vectors (default True).
                - species: bool
                    Whether to consider species information (default True).
            - kb_temp: float
                Temperature in eV for Boltzmann weighting (default 0.3).
            - frac_of_bcur: float
                Fraction of Boltzmann CUR selections (default 0.8).
            - bolt_max_num: int
                Maximum number of Boltzmann selections (default 3000).
            - kernel_exp: float
                Exponent for the kernel (default 4.0).
            - energy_label: str
                Label for the energy data (default 'energy').
        - random_seed: int | None
            A seed to ensure reproducibility of CUR selection. Default is None.
        - include_isolated_atom: bool
            If true, perform single-point calculations for isolated atoms.
        - isolatedatom_box: list[float]
            List of the lattice constants for an isolated atom configuration.
        - e0_spin: bool
            If true, include spin polarization in isolated atom and dimer calculations. Default is False.
        - include_dimer: bool
            If true, perform single-point calculations for dimers only once. Default is False.
        - dimer_box: list[float]
            The lattice constants of a dimer box.
        - dimer_range: list[float]
            Range of distances for dimer calculations.
        - dimer_num: int
            Number of different distances to consider for dimer calculations. Default is 21.
        - custom_incar: dict | None
            Dictionary of custom VASP input parameters. If provided, will update the
            default parameters. Default is None.
        - custom_potcar: dict | None
            Dictionary of POTCAR settings to update. Keys are element symbols, values are the desired POTCAR labels.
            Default is None.
        - vasp_ref_file: str
            Reference file for VASP data. Default is 'vasp_ref.extxyz'.
        - config_types: list[str]
            Configuration types for the VASP calculations. Default is None.
        - rss_group: list[str]
            Group name for RSS to setting up regularization.
        - test_ratio: float
            The proportion of the test set after splitting the data. The value is allowed to be set to 0;
            in this case, the testing error would not be meaningful anymore.
        - regularization: bool
            If True, apply regularization. This only works for GAP to date. Default is False.
        - scheme: str
            Method to use for regularization. Options are:
            - 'linear_hull': for single-composition system, use 2D convex hull (E, V)
            - 'volume-stoichiometry': for multi-composition system, use 3D convex hull of (E, V, mole-fraction)
        - reg_minmax: list[tuple]
            list of tuples of (min, max) values for energy, force, virial sigmas for regularization.
        - distillation: bool
            If true, apply data distillation. Default is True.
        - force_max: float | None
            Maximum force value to exclude structures. Default is 50.
        - force_label: str | None
            The label of force values to use for distillation. Default is 'REF_forces'.
        - pre_database_dir: str | None
            Directory where the previous database was saved.
        - mlip_type: str
            Choose one specific MLIP type to be fitted: 'GAP' | 'J-ACE' | 'NEQUIP' | 'M3GNET' | 'MACE'.
            Default is 'GAP'.
        - ref_energy_name: str
            Reference energy name. Default is 'REF_energy'.
        - ref_force_name: str
            Reference force name. Default is 'REF_forces'.
        - ref_virial_name: str
            Reference virial name. Default is 'REF_virial'.
        - auto_delta: bool
            If true, apply automatic determination of delta for GAP terms. Default is False.
        - num_processes_fit: int
            Number of processes used for fitting. Default is 1.
        - device_for_fitting: str
            Device to be used for model fitting, either "cpu" or "cuda".
        - **fit_kwargs:
            Additional keyword arguments for the MLIP fitting process.
        - scalar_pressure_method: str
            Method for adding external pressures.
            Acceptable options are:
            - 'exp': Applies pressure using an exponential distribution.
            - 'uniform': Applies pressure using a uniform distribution.
        - scalar_exp_pressure: float
            Scalar exponential pressure. Default is 100.
        - scalar_pressure_exponential_width: float
            Width for scalar pressure exponential. Default is 0.2.
        - scalar_pressure_low: float
            Low limit for scalar pressure. Default is 0.
        - scalar_pressure_high: float
            High limit for scalar pressure. Default is 50.
        - max_steps: int
            Maximum number of steps for relaxation. Default is 200.
        - force_tol: float
            Force residual tolerance for relaxation. Default is 0.05.
        - stress_tol: float
            Stress residual tolerance for relaxation. Default is 0.05.
        - hookean_repul: bool
            If true, apply Hookean repulsion. Default is False.
        - hookean_paras: dict[tuple[int, int], tuple[float, float]] | None
            Parameters for Hookean repulsion as a dictionary of tuples. Default is None.
        - keep_symmetry: bool
            If true, preserve symmetry during relaxation. Default is False.
        - write_traj: bool
            If true, write trajectory of RSS. Default is True.
        - num_processes_rss: int
            Number of processes used for running RSS. Default is 1.
        - device_for_rss: str
            Specify device to use "cuda" or "cpu" for running RSS. Default is "cpu".
        - stop_criterion: float
            Convergence criterion for stopping RSS iterations. Default is 0.01.
        - max_iteration_number: int
            Maximum number of RSS iterations to perform. Default is 25.
        - num_groups: int
            Number of structure groups, used for assigning tasks across multiple nodes.
            For example, if there are 10,000 trajectories to relax and 'num_groups=10',
            the trajectories will be divided into 10 groups and 10 independent jobs will be created,
            with each job handling 1,000 trajectories.
        - initial_kb_temp: float
            Initial temperature (in eV) for Boltzmann sampling. Default is 0.3.
        - current_iter_index: int
            Index for the current RSS iteration. Default is 1.

        Output
        ------
        dict
            A dictionary whose keys contains:
            - test_error: float
                The test error of the fitted MLIP.
            - pre_database_dir: str
                The directory of the latest RSS database.
            - mlip_path: str
                The path to the latest fitted MLIP.
            - isolated_atom_energies: dict
                The isolated energy values.
            - current_iter: int
                The current iteration index.
            - kb_temp: float
                The temperature (in eV) for Boltzmann sampling.
        """
        yaml = YAML(typ="safe", pure=True)

        with open(self.path_to_default_config_parameters) as f:
            config = yaml.load(f)

        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                new_config = yaml.load(f)
                config.update(new_config)

        config.update(kwargs)
        self._process_hookean_paras(config)

        config_params = config.copy()

        if "train_from_scratch" not in config_params:
            raise ValueError(
                "'train_from_scratch' must be set in the configuration file or passed as a keyword argument!!"
            )

        rss_flow = []

        if config_params["train_from_scratch"]:
            initial_exclude_keys = [
                "train_from_scratch",
                "resume_from_previous_state",
                "config_types",
                "rss_group",
                "num_of_rss_selected_structs",
                "rss_selection_method",
                "scalar_pressure_method",
                "scalar_exp_pressure",
                "scalar_pressure_exponential_width",
                "scalar_pressure_low",
                "scalar_pressure_high",
                "max_steps",
                "force_tol",
                "stress_tol",
                "stop_criterion",
                "max_iteration_number",
                "num_groups",
                "initial_kb_temp",
                "current_iter_index",
                "hookean_repul",
                "hookean_paras",
                "keep_symmetry",
                "write_traj",
                "num_processes_rss",
                "device_for_rss",
            ]
            initial_params = {
                k: v for k, v in config_params.items() if k not in initial_exclude_keys
            }
            initial_params.update(
                {
                    "config_type": config_params["config_types"][0],
                    "rss_group": config_params["rss_group"][0],
                }
            )
            initial_rss_job = initial_rss(**initial_params)
            rss_flow.append(initial_rss_job)

        rss_group = config_params["rss_group"]
        config_types = config_params["config_types"]
        do_rss_group = rss_group[0] if len(rss_group) == 1 else rss_group[-1]
        rss_config_type = (
            config_types[0] if len(config_types) == 1 else config_types[1:]
        )

        rss_exclude_keys = [
            "train_from_scratch",
            "resume_from_previous_state",
            "pre_database_dir",
        ]

        rss_params = {
            k: v for k, v in config_params.items() if k not in rss_exclude_keys
        }
        rss_params.update(
            {
                "num_of_initial_selected_structs": None,
                "initial_selection_enabled": False,
                "rss_group": do_rss_group,
                "config_types": rss_config_type,
            }
        )

        if config_params["train_from_scratch"]:
            do_rss_job = do_rss_iterations(
                input=initial_rss_job.output,
                **rss_params,
            )
        else:
            if "resume_from_previous_state" not in config_params:
                raise ValueError(
                    "The parameter 'resume_from_previous_state' must be specified when 'train_from_scratch' is False."
                )
            resume_from_previous_state = config_params["resume_from_previous_state"]
            do_rss_job = do_rss_iterations(
                input=resume_from_previous_state,
                **rss_params,
            )

        rss_flow.append(do_rss_job)

        return Response(replace=Flow(rss_flow), output=do_rss_job.output)

    def _process_hookean_paras(self, config):
        if "hookean_paras" in config:
            config["hookean_paras"] = {
                tuple(map(int, k.strip("()").split(", "))): tuple(v)
                for k, v in config["hookean_paras"].items()
            }
