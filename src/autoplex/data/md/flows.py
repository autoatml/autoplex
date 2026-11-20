"""Flows for running MD."""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from atomate2.forcefields.md import ForceFieldMDMaker
from emmet.core.math import Matrix3D
from jobflow import Flow, Maker, Response, job
from pymatgen.core.structure import Structure

from autoplex.data.common.utils import scale_cell
from autoplex.data.md.jobs import collect_md_trajs
from autoplex.data.md.utils import (
    generate_temperature_profile,
    plot_temperature_profile,
)
from autoplex.fitting.common.utils import extract_gap_label

logger = logging.getLogger(__name__)

# calculator_kwargs key of the model file of locally trained potentials;
# foundation models (e.g. MLFF.MACE_MPA_0) are downloaded and need no file
_MODEL_FILE_KEYS = {
    "MLFF.GAP": "param_filename",
    "MLFF.MACE": "model",
    "MLFF.Nequip": "model_path",
    "MLFF.NEP": "model_filename",
}


@dataclass
class ScaledMDMaker(Maker):
    """
    Maker to run the same MD on many cells, with an optional temperature profile.

    The MD itself is run by the MD maker of the chosen engine. With
    ``md_engine="ase"`` this is atomate2's `ForceFieldMDMaker` (ASE MD with a
    machine-learned force field), see
    https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/md.py
    Other engines are not implemented yet.

    Before the MD, every input structure can be turned into a supercell and into
    volume-scaled copies; one MD is run per resulting cell. The MD can follow any
    combination of quenching (cooling) and heating stages, such as:

    - isothermal equilibration
    - multi-stage quench (e.g., 3000 -> 1500 -> 300 K)
    - annealing or cyclic heating-cooling loops

    Parameters
    ----------
    name : str
        Job name.
    md_engine : str
        MD engine. Only "ase" (atomate2's `ForceFieldMDMaker`) is implemented.
    md_maker : ForceFieldMDMaker
        atomate2 MD maker with all MD settings (force field, ensemble, dynamics,
        time step, ...). It is copied for every run and never modified.
        If its `traj_file` is None, "MD.traj" is used.
    temperature_list : list[float]
        Stage temperatures of a multi-stage quench or anneal profile.
        If empty, `md_maker`'s own `temperature` and `n_steps` are used unchanged.
    eqm_step_list : list[int] | None, optional
        Number of MD steps to hold each temperature (one entry per temperature).
        If None, every stage is held for `md_maker.n_steps` steps.
    rate_list : list[float] | None, optional
        MD steps per kelvin for each ramp between stages (one entry per ramp, i.e.
        len(temperature_list) - 1). A rate of 0 jumps to the next temperature.
        None, [], or only zeros run without any ramps.
        With a 1 fs time step, rate=10 -> 10^14 K/s and rate=100 -> 10^13 K/s.
    ramp_step_list : list[int] | None, optional
        MD steps for each ramp (one entry per ramp), like in LAMMPS; 0 jumps to the
        next temperature. Alternative to rate_list, used instead of it if both are given.
    volume_custom_scale_factors : list[float] | None
        Volume scale factors applied to every structure, e.g. [0.95, 1.05]
        gives two MD runs per structure (-5 % and +5 % volume).
    supercell_matrix : Matrix3D | None
        Matrix for obtaining the supercell,
        e.g. ``[[2, 0, 0], [0, 2, 0], [0, 0, 2]]`` for a 2x2x2 supercell.
    """

    name: str = "ML-driven_MD"
    md_engine: str = "ase"
    md_maker: ForceFieldMDMaker | None = None
    temperature_list: list[float] = field(default_factory=list)
    eqm_step_list: list[int] | None = None
    rate_list: list[float] | None = None
    ramp_step_list: list[int] | None = None
    volume_custom_scale_factors: list[float] | None = None
    supercell_matrix: Matrix3D | None = None

    def __post_init__(self) -> None:
        """Check the engine and its MD maker as soon as the maker is created."""
        if self.md_engine != "ase":
            raise NotImplementedError(
                f"md_engine={self.md_engine!r} is not yet implemented; only 'ase' is available."
            )
        if not isinstance(self.md_maker, ForceFieldMDMaker):
            raise TypeError(
                "md_engine='ase' needs md_maker=ForceFieldMDMaker(...), e.g. "
                "ForceFieldMDMaker(force_field_name='MLFF.MACE_MPA_0', ensemble='nvt'); "
                f"got {type(self.md_maker).__name__}."
            )

    @job
    def make(self, structure: Structure | list[Structure]):
        """Run one MD per structure (after supercell and volume scaling).

        Parameters
        ----------
        structure : Structure | list[Structure]
            Pymatgen structure or a list of structures.

        Returns
        -------
        Response
            Replaces this job by the MD jobs. Its output is the list of
            trajectory file paths, one per MD run.
        """
        # Always work on a list, so supercells and volume scaling apply to every input
        structures = (
            [structure] if isinstance(structure, Structure) else list(structure)
        )
        if self.supercell_matrix is not None:
            logger.info(f"Applying supercell_matrix:\n{self.supercell_matrix}")
            structures = [s * self.supercell_matrix for s in structures]
            logger.info(f"Atoms per supercell: {[len(s) for s in structures]}")
            n_max = max(len(s) for s in structures)
            if n_max > 1000:
                raise ValueError(
                    f"The supercells have up to {n_max} atoms, but at most 1000 are allowed "
                    "to keep later DFT runs efficient. Use a smaller supercell_matrix."
                )
            if n_max >= 500:
                logger.warning(f"Some supercells have {n_max} atoms (500 or more).")

        if self.volume_custom_scale_factors is not None:
            logger.info("Applying custom volume scaling:")
            logger.info(f"  Scale factors: {self.volume_custom_scale_factors}")
            structures = [
                scaled
                for s in structures
                for scaled in scale_cell(
                    s, volume_custom_scale_factors=self.volume_custom_scale_factors
                )
            ]
            logger.info(f"  Number of scaled structures: {len(structures)}")

        # Work on a copy, so the user's md_maker is never modified
        md_maker = copy.deepcopy(self.md_maker)
        if md_maker.traj_file is None:  # atomate2's default writes no trajectory
            md_maker.traj_file = "MD.traj"

        # Build the multi-stage profile only if one was requested; otherwise keep
        # md_maker's own `temperature` and `n_steps`
        if self.temperature_list:
            md_maker.temperature, n_steps = generate_temperature_profile(
                temperature_list=self.temperature_list,
                eqm_step_list=self.eqm_step_list
                or [md_maker.n_steps] * len(self.temperature_list),
                rate_list=self.rate_list,
                ramp_step_list=self.ramp_step_list,
            )
            md_maker.n_steps = int(n_steps)
            plot_temperature_profile(md_maker.temperature, md_maker.time_step or 1.0)

        _prepare_calculator(md_maker)

        # own copy per job: atomate2 caches the calculator on the maker
        md_jobs = [copy.deepcopy(md_maker).make(structure=s) for s in structures]
        collec_trajs = collect_md_trajs(
            {f"md_job_{idx}": md_job.output for idx, md_job in enumerate(md_jobs)},
            traj_file=md_maker.traj_file,
        )

        return Response(
            replace=Flow([*md_jobs, collec_trajs]), output=collec_trajs.output
        )


def _prepare_calculator(md_maker: ForceFieldMDMaker) -> None:
    """Choose a device and check the model file of locally trained potentials.

    Modifies ``md_maker.calculator_kwargs`` in place, so call it on a copy.
    """
    calculator_kwargs = md_maker.calculator_kwargs
    if calculator_kwargs.get("device") is None:
        calculator_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified. Using {calculator_kwargs['device']}.")

    force_field_name = md_maker.force_field_name
    model_key = _MODEL_FILE_KEYS.get(force_field_name)
    if model_key is not None:
        model_file = calculator_kwargs.get(model_key)
        if model_file is None or not Path(model_file).exists():
            raise FileNotFoundError(
                f"{force_field_name} needs an existing model file in "
                f"calculator_kwargs['{model_key}'], got {model_file!r}. "
                "For a foundation model use its full name, e.g. MLFF.MACE_MPA_0."
            )
        logger.info(f"Using {force_field_name} model {model_file}")

    if force_field_name == "MLFF.GAP":
        gap_file = Path(calculator_kwargs["param_filename"])
        calculator_kwargs["args_str"] = (
            f"Potential xml_label={extract_gap_label(gap_file)}"
        )
    elif force_field_name == "MLFF.NEP":
        calculator_kwargs.pop("device", None)  # calorine takes no device
