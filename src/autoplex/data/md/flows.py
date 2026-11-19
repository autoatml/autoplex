"""Flows for running MD."""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from atomate2.forcefields.md import ForceFieldMDMaker
from emmet.core.math import Matrix3D
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure

from autoplex.data.common.utils import scale_cell
from autoplex.data.md.jobs import collect_md_trajs
from autoplex.data.md.utils import (
    generate_temperature_profile,
    plot_temperature_profile,
)
from autoplex.fitting.common.utils import extract_gap_label

logger = logging.getLogger(__name__)


@dataclass
class MDAseMaker(ForceFieldMDMaker):
    """
    Maker to perform molecular dynamics (MD) simulations using ASE.

    This class extends the `ForceFieldMDMaker` from atomate2, with all its
    standard keywords preserved. For reference, see:
    https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/md.py

    In addition, this MD maker supports running **any combination of quenching
    (cooling)** and **heating** stages.  By specifying multiple temperature points,
    the maker automatically builds a stepwise temperature path, enabling complex
    thermal protocols such as:
    - isothermal equilibration
    - multi-stage quench (e.g., 3000 -> 1500 -> 300 K)
    - annealing or cyclic heating-cooling loops

    This maker provides a flexible way to design and execute
    temperature-controlled MD workflows directly within atomate2.

    Parameters
    ----------
    temperature_list : list[float]
        Stage temperatures of a multi-stage quench or anneal profile.
        If empty, atomate2's own `temperature` and `n_steps` are used unchanged.
    eqm_step_list : list[int] | None, optional
        Number of MD steps to hold each temperature (one entry per temperature).
        If None, every stage is held for `n_steps` steps.
    rate_list : list[float] | None, optional
        MD steps per kelvin for each ramp between stages (one entry per ramp, i.e.
        len(temperature_list) - 1). A rate of 0 jumps to the next temperature.
        None, [], or only zeros run without any ramps.
        With a 1 fs time step, rate=10 -> 10^14 K/s and rate=100 -> 10^13 K/s.
    traj_file : str
        Name of the trajectory file written by every MD job (ASE format).
    name : str
        Job name.
    volume_custom_scale_factors : list[float] | None
        Volume scale factors applied to every structure, e.g. [0.95, 1.05]
        gives two MD runs per structure (-5 % and +5 % volume).
    supercell_matrix : Matrix3D | None
        Matrix for obtaining the supercell,
        e.g. ``[[2, 0, 0], [0, 2, 0], [0, 0, 2]]`` for a 2x2x2 supercell.
    """

    name: str = "ML-driven_MD_with_ASE"
    temperature_list: list[float] = field(default_factory=list)
    eqm_step_list: list[int] | None = None
    rate_list: list[float] | None = None
    volume_custom_scale_factors: list[float] | None = None
    supercell_matrix: Matrix3D | None = None
    traj_file: str = "MD.traj"  # atomate2 defaults to None, which writes no trajectory

    @job
    def make(self, structure: Structure):
        """Maker to run MD simulations.

        Parameters
        ----------
        structure: Structure
            Pymatgen structure.
        """
        logger.warning(
            "The value of `force_field_name` must use the full, case-sensitive name. "
            "Supported types include: CHGNet, M3GNet, MATPES_R2SCAN, MATPES_PBE, "
            "MACE, MACE_MP_0, MACE_MPA_0, MACE_MP_0B3, GAP, NEP, Nequip, SevenNet. "
        )

        # Always work on a list, so supercells and volume scaling apply to every input
        structures = (
            [structure] if isinstance(structure, Structure) else list(structure)
        )
        if self.supercell_matrix is not None:
            logger.info(f"Applying supercell_matrix:\n{self.supercell_matrix}")
            structures = [s * self.supercell_matrix for s in structures]
            logger.info(f"Supercell generated: {structures}.")
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

        if self.traj_file is None:
            raise ValueError("`traj_file` must be set, the trajectories are read later")

        # Build the multi-stage profile only if one was requested; otherwise keep
        # atomate2's own `temperature` and `n_steps`. The result goes into the
        # per-job copies below, so the user's maker is never modified.
        temperature, n_steps = self.temperature, self.n_steps
        if self.temperature_list:
            temperature, n_steps = generate_temperature_profile(
                temperature_list=self.temperature_list,
                eqm_step_list=self.eqm_step_list
                or [self.n_steps] * len(self.temperature_list),
                rate_list=self.rate_list,
            )
            plot_temperature_profile(temperature, self.time_step or 1.0)

        job_list = []
        job_output = {}
        device = self.calculator_kwargs.get("device", None)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("No device specified. Using CUDA (GPU).")
            else:
                device = "cpu"
                logger.info(
                    "No device specified. CUDA unavailable. Falling back to CPU."
                )

            self.calculator_kwargs["device"] = device

        if self.force_field_name == "MLFF.GAP":
            logger.info("Using GAP force field for MD simulation...")
            gap_label = self.calculator_kwargs.get("param_filename", None)
            if gap_label is None:
                raise ValueError(
                    "`param_filename` must be provided for GAP force field."
                )
            gap_path = Path(gap_label)
            if gap_path.exists():
                gap_control = "Potential xml_label=" + extract_gap_label(gap_path)
            else:
                logger.error(f"GAP parameter file not found: {gap_path}")
                raise FileNotFoundError(
                    f"GAP parameter file does not exist: {gap_path}"
                )
            self.calculator_kwargs["args_str"] = gap_control

        elif self.force_field_name == "MLFF.MACE":
            logger.info("Using MACE force field for MD simulation...")
            model_val = self.calculator_kwargs.get("model", None)
            if not Path(model_val).exists():
                raise ValueError(
                    "No local MACE model was specified in `calculator_kwargs['model']`.\n"
                    "Please provide the absolute path to your trained MACE model.\n\n"
                    "If you intend to use a MACE foundation model instead, please specify "
                    "a fully-qualified force_field_name, such as:\n"
                    "    'MLFF.MACE_MP_0'\n"
                    "instead of using the ambiguous name 'MACE'."
                )

        elif self.force_field_name == "MLFF.Nequip":
            logger.info("Using NequIP force field for MD simulation...")
            model_path = self.calculator_kwargs.get("model_path", None)
            if not Path(model_path).exists():
                raise ValueError(
                    "`model_path` must be provided for NequIP force field."
                )

        elif self.force_field_name == "MLFF.NEP":
            self.calculator_kwargs.pop("device", None)
            logger.info("Using NEP force field for MD simulation...")
            model_filename = self.calculator_kwargs.get("model_filename", None)
            if not Path(model_filename).exists():
                raise ValueError(
                    "`model_filename` must be provided for NEP force field."
                )

        for idx, struct in enumerate(structures):
            # own copy per job: atomate2 caches the calculator on the maker
            md_maker = copy.deepcopy(self)
            md_maker.temperature, md_maker.n_steps = temperature, int(n_steps)
            md_job = ForceFieldMDMaker.make(md_maker, structure=struct)
            job_list.append(md_job)
            job_output[f"md_job_{idx}"] = md_job.output

        collec_trajs = collect_md_trajs(job_output, traj_file=self.traj_file)
        job_list.append(collec_trajs)

        return Response(replace=Flow(job_list), output=collec_trajs.output)
