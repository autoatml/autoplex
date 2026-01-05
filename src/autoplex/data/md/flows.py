"""Flows for running MD."""

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
from autoplex.data.md.utils import generate_temperature_profile
from autoplex.fitting.common.utils import extract_gap_label

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
        • isothermal equilibration
        • multi-stage quench (e.g., 3000 -> 1500 -> 300 K)
        • annealing or cyclic heating-cooling loops

    This maker provides a flexible way to design and execute
    temperature-controlled MD workflows directly within atomate2.

    Parameters
    ----------
    temperature_list : list[float]
        A list of values defines a multi-stage quench or anneal profile.
    eqm_step_list : list[int] | None, optional
        Number of MD steps to hold each temperature value.
        Length must match `temperatures`. If None, defaults to 10,000.
    rate_list : list[float] | None, optional
        Relative cooling/heating rates between stages (len = len(temperatures) - 1).
        If None, linear interpolation is used. A larger `rate` value produces more
        intermediate temperatures (slower quench), while a smaller value gives fewer steps
        (faster quench). For example, rate=10 -> 10^14 K/s; rate=100 -> 10^13 K/s.
    name : str
        Job name.
    volume_scale_factor_range: list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume -+10%.
    supercell_matrix: Matrix3D.
        Matrix for obtaining the supercell.
        For example, A 2x2x2 cubic supercell is defined as:
        supercell_matrix = [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ].

    Returns
    -------
    list[Path]
        A list of absolute file paths to the "MD.traj" files from each job output.
    """

    name: str = "ML-driven_MD_with_ASE"
    temperature_list: list[float] = field(default_factory=list)
    eqm_step_list: list[int] | None = None
    rate_list: list[int] | None = None
    volume_custom_scale_factors: list[float] | None = None
    supercell_matrix: Matrix3D | None = None

    @job
    def make(self, structure: Structure):
        """Maker to run MD simulations.

        Parameters
        ----------
        structure: Structure
            Pymatgen structure.
        """
        logging.warning(
            "The value of `force_field_name` must use the full, case-sensitive name. "
            "Supported types include: CHGNet, M3GNet, MATPES_R2SCAN, MATPES_PBE, "
            "MACE, MACE_MP_0, MACE_MPA_0, MACE_MP_0B3, GAP, NEP, Nequip, SevenNet. "
        )

        if self.supercell_matrix is not None:
            logging.info(f"Applying supercell_matrix:\n{self.supercell_matrix}")
            structure = structure * self.supercell_matrix
            logging.info(f"Supercell generated: {structure}.")

        if self.volume_custom_scale_factors is not None:
            n_structures = len(self.volume_custom_scale_factors)
            logging.info("Applying custom volume scaling:")
            logging.info(f"  Scale factors: {self.volume_custom_scale_factors}")
            logging.info(f"  Number of scaled structures to generate: {n_structures}")
            structure = scale_cell(
                structure=structure,
                n_structures=n_structures,
                volume_custom_scale_factors=self.volume_custom_scale_factors,
            )

        print("self.time_step", self.time_step)
        temps, n_steps = generate_temperature_profile(
            temperature_list=self.temperature_list,
            eqm_step_list=self.eqm_step_list,
            rate_list=self.rate_list,
            time_step=self.time_step,
        )
        self.temperature = temps
        self.n_steps = int(n_steps)

        job_list = []
        job_output = {}
        structures = [structure] if isinstance(structure, Structure) else structure
        device = self.calculator_kwargs.get("device", None)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logging.info("No device specified. Using CUDA (GPU).")
            else:
                device = "cpu"
                logging.info(
                    "No device specified. CUDA unavailable. Falling back to CPU."
                )

            self.calculator_kwargs["device"] = device

        if self.force_field_name == "MLFF.GAP":
            logging.info("Using GAP force field for MD simulation...")
            gap_label = self.calculator_kwargs.get("param_filename", None)
            if gap_label is None:
                raise ValueError(
                    "`param_filename` must be provided for GAP force field."
                )
            gap_path = Path(gap_label)
            if gap_path.exists():
                gap_control = "Potential xml_label=" + extract_gap_label(gap_path)
            else:
                logging.error(f"GAP parameter file not found: {gap_path}")
                raise FileNotFoundError(
                    f"GAP parameter file does not exist: {gap_path}"
                )
            self.calculator_kwargs["args_str"] = gap_control

        elif self.force_field_name == "MLFF.MACE":
            logging.info("Using MACE force field for MD simulation...")
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
            logging.info("Using NequIP force field for MD simulation...")
            model_path = self.calculator_kwargs.get("model_path", None)
            if not Path(model_path).exists():
                raise ValueError(
                    "`model_path` must be provided for NequIP force field."
                )

        elif self.force_field_name == "MLFF.NEP":
            self.calculator_kwargs.pop("device", None)
            logging.info("Using NEP force field for MD simulation...")
            model_filename = self.calculator_kwargs.get("model_filename", None)
            if not Path(model_filename).exists():
                raise ValueError(
                    "`model_filename` must be provided for NEP force field."
                )

        for idx, struct in enumerate(structures):
            md_job = ForceFieldMDMaker.make(self, structure=struct)
            job_list.append(md_job)
            job_output[f"md_job_{idx}"] = md_job.output

        collec_trajs = collect_md_trajs(job_output)
        job_list.append(collec_trajs)

        return Response(replace=Flow(job_list), output=collec_trajs.output)
