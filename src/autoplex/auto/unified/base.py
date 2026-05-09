from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, Response, job

from autoplex.auto.unified.flows import DataGenerationMaker
from autoplex.data.common.flows import (
    _DEFAULT_RELAXATION_MAKER,
    _DEFAULT_STATIC_ENERGY_MAKER,
)
from autoplex.fitting.common.flows import MLIPFitMaker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atomate2.forcefields.jobs import (
        ForceFieldRelaxMaker,
        ForceFieldStaticMaker,
    )
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from pymatgen.core.structure import Structure

    from autoplex.misc.castep.jobs import BaseCastepMaker

    # Placeholder types.
    # These are expected to be replaced by real config classes for MD and rattling later.
    from autoplex.settings import MDConfig, RattleConfig, RssConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMaker(Maker):
    """
    A extensible workflow that unifies data generation and MLIP fitting via optional multi-stage iteration.

    The workflow supports arbitrary, reproducible sequences of RSS, MD, and rattling, where both the order
    and repetition of stages are allowed (e.g. RSS -> MD -> RSS or MD -> RSS -> rattling).
    Note that this class is intentionally written as an abstract-style template.
    """

    name: str = "unified workflow"
    # Ordered list of data-generation stages.
    # Example: ["rss", "md", "rattling"] means the workflow runs RSS first,
    # then MD, and finally rattling.
    multi_stages: Sequence[str] = field(
        default_factory=lambda: ["rss", "md", "rattling"]
    )

    # Keyword arguments passed directly to MD jobs
    md_kwargs: dict | None = None
    # Optional MD configurations loaded from YAML files, to be implemented in the near future.
    md_config: MDConfig | None = None
    rss_kwargs: dict | None = None
    rss_config: RssConfig | None = None
    rattle_kwargs: dict | None = None
    rattle_config: RattleConfig | None = None

    static_energy_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker = (
        field(default_factory=lambda: _DEFAULT_STATIC_ENERGY_MAKER)
    )
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker | None
    ) = None
    bulk_relax_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldRelaxMaker = field(
        default_factory=lambda: _DEFAULT_RELAXATION_MAKER
    )

    stages_requiring_fit: Sequence[str] = field(default_factory=lambda: ["rattling"])

    @job
    def make(
        self,
        structure: Structure | None = None,
        **fit_kwargs,
    ) -> Response:
        """
        Build the unified workflow.

        Workflow logic:
            For each stage in self.multi_stages:
                1. Run DataGenerationMaker for the stage
                2. If the stage requires MLIP fitting:
                    - Run MLIPFitMaker
                    - Store outputs for the next stage
                3. Inject MLIP outputs into the next stage configs
        """
        jobs = []
        # Cache MLIP outputs from the previous stage (if any)
        last_fit_outputs = {}

        for stage in self.multi_stages:

            md_config = deepcopy(self.md_config) if self.md_config is not None else {}
            md_config.update(deepcopy(self.md_kwargs))

            rss_config = (
                deepcopy(self.rss_config) if self.rss_config is not None else {}
            )
            rss_config.update(deepcopy(self.rss_kwargs))

            rattle_config = (
                deepcopy(self.rattle_config) if self.rattle_config is not None else {}
            )
            rattle_config.update(deepcopy(self.rattle_kwargs))

            # ------------------------------------------------------------------
            # Inject MLIP outputs from the previous stage, if available
            # This enables iterative refinement across stages
            # ------------------------------------------------------------------
            if last_fit_outputs:
                mlip_path = last_fit_outputs.get("mlip_path")
                pre_db = last_fit_outputs.get("pre_database_dir")

                # MD config
                md_config.setdefault("calculator_kwargs", {})
                md_config["calculator_kwargs"]["param_filename"] = mlip_path
                md_config["pre_database_dir"] = pre_db

                # RSS config
                rss_config.setdefault("resume_from_previous_state", {})
                rss_config["resume_from_previous_state"].update(
                    {
                        "pre_database_dir": pre_db,
                        "mlip_path": mlip_path,
                    }
                )

                # Rattling config
                rattle_config["pre_database_dir"] = pre_db

            # ------------------------------------------------------------------
            # Data generation stage
            # ------------------------------------------------------------------
            do_data_gen = DataGenerationMaker(
                method=stage,
                md_config=md_config,
                rss_config=rss_config,
                rattle_config=rattle_config,
                static_energy_maker=self.static_energy_maker,
                static_energy_maker_isolated_atoms=self.static_energy_maker_isolated_atoms,
                bulk_relax_maker=self.bulk_relax_maker,
            ).make(structure=structure)

            jobs.append(do_data_gen)

            # ------------------------------------------------------------------
            # Optional MLIP fitting stage
            # ------------------------------------------------------------------
            if stage in self.stages_requiring_fit:
                do_mlip_fit = MLIPFitMaker(...).make(
                    database_dir=do_data_gen.output["pre_database_dir"],
                    isolated_atom_energies=do_data_gen.output["isolated_atom_energies"],
                    **fit_kwargs,
                )

                jobs.append(do_mlip_fit)

                # Cache outputs for the next stage
                last_fit_outputs = {
                    "pre_database_dir": do_mlip_fit.output["pre_database_dir"],
                    "mlip_path": do_mlip_fit.output.get("mlip_path")[0],
                    "isolated_atom_energies": do_mlip_fit.output[
                        "isolated_atom_energies"
                    ],
                }

        return Response(
            replace=Flow(jobs),
            output={
                "pre_database_dir": jobs[-1].output["pre_database_dir"],
                "mlip_path": jobs[-1].output["mlip_path"][0],
                "isolated_atom_energies": jobs[-1].output["isolated_atom_energies"],
            },
        )
