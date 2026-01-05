"""Unified flow for automatic data generation and MLIP fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from atomate2.forcefields.jobs import (
        ForceFieldRelaxMaker,
        ForceFieldStaticMaker,
    )
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from pymatgen.core.structure import Structure

    from autoplex.misc.castep.jobs import BaseCastepMaker
    from autoplex.settings import RssConfig

from jobflow import Flow, Maker, Response, job

from autoplex.auto.md.jobs import do_md_single_step
from autoplex.auto.rss.flows import RssMaker
from autoplex.data.common.flows import (
    _DEFAULT_RELAXATION_MAKER,
    _DEFAULT_STATIC_ENERGY_MAKER,
    RattledTrainingDataMaker,
)


@dataclass
class DataGenerationMaker(Maker):
    """
    Unified data generation maker.

    Parameters
    ----------
    name : str
        Name of the maker.
    method : {"md", "rss", "rattle"}
        Select which data generation workflow to run.
    static_energy_maker: BaseVaspMaker | CastepStaticMaker | ForceFieldStaticMaker
        Maker for static energy jobs: either BaseVaspMaker (VASP-based) or CastepStaticMaker (CASTEP-based) or
        ForceFieldStaticMaker (force field-based). Defaults to StaticMaker (VASP-based).
    static_energy_maker_isolated_atoms: BaseVaspMaker | ForceFieldStaticMaker | None
        Maker for static energy jobs of isolated atoms: either BaseVaspMaker (VASP-based) or CastepStaticMaker
        (CASTEP-based) or ForceFieldStaticMaker (force field-based) or None. If set to `None`, the parameters
        from `static_energy_maker` will be used as the default for isolated atoms. In this case,
        if `static_energy_maker` is a `StaticMaker`, all major settings will be inherited,
        except that `kspacing` will be automatically set to 100 to enforce a Gamma-point-only calculation.
        This is typically suitable for single-atom systems. Default is None. If a non-`StaticMaker` maker
        is used here, its output must include a `dir_name` field to ensure compatibility with downstream workflows.
    bulk_relax_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldRelaxMaker
        Maker used to produce the relaxed structure that will be
        perturbed. Defaults to _DEFAULT_RELAXATION_MAKER.
    md_kwargs : dict
        Keyword arguments forwarded to ``do_md_single_step`` when ``method="md"``.
    rss_config
        RSS configuration object used when ``method="rss"``.
    rss_kwargs : dict
        Keyword arguments forwarded to ``RssMaker.make`` when ``method="rss"``.
    rattle_kwargs : dict
        Keyword arguments used to construct ``RattledTrainingDataMaker`` when
        ``method="rattle"``.
    """

    name: str = "data_generation"
    method: Literal["md", "rss", "rattle"] = "md"
    static_energy_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker = (
        field(default_factory=lambda: _DEFAULT_STATIC_ENERGY_MAKER)
    )
    static_energy_maker_isolated_atoms: (
        BaseVaspMaker | BaseCastepMaker | ForceFieldStaticMaker | None
    ) = None
    bulk_relax_maker: BaseVaspMaker | BaseCastepMaker | ForceFieldRelaxMaker = field(
        default_factory=lambda: _DEFAULT_RELAXATION_MAKER
    )
    md_kwargs: dict[str, Any] = field(default_factory=dict)
    rss_config: RssConfig | None = None
    rss_kwargs: dict[str, Any] = field(default_factory=dict)
    rattle_kwargs: dict[str, Any] = field(default_factory=dict)

    @job
    def make(self, structure: Structure | None = None):
        """
        Make to generate reference data.

        Parameters
        ----------
        structure : Optional[Structure]
            Input crystal structure to serve as the starting point. Typically
            the primitive or conventional cell of a bulk material.

        Returns
        -------
        dict
            A dictionary containing:
            - pre_database_dir: Path to the directory with collected DFT data.
            - isolated_atom_energies: Mapping of element symbols to their isolated-atom reference
            energies.
            - mlip_path: Path to the fitted MLIP model when ``method`` is
            ``"md"`` or ``"rss"``. Set to ``None`` when ``method`` is ``"rattle"``.
        """
        job_list = []
        final_job = None

        if self.method not in ("md", "rss", "rattle"):
            raise ValueError("method must be one of: 'md', 'rss', 'rattle'")

        if self.method == "md":
            if structure is None:
                raise ValueError("structure must be provided for method='md'")

            md_args = dict(self.md_kwargs)
            if self.static_energy_maker is not None:
                md_args["static_energy_maker"] = self.static_energy_maker
            if self.static_energy_maker_isolated_atoms is not None:
                md_args["static_energy_maker_isolated_atoms"] = (
                    self.static_energy_maker_isolated_atoms
                )

            fit_kwargs = md_args.pop("fit_kwargs", {}) or {}
            final_job = do_md_single_step(structure=structure, **md_args, **fit_kwargs)

        elif self.method == "rss":
            if self.rss_config is None:
                raise ValueError("rss_config must be provided when method='rss'")

            rss_maker_ctor_kwargs = {}
            if self.static_energy_maker is not None:
                rss_maker_ctor_kwargs["static_energy_maker"] = self.static_energy_maker

            if self.static_energy_maker_isolated_atoms is not None:
                rss_maker_ctor_kwargs["static_energy_maker_isolated_atoms"] = (
                    self.static_energy_maker_isolated_atoms
                )

            rss_maker = RssMaker(
                name=self.name,
                rss_config=self.rss_config,
                **rss_maker_ctor_kwargs,
            )

            final_job = rss_maker.make(**self.rss_kwargs)

        elif self.method == "rattle":
            if structure is None:
                raise ValueError("structure must be provided for method='rattle'")

            rattle_args = dict(self.rattle_kwargs)

            if self.static_energy_maker is not None:
                rattle_args["static_energy_maker"] = self.static_energy_maker

            if self.static_energy_maker_isolated_atoms is not None:
                rattle_args["static_energy_maker_isolated_atoms"] = (
                    self.static_energy_maker_isolated_atoms
                )

            final_job = RattledTrainingDataMaker(**rattle_args).make(
                structure=structure
            )

        job_list.append(final_job)

        if self.method in ("rss", "md"):
            return Response(
                replace=Flow(job_list),
                output={
                    "pre_database_dir": final_job.output["pre_database_dir"],
                    "mlip_path": final_job.output["mlip_path"],
                    "isolated_atom_energies": final_job.output[
                        "isolated_atom_energies"
                    ],
                },
            )

        return Response(
            replace=Flow(job_list),
            output={
                "pre_database_dir": final_job.output["pre_database_dir"],
                "mlip_path": None,
                "isolated_atom_energies": final_job.output["isolated_atom_energies"],
            },
        )
