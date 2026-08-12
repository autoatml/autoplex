"""Module defining hyperparameter sets for MLIP fits."""

from pydantic import ConfigDict, Field

from autoplex._basemodel import AutoplexBaseModel
from autoplex.fitting.mlip_hypers._gap_hypers import GAPSettings
from autoplex.fitting.mlip_hypers._jace_hypers import JACESettings
from autoplex.fitting.mlip_hypers._m3gnet_hypers import M3GNETSettings
from autoplex.fitting.mlip_hypers._mace_hypers import MACESettings
from autoplex.fitting.mlip_hypers._nep_hypers import NEPSettings
from autoplex.fitting.mlip_hypers._nequip_hypers import NEQUIPSettings
from autoplex.fitting.mlip_hypers._pace_hypers import PacemakerSettings


class MLIPHypers(AutoplexBaseModel):
    """Model containing the hyperparameter defaults for supported MLIPs in autoplex."""

    GAP: GAPSettings = Field(
        default_factory=GAPSettings, description="Hyperparameters for the GAP model"
    )
    J_ACE: JACESettings = Field(
        default_factory=JACESettings,
        description="Hyperparameters for the J-ACE model",
        alias="J-ACE",
    )
    NEQUIP: NEQUIPSettings = Field(
        default_factory=NEQUIPSettings,
        description="Hyperparameters for the NEQUIP model",
    )
    M3GNET: M3GNETSettings = Field(
        default_factory=M3GNETSettings,
        description="Hyperparameters for the M3GNET model",
    )
    MACE: MACESettings = Field(
        default_factory=MACESettings, description="Hyperparameters for the MACE model"
    )
    NEP: NEPSettings = Field(
        default_factory=NEPSettings, description="Hyperparameters for the NEP model"
    )

    P_ACE: PacemakerSettings = Field(
        default_factory=PacemakerSettings,
        description="Hyperparameters for the P-ACE model",
        alias="P-ACE",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
        revalidate_instances="never",
    )
