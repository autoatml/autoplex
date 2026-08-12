from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


class JACESettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the J-ACE fits."""

    order: int = Field(default=3, description="Order of the J-ACE model")
    totaldegree: int = Field(default=6, description="Total degree of the J-ACE model")
    cutoff: float = Field(default=2.0, description="Radial cutoff distance")
    solver: str = Field(default="BLR", description="Solver for the J-ACE model")
