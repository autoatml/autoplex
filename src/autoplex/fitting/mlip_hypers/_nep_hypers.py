from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


class NEPSettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the NEP fits."""

    version: int = Field(default=4, description="Version of the NEP model")
    type: list[int | str] = Field(
        default_factory=lambda: [1, "X"],
        description="Mandatory Parameter. Number of atom types and list of "
        "chemical species. Number of atom types must be an integer, followed by "
        "chemical symbols of species as in periodic table "
        "for which model needs to be trained, separated by comma. "
        "Default is [1, 'X'] as a placeholder. Example: [2, 'Pb', 'Te']",
    )
    type_weight: float = Field(
        default=1.0, description="Weights for different chemical species"
    )
    model_type: int = Field(
        default=0,
        description="Type of model that is being trained. "
        "Can be 0 (potential), 1 (dipole), "
        "2 (polarizability)",
    )
    prediction: int = Field(
        default=0, description="Mode of NEP run. Set 0 for training and 1 for inference"
    )
    cutoff: list[int, int] = Field(
        default_factory=lambda: [6, 5],
        description="Radial and angular cutoff. First element is for radial cutoff "
        "and second element is for angular cutoff",
    )
    n_max: list[int, int] = Field(
        default_factory=lambda: [4, 4],
        description="Number of radial and angular descriptors. First element "
        "is for radial and second element is for angular.",
    )
    basis_size: list[int, int] = Field(
        default_factory=lambda: [8, 8],
        description="Number of basis functions that are used to build the radial and angular descriptor. "
        "First element is for radial descriptor and second element is for angular descriptor",
    )
    l_max: list[int] = Field(
        default_factory=lambda: [4, 2, 1],
        description="The maximum expansion order for the angular terms. "
        "First element is for three-body, second element is for four-body and third element is for five-body",
    )
    neuron: int = Field(
        default=80, description="Number of neurons in the hidden layer."
    )
    lambda_1: float = Field(
        default=0.0, description="Weight for the L1 regularization term."
    )
    lambda_e: float = Field(default=1.0, description="Weight for the energy loss term.")
    lambda_f: float = Field(default=1.0, description="Weight for the force loss term.")
    lambda_v: float = Field(default=0.1, description="Weight for the virial loss term.")
    force_delta: int = Field(
        default=0,
        description=" Sets bias the on the loss function to put more emphasis "
        "on obtaining accurate predictions for smaller forces.",
    )
    batch: int = Field(default=1000, description="Batch size for training.")
    population: int = Field(
        default=60, description="Size of the population used by the SNES algorithm."
    )
    generation: int = Field(
        default=100000, description="Number of generations used by the SNES algorithm."
    )
    zbl: int = Field(
        default=2,
        description="Cutoff to use in universal ZBL potential at short distances. "
        "Acceptable values are in range 1 to 2.5.",
    )
