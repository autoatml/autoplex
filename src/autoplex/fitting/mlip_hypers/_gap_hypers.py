from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


class GAPGeneralSettings(AutoplexBaseModel):
    """Model describing general hyperparameters for the GAP fits."""

    at_file: str = Field(
        default="train.extxyz", description="Name of the training file"
    )
    default_sigma: str = Field(
        default="{0.0001 0.05 0.05 0}", description="Default sigma values"
    )
    energy_parameter_name: str = Field(
        default="REF_energy", description="Name of the energy parameter"
    )
    force_parameter_name: str = Field(
        default="REF_forces", description="Name of the force parameter"
    )
    virial_parameter_name: str = Field(
        default="REF_virial", description="Name of the virial parameter"
    )
    sparse_jitter: float = Field(default=1.0e-8, description="Sparse jitter")
    do_copy_at_file: str = Field(default="F", description="Copy the training file to")
    openmp_chunk_size: int = Field(default=10000, description="OpenMP chunk size")
    gp_file: str = Field(default="gap_file.xml", description="Name of the GAP file")
    e0_offset: float = Field(default=0.0, description="E0 offset")
    two_body: bool = Field(
        default=False, description="Whether to include two-body terms"
    )
    three_body: bool = Field(
        default=False, description="Whether to include three-body terms"
    )
    soap: bool = Field(default=True, description="Whether to include SOAP terms")


class TwobSettings(AutoplexBaseModel):
    """Model describing two body hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=2,
        description="Distance_Nb order for two-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for two-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=5.0, description="Radial cutoff distance")
    n_sparse: int = Field(default=15, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for two-body"
    )
    delta: float = Field(default=2.00, description="Delta value for two-body")
    theta_uniform: float = Field(
        default=0.5, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for two-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class ThreebSettings(AutoplexBaseModel):
    """Model describing threebody hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=3,
        description="Distance_Nb order for three-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for three-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=3.25, description="Radial cutoff distance")
    n_sparse: int = Field(default=100, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for three-body"
    )
    delta: float = Field(default=2.00, description="Delta value for three-body")
    theta_uniform: float = Field(
        default=1.0, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for three-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class SoapSettings(AutoplexBaseModel):
    """Model describing soap hyperparameters for the GAP fits."""

    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    l_max: int = Field(default=10, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.5, description="Width of Gaussian smearing")
    zeta: int = Field(default=4, description="Exponent for dot-product SOAP kernel")
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    central_weight: float = Field(default=1.0, description="Weight for central atom")
    n_sparse: int = Field(default=6000, description="Number of sparse points")
    delta: float = Field(default=1.00, description="Delta value for SOAP")
    f0: float = Field(default=0.0, description="F0 value for SOAP")
    covariance_type: str = Field(
        default="dot_product", description="Covariance type for SOAP"
    )
    sparse_method: str = Field(
        default="cur_points", description="Sparse method for SOAP"
    )


class GAPSettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the GAP fits for Phonons."""

    general: GAPGeneralSettings = Field(
        default_factory=GAPGeneralSettings,
        description="General hyperparameters for the GAP fits",
    )
    twob: TwobSettings = Field(
        default_factory=TwobSettings,
        description="Two body hyperparameters for the GAP fits",
    )
    threeb: ThreebSettings = Field(
        default_factory=ThreebSettings,
        description="Three body hyperparameters for the GAP fits",
    )
    soap: SoapSettings = Field(
        default_factory=SoapSettings,
        description="Soap hyperparameters for the GAP fits",
    )
