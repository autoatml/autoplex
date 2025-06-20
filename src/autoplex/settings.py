"""Settings for autoplex."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np  # noqa: TC002
from monty.json import MontyDecoder, jsanitize
from monty.serialization import loadfn
from pydantic import BaseModel, ConfigDict, Field
from torch.optim import Optimizer  # noqa: TC002
from torch.optim.lr_scheduler import LRScheduler  # noqa: TC002

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

__all__ = [
    "AutoplexBaseModel",
    "GAPSettings",
    "JACESettings",
    "M3GNETSettings",
    "MACESettings",
    "MLIPHypers",
    "NEPSettings",
    "NEQUIPSettings",
    "RssConfig",
]


class AutoplexBaseModel(BaseModel):
    """Base class for all models in autoplex."""

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def update_parameters(self, updates: dict[str, Any]):
        """
        Update the default parameters of the model instance, including nested fields.

        Args:
            updates (Dict[str, Any]): A dictionary containing the fields as keys to update.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                field_value = getattr(self, key)
                if isinstance(field_value, self.__class__) and isinstance(value, dict):
                    # Update nested model
                    field_value.update_parameters(
                        value
                    )  # Recursively call update_parameters
                else:
                    # Update field value
                    setattr(self, key, value)

            else:
                logging.warning(
                    f"Field {key} not found in default {self.__class__.__name__} model."
                    f"New field has been added. Please ensure the added field contains correct datatype."
                )
                setattr(self, key, value)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the parameters from a file.

        Args:
            filename (str): The name of the file to load the parameters from.
        """
        custom_params = loadfn(filename)

        return cls(**custom_params)

    def as_dict(self):
        """Return the model as a MSONable dictionary."""
        return jsanitize(
            self.model_copy(deep=True), strict=True, allow_bson=True, enum_values=True
        )

    @classmethod
    def from_dict(cls, d: dict):
        """Create a model from a MSONable dictionary.

        Args:
            d (dict): A MSONable dictionary representation of the Model.
        """
        decoded = {
            k: MontyDecoder().process_decoded(v)
            for k, v in d.items()
            if not k.startswith("@")
        }
        return cls(**decoded)


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


class JACESettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the J-ACE fits."""

    order: int = Field(default=3, description="Order of the J-ACE model")
    totaldegree: int = Field(default=6, description="Total degree of the J-ACE model")
    cutoff: float = Field(default=2.0, description="Radial cutoff distance")
    solver: str = Field(default="BLR", description="Solver for the J-ACE model")


class Nonlinearity(AutoplexBaseModel):
    """Model describing the nonlinearity to be used for the NEQUIP fits."""

    e: Literal["silu", "ssp", "tanh", "abs"] = Field(
        default="silu", description="Even nonlinearity"
    )
    o: Literal["silu", "ssp", "tanh", "abs"] = Field(
        default="tanh", description="Odd nonlinearity"
    )


class LossCoeff(BaseModel):
    """Model describing different weights to use in a weighted loss functions."""

    forces: int | list[int | str] = Field(
        default=1, description="Forces loss coefficient"
    )
    total_energy: int | list[int | str] = Field(
        default=[1, "PerAtomMSELoss"], description="Total energy loss coefficient"
    )


class NEQUIPSettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the NEQUIP fits.

    References
    ----------
        * Defaults taken from https://github.com/mir-group/nequip/blob/main/configs/
    """

    root: str = Field(default="results", description="Root directory")
    run_name: str = Field(default="autoplex", description="Name of the run")
    seed: int = Field(default=123, description="Model seed")
    dataset_seed: int = Field(default=123, description="Dataset seed")
    append: bool = Field(
        default=False,
        description="When true a restarted run will append to the previous log file",
    )
    default_dtype: str = Field(default="float64", description="Default data type")
    model_dtype: str = Field(default="float64", description="Model data type")
    allow_tf32: bool = Field(
        default=True,
        description="Consider setting to false if you plan to mix "
        "training/inference over any devices that are "
        "not NVIDIA Ampere or later",
    )
    r_max: float = Field(default=4.0, description="Radial cutoff distance")
    num_layers: int = Field(default=4, description="Number of layers")
    l_max: int = Field(default=2, description="Maximum degree of spherical harmonics")
    parity: bool = Field(
        default=True,
        description="Whether to include features with odd mirror parity; "
        "often turning parity off gives equally good results but faster networks",
    )
    num_features: int = Field(default=32, description="Number of features")
    nonlinearity_type: Literal["gate", "norm"] = Field(
        default="gate", description="Type of nonlinearity, 'gate' is recommended"
    )
    nonlinearity_scalars: Nonlinearity = Field(
        default_factory=Nonlinearity, description="Nonlinearity scalars"
    )
    nonlinearity_gates: Nonlinearity = Field(
        default_factory=Nonlinearity, description="Nonlinearity gates"
    )
    num_basis: int = Field(
        default=8, description="Number of basis functions used in the radial basis"
    )
    besselbasis_trainable: bool = Field(
        default=True,
        description="If true, train the bessel weights",
        alias="BesselBasis_trainable",
    )
    polynomialcutoff_p: int = Field(
        default=5,
        description="p-exponent used in polynomial cutoff function, "
        "smaller p corresponds to stronger decay with distance",
        alias="PolynomialCutoff_p",
    )

    invariant_layers: int = Field(
        default=2, description="Number of radial layers, smaller is faster"
    )
    invariant_neurons: int = Field(
        default=64,
        description="Number of hidden neurons in radial function, smaller is faster",
    )
    avg_num_neighbors: None | Literal["auto"] = Field(
        default="auto",
        description="Number of neighbors to divide by, "
        "None => no normalization, "
        "auto computes it based on dataset",
    )
    use_sc: bool = Field(
        default=True,
        description="Use self-connection or not, usually gives big improvement",
    )
    dataset: Literal["ase"] = Field(
        default="ase",
        description="Type of data set, can be npz or ase."
        "Note that autoplex only supports ase at this point",
    )
    validation_dataset: Literal["ase"] = Field(
        default="ase",
        description="Type of validation data set, can be npz or ase."
        "Note that autoplex only supports ase at this point",
    )
    dataset_file_name: str = Field(
        default="./train_nequip.extxyz", description="Name of the dataset file"
    )
    validation_dataset_file_name: str = Field(
        default="./test.extxyz", description="Name of the validation dataset file"
    )
    ase_args: dict = Field(
        default={"format": "extxyz"}, description="Any arguments needed by ase.io.read"
    )
    dataset_key_mapping: dict = Field(
        default={"forces": "forces", "energy": "total_energy"},
        description="Mapping of keys in the dataset to the expected keys",
    )
    validation_dataset_key_mapping: dict = Field(
        default={"forces": "forces", "energy": "total_energy"},
        description="Mapping of keys in the validation dataset to the expected keys",
    )
    chemical_symbols: list[str] = Field(
        default=[], description="List of chemical symbols"
    )
    wandb: bool = Field(default=False, description="Use wandb for logging")
    verbose: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Verbosity level"
    )
    log_batch_freq: int = Field(
        default=10,
        description="Batch frequency, how often to print training errors within the same epoch",
    )
    log_epoch_freq: int = Field(
        default=1, description="Epoch frequency, how often to print training errors"
    )
    save_checkpoint_freq: int = Field(
        default=-1,
        description="Frequency to save the intermediate checkpoint. "
        "No saving of intermediate checkpoints when the value is not positive.",
    )
    save_ema_checkpoint_freq: int = Field(
        default=-1,
        description="Frequency to save the intermediate EMA checkpoint. "
        "No saving of intermediate EMA checkpoints when the value is not positive.",
    )
    n_train: int = Field(default=1000, description="Number of training samples")
    n_val: int = Field(default=1000, description="Number of validation samples")
    learning_rate: float = Field(default=0.005, description="Learning rate")
    batch_size: int = Field(default=5, description="Batch size")
    validation_batch_size: int = Field(default=10, description="Validation batch size")
    max_epochs: int = Field(default=10000, description="Maximum number of epochs")
    shuffle: bool = Field(default=True, description="Shuffle the dataset")
    metrics_key: str = Field(
        default="validation_loss",
        description="Metrics used for scheduling and saving best model",
    )
    use_ema: bool = Field(
        default=True,
        description="Use exponential moving average on weights for val/test",
    )
    ema_decay: float = Field(
        default=0.99, description="Exponential moving average decay"
    )
    ema_use_num_updates: bool = Field(
        default=True, description="Use number of updates for EMA decay"
    )
    report_init_validation: bool = Field(
        default=True,
        description="Report the validation error for just initialized model",
    )
    early_stopping_patiences: dict = Field(
        default={"validation_loss": 50},
        description="Stop early if a metric value stopped decreasing for n epochs",
    )
    early_stopping_lower_bounds: dict = Field(
        default={"LR": 1.0e-5},
        description="Stop early if a metric value is lower than the given value",
    )
    loss_coeffs: LossCoeff = Field(
        default_factory=LossCoeff, description="Loss coefficients"
    )
    metrics_components: list = Field(
        default_factory=lambda: [
            ["forces", "mae"],
            ["forces", "rmse"],
            ["forces", "mae", {"PerSpecies": True, "report_per_component": False}],
            ["forces", "rmse", {"PerSpecies": True, "report_per_component": False}],
            ["total_energy", "mae"],
            ["total_energy", "mae", {"PerAtom": True}],
        ],
        description="Metrics components",
    )
    optimizer_name: str = Field(default="Adam", description="Optimizer name")
    optimizer_amsgrad: bool = Field(
        default=True, description="Use AMSGrad variant of Adam"
    )
    lr_scheduler_name: str = Field(
        default="ReduceLROnPlateau", description="Learning rate scheduler name"
    )
    lr_scheduler_patience: int = Field(
        default=100, description="Patience for learning rate scheduler"
    )
    lr_scheduler_factor: float = Field(
        default=0.5, description="Factor for learning rate scheduler"
    )
    per_species_rescale_shifts_trainable: bool = Field(
        default=False,
        description="Whether the shifts are trainable. Defaults to False.",
    )
    per_species_rescale_scales_trainable: bool = Field(
        default=False,
        description="Whether the scales are trainable. Defaults to False.",
    )
    per_species_rescale_shifts: (
        float
        | list[float]
        | Literal[
            "dataset_per_atom_total_energy_mean",
            "dataset_per_species_total_energy_mean",
        ]
    ) = Field(
        default="dataset_per_atom_total_energy_mean",
        description="The value can be a constant float value, an array for each species, or a string. "
        "If float values are prpvided , they must be in the same energy units as the training data",
    )
    per_species_rescale_scales: (
        float
        | list[float]
        | Literal[
            "dataset_forces_absmax",
            "dataset_per_atom_total_energy_std",
            "dataset_per_species_total_energy_std",
            "dataset_per_species_forces_rms",
        ]
    ) = Field(
        default="dataset_per_species_forces_rms",
        description="The value can be a constant float value, an array for each species, or a string. "
        "If float values are prpvided , they must be in the same energy units as the training data",
    )


class M3GNETSettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the M3GNET fits."""

    exp_name: str = Field(default="training", description="Name of the experiment")
    results_dir: str = Field(
        default="m3gnet_results", description="Directory to save the results"
    )
    foundation_model: str | None = Field(
        default=None,
        description="Pretrained model. Can be a Path to locally stored model "
        "or name of pretrained PES model available in the "
        "matgl (`M3GNet-MP-2021.2.8-PES` or "
        "`M3GNet-MP-2021.2.8-DIRECT-PES`). When name of "
        "model is provided, ensure system has internet "
        "access to be able to download the model."
        "If None, the model will be trained from scratch.",
    )
    use_foundation_model_element_refs: bool = Field(
        default=False, description="Use element refs from the foundation model"
    )
    allow_missing_labels: bool = Field(
        default=False, description="Allow missing labels"
    )
    cutoff: float = Field(default=5.0, description="Cutoff radius of the graph")
    threebody_cutoff: float = Field(
        default=4.0, description="Cutoff radius for 3 body interactions"
    )
    batch_size: int = Field(default=10, description="Batch size")
    max_epochs: int = Field(default=1000, description="Maximum number of epochs")
    include_stresses: bool = Field(
        default=True, description="Whether to include stresses"
    )
    data_mean: float = Field(default=0.0, description="Mean of the training data")
    data_std: float = Field(
        default=1.0, description="Standard deviation of the training data"
    )
    decay_steps: int = Field(
        default=1000, description="Number of steps for decaying learning rate"
    )
    decay_alpha: float = Field(
        default=0.96, description="Parameter determines the minimum learning rate"
    )
    dim_node_embedding: int = Field(
        default=128, description="Dimension of node embedding"
    )
    dim_edge_embedding: int = Field(
        default=128, description="Dimension of edge embedding"
    )
    dim_state_embedding: int = Field(
        default=0, description="Dimension of state embedding"
    )
    energy_weight: float = Field(default=1.0, description="Weight for energy loss")
    element_refs: np.ndarray | None = Field(
        default=None, description="Element offset for PES"
    )
    force_weight: float = Field(default=1.0, description="Weight for forces loss")
    include_line_graph: bool = Field(
        default=True, description="Whether to include line graph"
    )
    loss: Literal["mse_loss", "huber_loss", "smooth_l1_loss", "l1_loss"] = Field(
        default="mse_loss", description="Loss function used for training"
    )
    loss_params: dict | None = Field(
        default=None, description="Loss function parameters"
    )
    lr: float = Field(default=0.001, description="Learning rate for training")
    magmom_target: Literal["absolute", "symbreak"] | None = Field(
        default="absolute",
        description="Whether to predict the absolute "
        "site-wise value of magmoms or adapt the loss "
        "function to predict the signed value "
        "breaking symmetry. If None "
        "given the loss function will be adapted.",
    )
    magmom_weight: float = Field(default=0.0, description="Weight for magnetic moments")
    max_l: int = Field(default=4, description="Maximum degree of spherical harmonics")
    max_n: int = Field(
        default=4, description="Maximum number of radial basis functions"
    )
    nblocks: int = Field(default=3, description="Number of blocks")
    optimizer: Optimizer | None = Field(default=None, description="Optimizer")
    rbf_type: Literal["Gaussian", "SphericalBessel"] = Field(
        default="Gaussian", description="Type of radial basis function"
    )
    scheduler: LRScheduler | None = Field(
        default=None, description="Learning rate scheduler"
    )
    stress_weight: float = Field(default=0.0, description="Weight for stress loss")
    sync_dist: bool = Field(
        default=False, description="Sync logging across all GPU workers"
    )
    is_intensive: bool = Field(
        default=False, description="Whether the prediction is intensive"
    )
    units: int = Field(default=128, description="Number of neurons in each MLP layer")


class MACESettings(AutoplexBaseModel):
    """Model describing the hyperparameters for the MACE fits."""

    model: Literal[
        "BOTNet",
        "MACE",
        "ScaleShiftMACE",
        "ScaleShiftBOTNet",
        "AtomicDipolesMACE",
        "EnergyDipolesMACE",
    ] = Field(default="MACE", description="type of the model")
    name: str = Field(default="MACE_model", description="Experiment name")
    amsgrad: bool = Field(default=True, description="Use amsgrad variant of optimizer")
    batch_size: int = Field(default=10, description="Batch size")
    compute_avg_num_neighbors: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(default=True, description="Compute average number of neighbors")
    compute_forces: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(default=True, description="Compute forces")
    config_type_weights: str = Field(
        default="{'Default':1.0}",
        description="String of dictionary containing the weights for each config type",
    )
    compute_stress: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(default=False, description="Compute stress")
    compute_statistics: bool = Field(default=False, description="Compute statistics")
    correlation: int = Field(default=3, description="Correlation order at each layer")
    default_dtype: Literal["float32", "float64"] = Field(
        default="float32", description="Default data type"
    )
    device: Literal["cpu", "cuda", "mps", "xpu"] = Field(
        default="cpu", description="Device to be used for model fitting"
    )
    distributed: bool = Field(
        default=False, description="Train in multi-GPU data parallel mode"
    )
    energy_weight: float = Field(default=1.0, description="Weight for the energy loss")
    ema: bool = Field(default=True, description="Whether to use EMA")
    ema_decay: float = Field(
        default=0.99, description="Exponential moving average decay"
    )
    E0s: str | None = Field(
        default=None, description="Dictionary of isolated atom energies"
    )
    forces_weight: float = Field(
        default=100.0, description="Weight for the forces loss"
    )
    foundation_filter_elements: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(default=True, description="Filter element during fine-tuning")
    foundation_model: str | None = Field(
        default=None, description="Path to the foundation model for finetuning"
    )
    foundation_model_readout: bool = Field(
        default=True, description="Use readout of foundation model for finetuning"
    )
    keep_checkpoint: bool = Field(default=False, description="Keep all checkpoints")
    keep_isolated_atoms: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(
        default=False,
        description="Keep isolated atoms in the dataset, useful for finetuning",
    )
    hidden_irreps: str = Field(default="128x0e + 128x1o", description="Hidden irreps")
    loss: Literal[
        "ef",
        "weighted",
        "forces_only",
        "virials",
        "stress",
        "dipole",
        "huber",
        "universal",
        "energy_forces_dipole",
    ] = Field(default="huber", description="Loss function")
    lr: float = Field(default=0.001, description="Learning rate")
    multiheads_finetuning: (
        bool | Literal["yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"]
    ) = Field(default=False, description="Multiheads finetuning")
    max_num_epochs: int = Field(default=1500, description="Maximum number of epochs")
    pair_repulsion: bool = Field(
        default=False, description="Use pair repulsion term with ZBL potential"
    )
    patience: int = Field(
        default=2048,
        description="Maximum number of consecutive epochs of increasing loss",
    )
    r_max: float = Field(default=5.0, description="Radial cutoff distance")
    restart_latest: bool = Field(
        default=False, description="Whether to restart the latest model"
    )
    seed: int = Field(default=123, description="Seed for the random number generator")
    save_cpu: bool = Field(default=True, description="Save CPU")
    save_all_checkpoints: bool = Field(
        default=False, description="Save all checkpoints"
    )
    scaling: Literal["std_scaling", "rms_forces_scaling", "no_scaling"] = Field(
        default="rms_forces_scaling", description="Scaling"
    )
    stress_weight: float = Field(default=1.0, description="Weight for the stress loss")
    start_swa: int = Field(
        default=1200, description="Start of the SWA", alias="start_stage_two"
    )
    swa: bool = Field(
        default=True,
        description="Use Stage Two loss weight, it will decrease the learning "
        "rate and increases the energy weight at the end of the training",
        alias="stage_two",
    )
    valid_batch_size: int = Field(default=10, description="Validation batch size")
    virials_weight: float = Field(
        default=1.0, description="Weight for the virials loss"
    )
    wandb: bool = Field(default=False, description="Use Weights and Biases for logging")


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


# RSS Configuration


class ResumeFromPreviousState(AutoplexBaseModel):
    """
    A model describing the state information.

    Useful to resume a previously interrupted or saved RSS workflow.
    When 'train_from_scratch' is set to False, this parameter is mandatory
    for the workflow to pick up from a saved state.
    """

    test_error: float | None = Field(
        default=None,
        description="The test error from the last completed training step.",
    )
    pre_database_dir: str | None = Field(
        default=None,
        description="Path to the directory containing the pre-existing database for resuming",
    )
    mlip_path: str | None = Field(
        default=None, description="Path to the file of a previous MLIP model."
    )
    isolated_atom_energies: dict | None = Field(
        default=None,
        description="A dictionary with isolated atom energy values mapped to atomic numbers",
    )


class SoapParas(AutoplexBaseModel):
    """A model describing the SOAP parameters."""

    l_max: int = Field(default=12, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.0875, description="idth of Gaussian smearing")
    cutoff: float = Field(default=10.5, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    zeta: float = Field(default=4.0, description="Exponent for dot-product SOAP kernel")
    average: bool = Field(
        default=True, description="Whether to average the SOAP vectors"
    )
    species: bool = Field(
        default=True, description="Whether to consider species information"
    )


class BcurParams(AutoplexBaseModel):
    """A model describing the parameters for the BCUR method."""

    soap_paras: SoapParas = Field(default_factory=SoapParas)
    frac_of_bcur: float = Field(
        default=0.8, description="Fraction of Boltzmann CUR selections"
    )
    bolt_max_num: int = Field(
        default=3000, description="Maximum number of Boltzmann selections"
    )


class BuildcellOptions(AutoplexBaseModel):
    """A model describing the parameters for buildcell."""

    ABFIX: bool = Field(default=False, description="Whether to fix the lattice vectors")
    NFORM: str | None = Field(default=None, description="The number of formula units")
    SYMMOPS: str | None = Field(
        default=None,
        description="	Build structures having a specified "
        "number of symmetry operations. For crystals, "
        "the allowed values are (1,2,3,4,6,8,12,16,24,48). "
        "For clusters (indicated with #CLUSTER), the allowed "
        "values are (1,2,3,5,4,6,7,8,9,10,11,12,24). "
        "Ranges are allowed (e.g., #SYMMOPS=1-4).",
    )
    SYSTEM: (
        None
        | Literal["Rhom", "Tric", "Mono", "Cubi", "Hexa", "Orth", "Tetra"]
        | set[Literal["Rhom", "Tric", "Mono", "Cubi", "Hexa", "Orth", "Tetra"]]
    ) = Field(default=None, description="Enforce a crystal system")
    SLACK: float | None = Field(default=None, description="The slack factor")
    OCTET: bool = Field(
        default=False,
        description="Check number of valence electrons is a multiple of eight",
    )
    OVERLAP: float | None = Field(default=None, description="The overlap factor")
    MINSEP: str | None = Field(default=None, description="The minimum separation")


class CustomIncar(AutoplexBaseModel):
    """A model describing the INCAR parameters."""

    ISMEAR: int = 0
    SIGMA: float = 0.05
    PREC: str = "Accurate"
    ADDGRID: str = ".TRUE."
    EDIFF: float = 1e-07
    NELM: int = 250
    LWAVE: str = ".FALSE."
    LCHARG: str = ".FALSE."
    ALGO: str = "Normal"
    AMIX: float | None = None
    LREAL: str = ".FALSE."
    ISYM: int = 0
    ENCUT: float = 520.0
    KSPACING: float = 0.2
    GGA: str | None = None
    KPAR: int = 8
    NCORE: int = 16
    LSCALAPACK: str = ".FALSE."
    LPLANE: str = ".FALSE."


class RssConfig(AutoplexBaseModel):
    """A model describing the complete RSS configuration."""

    tag: str | None = Field(
        default=None,
        description="Tag of systems. It can also be used for setting up elements "
        "and stoichiometry. For example, the tag of 'SiO2' will be recognized "
        "as a 1:2 ratio of Si to O and passed into the parameters of buildcell. "
        "However, note that this will be overwritten if the stoichiometric ratio "
        "of elements is defined in the 'cell_seed_paths' or 'buildcell_options'",
    )
    train_from_scratch: bool = Field(
        default=True,
        description="If True, it starts the workflow from scratch "
        "If False, it resumes from a previous state.",
    )
    resume_from_previous_state: ResumeFromPreviousState = Field(
        default_factory=ResumeFromPreviousState
    )
    generated_struct_numbers: list[int, int] = Field(
        default_factory=lambda: [10000],
        description="Expected number of generated "
        "randomized unit cells by buildcell.",
    )
    cell_seed_paths: list[str] | None = Field(
        default=None, description="Custom buildcell control files."
    )
    buildcell_options: list[BuildcellOptions] | None = Field(
        default=None, description="Customized parameters for buildcell."
    )
    fragment_file: str | None = Field(default=None, description="")
    fragment_numbers: list[int] | None = Field(
        default=None,
        description="Numbers of each fragment to be included in the random structures. "
        "Defaults to 1 for all specified.",
    )
    num_processes_buildcell: int = Field(
        default=128, description="Number of processes for buildcell."
    )
    num_of_initial_selected_structs: list[int, int] = Field(
        default_factory=lambda: [100],
        description="Number of structures to be sampled directly "
        "from the buildcell-generated randomized cells.",
    )
    num_of_rss_selected_structs: int = Field(
        default=100,
        description="Number of structures to be selected from each RSS iteration.",
    )
    initial_selection_enabled: bool = Field(
        default=True,
        description="If true, sample structures from initially generated "
        "randomized cells using CUR.",
    )
    rss_selection_method: Literal["bcur1s", "bcur2i", None] = Field(
        default="bcur2i",
        description="Method for selecting samples from the RSS trajectories: "
        "Boltzmann flat histogram in enthalpy first, then CUR. Options are as follows",
    )
    bcur_params: BcurParams = Field(
        default_factory=BcurParams, description="Parameters for the BCUR method."
    )
    random_seed: int | None = Field(
        default=None, description="A seed to ensure reproducibility of CUR selection."
    )
    include_isolated_atom: bool = Field(
        default=True,
        description="Perform single-point calculations for isolated atoms.",
    )
    isolatedatom_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="List of the lattice constants for an "
        "isolated atom configuration.",
    )
    e0_spin: bool = Field(
        default=False,
        description="Include spin polarization in isolated atom and dimer calculations",
    )
    include_dimer: bool = Field(
        default=True,
        description="Perform single-point calculations for dimers only once",
    )
    dimer_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="The lattice constants of a dimer box.",
    )
    dimer_range: list[float, float] = Field(
        default_factory=lambda: [1.0, 5.0],
        description="The range of the dimer distance.",
    )
    dimer_num: int = Field(
        default=21,
        description="Number of different distances to consider for dimer calculations.",
    )
    custom_incar: CustomIncar = Field(
        default_factory=CustomIncar,
        description="Custom VASP input parameters. "
        "If provided, will update the default parameters",
    )
    custom_potcar: str | None = Field(
        default=None,
        description="POTCAR settings to update. Keys are element symbols, "
        "values are the desired POTCAR labels.",
    )
    vasp_ref_file: str = Field(
        default="vasp_ref.extxyz", description="Reference file for VASP data"
    )
    config_types: list[str] = Field(
        default_factory=lambda: ["initial", "traj_early", "traj"],
        description="Configuration types for the VASP calculations",
    )
    rss_group: list[str] | str = Field(
        default_factory=lambda: ["traj"],
        description="Group of configurations for the RSS calculations",
    )
    test_ratio: float = Field(
        default=0.1,
        description="The proportion of the test set after splitting the data",
    )
    regularization: bool = Field(
        default=True,
        description="Whether to apply regularization. This only works for GAP to date.",
    )
    retain_existing_sigma: bool = Field(
        default=False,
        description="Whether to retain the existing sigma values for specific configuration types."
        "If True, existing sigma values for specific configurations will remain unchanged",
    )
    scheme: Literal["linear-hull", "volume-stoichiometry", None] = Field(
        default="linear-hull", description="Method to use for regularization"
    )
    reg_minmax: list[list[float]] = Field(
        default_factory=lambda: [
            [0.1, 1],
            [0.001, 0.1],
            [0.0316, 0.316],
            [0.0632, 0.632],
        ],
        description="List of tuples of (min, max) values for energy, force, "
        "virial sigmas for regularization",
    )
    distillation: bool = Field(
        default=False, description="Whether to apply data distillation"
    )
    force_max: float | None = Field(
        default=None, description="Maximum force value to exclude structures"
    )
    force_label: str | None = Field(
        default=None, description="The label of force values to use for distillation"
    )
    pre_database_dir: str | None = Field(
        default=None, description="Directory where the previous database was saved."
    )
    mlip_type: Literal["GAP", "J-ACE", "NEQUIP", "M3GNET", "MACE"] = Field(
        default="GAP", description="MLIP to be fitted"
    )
    ref_energy_name: str = Field(
        default="REF_energy", description="Reference energy name."
    )
    ref_force_name: str = Field(
        default="REF_forces", description="Reference force name."
    )
    ref_virial_name: str = Field(
        default="REF_virial", description="Reference virial name."
    )
    auto_delta: bool = Field(
        default=True,
        description="Whether to automatically calculate the delta value for GAP terms.",
    )
    num_processes_fit: int = Field(
        default=32, description="Number of processes used for fitting"
    )
    device_for_fitting: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for model fitting"
    )
    scalar_pressure_method: Literal["exp", "uniform"] = Field(
        default="uniform", description="Method for adding external pressures."
    )
    scalar_exp_pressure: int = Field(
        default=1, description="Scalar exponential pressure"
    )
    scalar_pressure_exponential_width: float = Field(
        default=0.2, description="Width for scalar pressure exponential"
    )
    scalar_pressure_low: int = Field(
        default=0, description="Lower limit for scalar pressure"
    )
    scalar_pressure_high: int = Field(
        default=25, description="Upper limit for scalar pressure"
    )
    max_steps: int = Field(
        default=300, description="Maximum number of steps for the GAP optimization"
    )
    force_tol: float = Field(
        default=0.01, description="Force residual tolerance for relaxation"
    )
    stress_tol: float = Field(
        default=0.01, description="Stress residual tolerance for relaxation."
    )
    stop_criterion: float = Field(
        default=0.01, description="Convergence criterion for stopping RSS iterations."
    )
    max_iteration_number: int = Field(
        default=25, description="Maximum number of RSS iterations to perform."
    )
    num_groups: int = Field(
        default=6,
        description="Number of structure groups, used for assigning tasks across multiple nodes."
        "For example, if there are 10,000 trajectories to relax and 'num_groups=10',"
        "the trajectories will be divided into 10 groups and 10 independent jobs will be created,"
        "with each job handling 1,000 trajectories.",
    )
    initial_kt: float = Field(
        default=0.3, description="Initial temperature (in eV) for Boltzmann sampling."
    )
    current_iter_index: int = Field(
        default=1, description="Current iteration index for the RSS."
    )
    hookean_repul: bool = Field(
        default=False, description="Whether to apply Hookean repulsion"
    )
    hookean_paras: dict | None = Field(
        default=None,
        description="Parameters for the Hookean repulsion as a "
        "dictionary of tuples.",
    )
    keep_symmetry: bool = Field(
        default=False, description="Whether to preserve symmetry during relaxations."
    )
    remove_traj_files: bool = Field(
        default=False,
        description="Bool indicating whether to remove the RSS trajectory files.",
    )
    num_processes_rss: int = Field(
        default=128, description="Number of processes used for running RSS."
    )
    device_for_rss: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for RSS calculations."
    )
    mlip_hypers: MLIPHypers = Field(
        default_factory=MLIPHypers, description="MLIP hyperparameters"
    )

    @classmethod
    def from_file(cls, filename: str):
        """Create RSS configuration object from a file."""
        config_params = loadfn(filename)

        # check if config file has the required keys when train_from_scratch is False
        train_from_scratch = config_params.get("train_from_scratch")
        resume_from_previous_state = config_params.get("resume_from_previous_state")

        if not train_from_scratch:
            for key, value in resume_from_previous_state.items():
                if value is None:
                    raise ValueError(
                        f"Value for {key} in `resume_from_previous_state` cannot be None when "
                        f"`train_from_scratch` is set to False"
                    )

        # check if mlip arg is in the config file
        # Needed for backward compatibility with older config files of RSS workflow
        mlip_type = config_params["mlip_type"].replace("-", "_")
        mlip_hypers = MLIPHypers().__getattribute__(mlip_type)

        if "mlip_hypers" not in config_params:
            config_params["mlip_hypers"] = {config_params["mlip_type"]: {}}

        old_config_keys = []
        for arg in config_params:
            mlip_type = config_params["mlip_type"].replace("-", "_")
            if arg in mlip_hypers.model_fields:
                config_params["mlip_hypers"][mlip_type].update(
                    {arg: config_params[arg]}
                )
                old_config_keys.append(arg)

        for key in old_config_keys:
            del config_params[key]

        return cls(**config_params)
