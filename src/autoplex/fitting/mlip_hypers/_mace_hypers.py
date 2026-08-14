from typing import Literal

from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


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
    correlation: int = Field(default=3, description="Correlation order at each layer")
    default_dtype: Literal["float32", "float64"] = Field(
        default="float32", description="Default data type"
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
    keep_checkpoints: bool = Field(default=False, description="Keep all checkpoints")
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
