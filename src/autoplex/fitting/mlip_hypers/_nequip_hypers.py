from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field

from autoplex.settings import AutoplexBaseModel

# ---------------------------------------------------------------------------
# Config to adjust Hydra run dir
# ---------------------------------------------------------------------------


class HydraRunConfig(AutoplexBaseModel):
    dir: str = "./nequip_model"


class HydraConfig(AutoplexBaseModel):
    run: HydraRunConfig = HydraRunConfig()


# ---------------------------------------------------------------------------
# Hydra-style `_target_: ...` block
# ---------------------------------------------------------------------------
class TargetBlock(AutoplexBaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    target_: str = Field(alias="_target_")


class Transform(TargetBlock):
    """
    Any entry under data.transforms.

    Extra kwargs pass through.
    (NeighborListTransform, ChemicalSpeciesToAtomTypeMapper, ...)
    """


# ---------------------------------------------------------------------------
# Models defining data section of nequip config
# ---------------------------------------------------------------------------


class SplitDataset(AutoplexBaseModel):
    file_path: str = "./train.extxyz"
    train: float = 0.9
    val: float = 0.1
    test: float | None = None


class TrainDataLoaderConfig(TargetBlock):
    target_: str = Field("torch.utils.data.DataLoader", alias="_target_")
    batch_size: int = 32
    num_workers: int = 5
    shuffle: bool = True


class ValDataLoaderConfig(TargetBlock):
    target_: str = Field("torch.utils.data.DataLoader", alias="_target_")
    batch_size: int = 10
    num_workers: Any = "${data.train_dataloader.num_workers}"


class StatsManagerConfig(TargetBlock):
    target_: str = Field("nequip.data.CommonDataStatisticsManager", alias="_target_")
    dataloader_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"batch_size": 10}
    )
    type_names: Any = "${model_type_names}"


class DataConfig(TargetBlock):
    target_: str = Field("nequip.data.datamodule.ASEDataModule", alias="_target_")
    seed: int = 123
    split_dataset: SplitDataset = Field(default_factory=SplitDataset)
    transforms: list[Transform] = Field(
        default_factory=lambda: [
            Transform(
                _target_="nequip.data.transforms.NeighborListTransform",
                r_max="${cutoff_radius}",
            ),
            Transform(
                _target_="nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                model_type_names="${model_type_names}",
            ),
        ]
    )
    train_dataloader: TrainDataLoaderConfig = Field(
        default_factory=TrainDataLoaderConfig
    )
    val_dataloader: ValDataLoaderConfig = Field(default_factory=ValDataLoaderConfig)
    test_dataloader: Any = "${data.val_dataloader}"
    stats_manager: StatsManagerConfig = Field(default_factory=StatsManagerConfig)
    key_mapping: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Models defining section of nequip config
# ---------------------------------------------------------------------------
class LoggerConfig(TargetBlock):
    target_: str = Field(
        "lightning.pytorch.loggers.csv_logs.CSVLogger", alias="_target_"
    )
    name: str = "tutorial_log"
    save_dir: str = "./results"
    flush_logs_every_n_steps: int = 20


class TrainerConfig(TargetBlock):
    target_: str = Field("lightning.Trainer", alias="_target_")
    accelerator: str = "cpu"
    enable_checkpointing: bool = True
    max_epochs: int = 2
    max_time: str = "03:00:00:00"
    log_every_n_steps: int = 1
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: list[TargetBlock] = Field(
        default_factory=lambda: [
            TargetBlock(
                _target_="lightning.pytorch.callbacks.EarlyStopping",
                monitor="${monitored_metric}",
                min_delta=1e-3,
                patience=20,
            ),
            TargetBlock(
                _target_="lightning.pytorch.callbacks.ModelCheckpoint",
                monitor="${monitored_metric}",
                dirpath="${hydra:runtime.output_dir}",
                filename="best",
                save_last=True,
            ),
            TargetBlock(
                _target_="lightning.pytorch.callbacks.LearningRateMonitor",
                logging_interval="epoch",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Models defining training_module section of nequip config
# ---------------------------------------------------------------------------
class LossCoeffs(AutoplexBaseModel):
    total_energy: float = 1.0
    forces: float = 1.0


class MetricCoeffs(AutoplexBaseModel):
    total_energy_mae: float = 1.0
    forces_mae: float = 1.0
    total_energy_rmse: float | None = None
    forces_rmse: float | None = None
    per_atom_energy_rmse: float | None = None
    per_atom_energy_mae: float | None = None


class LossConfig(TargetBlock):
    target_: str = Field("nequip.train.EnergyForceLoss", alias="_target_")
    per_atom_energy: bool = True
    coeffs: LossCoeffs = Field(default_factory=LossCoeffs)


class MetricsConfig(TargetBlock):
    target_: str = Field("nequip.train.EnergyForceMetrics", alias="_target_")
    coeffs: MetricCoeffs = Field(default_factory=MetricCoeffs)


class OptimizerConfig(TargetBlock):
    target_: str = Field("torch.optim.Adam", alias="_target_")
    lr: float = 0.005  # 0.1


class SchedulerConfig(TargetBlock):
    target_: str = Field("torch.optim.lr_scheduler.ReduceLROnPlateau", alias="_target_")
    factor: float = 0.6
    patience: int = 5
    threshold: float = 0.2
    min_lr: float = 1.0e-06


class LRSchedulerConfig(AutoplexBaseModel):
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    monitor: str = "${monitored_metric}"
    interval: str = "epoch"
    frequency: int = 1


# ---------------------------------------------------------------------------
# Models defining model section of nequip config
# ---------------------------------------------------------------------------


class PairPotentialConfig(TargetBlock):
    target_: str = Field("nequip.nn.pair_potential.ZBL", alias="_target_")
    units: str = "metal"
    chemical_species: Any = "${chemical_symbols}"


class ModelConfig(TargetBlock):
    target_: str = Field("nequip.model.NequIPGNNModel", alias="_target_")
    compile_mode: str = "compile"
    seed: int = 456
    model_dtype: str = "float32"
    type_names: Any = "${model_type_names}"
    r_max: Any = "${cutoff_radius}"
    num_bessels: int = 8
    bessel_trainable: bool = False
    polynomial_cutoff_p: int = 6
    num_layers: Any = "${num_layers}"
    l_max: int = 1
    parity: bool = True
    num_features: int = 64
    radial_mlp_depth: int = 2
    radial_mlp_width: int = 64
    avg_num_neighbors: Any = "${training_data_stats:num_neighbors_mean}"
    per_type_energy_scales: Any = "${training_data_stats:per_type_forces_rms}"
    per_type_energy_shifts: Any = "${training_data_stats:per_atom_energy_mean}"
    per_type_energy_scales_trainable: bool = False
    per_type_energy_shifts_trainable: bool = False
    pair_potential: PairPotentialConfig = Field(default_factory=PairPotentialConfig)


class TrainingModuleConfig(TargetBlock):
    target_: str = Field("nequip.train.EMALightningModule", alias="_target_")
    ema_decay: float = 0.999
    loss: LossConfig = Field(default_factory=LossConfig)
    val_metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    train_metrics: Any = "${training_module.val_metrics}"
    test_metrics: Any = "${training_module.val_metrics}"
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Model defining complete nequip config
# ---------------------------------------------------------------------------


class NEQUIPSettings(AutoplexBaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hydra: HydraConfig | None = Field(default_factory=HydraConfig)
    run: list[str] = Field(default_factory=lambda: ["train", "val"])

    cutoff_radius: float = 6.0
    num_layers: int = 4
    l_max: int = 1
    num_features: int = 32

    chemical_symbols: list[str] = Field(default_factory=lambda: ["Si"])
    model_type_names: Any = "${chemical_symbols}"
    chemical_species: Any = "${model_type_names}"

    monitored_metric: str = "val0_epoch/weighted_sum"

    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    training_module: TrainingModuleConfig = Field(default_factory=TrainingModuleConfig)

    def to_yaml(self, path: str | None = None) -> str:
        """
        Serialize to YAML text, restoring `_target_` keys and preserving field order.

        The `_target_` keys are restored during serialization and field order is
        preserved without alphabetical sorting.
        """
        data = self.model_dump(by_alias=True, exclude_none=True)
        text = yaml.dump(data, sort_keys=False, default_flow_style=False, width=88)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text


# ---------------------------------------------------------------------------
# Following models are only applicable for nequip <=v0.6.1
# ---------------------------------------------------------------------------


class Nonlinearity(AutoplexBaseModel):
    """Model describing the nonlinearity to be used for the NEQUIP fits."""

    e: Literal["silu", "ssp", "tanh", "abs"] = Field(
        default="silu", description="Even nonlinearity"
    )
    o: Literal["silu", "ssp", "tanh", "abs"] = Field(
        default="tanh", description="Odd nonlinearity"
    )


class LossCoeff(AutoplexBaseModel):
    """Model describing different weights to use in a weighted loss functions."""

    forces: int | list[int | str] = Field(
        default=1, description="Forces loss coefficient"
    )
    total_energy: int | list[int | str] = Field(
        default=[1, "PerAtomMSELoss"], description="Total energy loss coefficient"
    )


class NEQUIPSettingsOld(AutoplexBaseModel):
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
