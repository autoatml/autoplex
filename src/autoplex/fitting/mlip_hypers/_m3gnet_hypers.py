from typing import Literal

import numpy as np
from pydantic import Field

try:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    Optimizer = object  # type: ignore[misc, assignment]
    LRScheduler = object  # type: ignore[misc, assignment]

from autoplex._basemodel import AutoplexBaseModel


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
