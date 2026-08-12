from typing import Any

from pydantic import Field

from autoplex._basemodel import AutoplexBaseModel


class PacemakerSettings(AutoplexBaseModel):
    """
    Model describing the hyperparameters for the Pacemaker (P-ACE) fits.

    Structure matches the input.yaml sections of Pacemaker.
    Dictionary-based fields provide flexibility and allow any valid Pacemaker input.
    Note: If user provides a value for any nested dict field (e.g., 'fit', 'potential'),
    it will COMPLETELY REPLACE the default value, not merge with it.
    Users should provide complete configurations for any field they customize.
    """

    seed: int | None = Field(default=42, description="Random seed")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata dictionary"
    )

    cutoff: float | int | dict[str, Any] = Field(
        default=5.0,
        description="Cutoff radius (float) or dict config (e.g. {name: hard, r_cut: 7.0})",
    )

    data: dict[str, Any] = Field(
        default_factory=lambda: {
            "filename": "train.pckl.gzip",
            "test_filename": None,
        },
        description="Data configuration block. If provided by user, completely replaces default.",
    )

    potential: dict[str, Any] = Field(
        default_factory=lambda: {
            "deltaSplineBins": 0.001,
            "elements": None,
            "embeddings": {
                "ALL": {
                    "npot": "FinnisSinclairShiftedScaled",
                    "fs_parameters": [1, 1, 1, 0.5],
                    "ndensity": 2,
                }
            },
            "bonds": {
                "ALL": {
                    "radbase": "ChebExpCos",
                    "radparameters": [5.25],
                    "rcut": 7.0,
                    "dcut": 0.01,
                    "r_in": 1.0,
                    "delta_in": 0.5,
                }
            },
            "functions": {
                "ALL": {
                    "nradmax_by_orders": [15, 3, 2, 2],
                    "lmax_by_orders": [0, 2, 2, 1],
                },
                "number_of_functions_per_element": 200,
            },
        },
        description="Potential configuration block. If provided by user, completely replaces default.",
    )

    fit: dict[str, Any] = Field(
        default_factory=lambda: {
            "optimizer": "BFGS",
            "maxiter": 100,
            "loss": {"kappa": 0.8, "w_energy": 1.0, "w_forces": 1.0, "w_stress": 0.1},
            "weighting": {"type": "EnergyBasedWeightingPolicy", "nfit": 10000},
            "repulsion": "auto",
            "trainable_parameters": "ALL",
        },
        description="Fitting configuration block. If provided by user, completely replaces default.",
    )

    # --- Backend Section ---
    backend: dict[str, Any] = Field(
        default_factory=lambda: {
            "evaluator": "tensorpot",
            "batch_size_evaluation": 1000,
            "batch_size_training": 100,
        },
        description="Backend execution settings. If provided by user, completely replaces default.",
    )
