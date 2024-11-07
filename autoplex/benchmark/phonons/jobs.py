"""Atomistic Jobs to Benchmark Potentials."""

from __future__ import annotations

from jobflow import Response, job


@job
def write_benchmark_metrics(
    benchmark_structures: list,
    metrics: list,
    filename_prefix: str = "results_",
):
    """
    Generate a text file with evaluated benchmark metrics.

    Parameters
    ----------
    benchmark_structures: List[Structure].
        list of benchmark Structure used for benchmarking.
    metrics: List[float]
        root mean squared error between band structures, imagmodesdft-bool and imagmodesml-bool.
    filename_prefix: str
        Prefix of the result summary file.

    Returns
    -------
    A text file with root mean squared error between DFT and ML potential phonon band-structure
    """
    # TODO: fix this part
    metrics_flattened = [item for sublist in metrics for item in sublist]
    # TODO: think about a better solution here
    # the following code assumes all benchmark structures have the same composition
    structure_composition = benchmark_structures[0].composition.reduced_formula
    with open(
        f"{filename_prefix}{structure_composition}.txt",
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            "%-11s%-11s%-12s%-18s%-12s%-55s%-16s%-16s%-14s"
            % (
                "Potential",
                "Structure",
                "MPID",
                "Displacement (Å)",
                "RMSE (THz)",
                "Hyperparameters (atom-wise f, n_sparse, SOAP delta)",
                "Database type",
                "imagmodes(pot)",
                "imagmodes(dft)",
            )
        )

    for metric in metrics_flattened:
        with open(
            f"{filename_prefix}{structure_composition}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            # Build the SOAP dictionary or suffix value
            soap_params = {
                f"f={metric['atomwise_regularization_parameter']}": metric["soap_dict"]
            }

            file.write(
                f"\n{metric['ml_model']:<11}{structure_composition:<11}{metric['mp_id']:<12}"
                f"{metric['displacement']:<18.2f}{metric['benchmark_phonon_rmse']:<12.5f}"
                f"{soap_params!s:<55}{metric['suffix']!s:<16}{metric['ml_imaginary_modes']!s:<16}"
                f"{metric['dft_imaginary_modes']!s:<5}"
            )

    return Response(output=metrics)
