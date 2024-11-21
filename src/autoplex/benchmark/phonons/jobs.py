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
        List of benchmark Structure used for benchmarking.
    metrics: List[float]
        Root mean squared error between band structures, imagmodesdft-bool and imagmodesml-bool.
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
            "%-11s%-11s%-12s%-18s%-12s%-16s%-16s%-16s%-18s"
            % (
                "Potential",
                "Structure",
                "MPID",
                "Displacement (Å)",
                "RMSE (THz)",
                "imagmodes(pot)",
                "imagmodes(dft)",
                "Database type",
                "(Hyper-)Parameters",
            )
        )

    for metric in metrics_flattened:
        with open(
            f"{filename_prefix}{structure_composition}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            # Build the SOAP dictionary or suffix value
            soap_params = {  # (atom-wise f, n_sparse, SOAP delta)
                f"f={metric['atomwise_regularization_parameter']}": metric["soap_dict"]
            }

            if metric["ml_model"] == "GAP":
                key = next(iter(soap_params.keys()))
                value = next(iter(soap_params.values()))
                pretty_hyper_params = f"atom-wise {key}: n_sparse = {value['n_sparse']}, SOAP delta = {value['delta']}"
            else:
                pretty_hyper_params = "user defined"

            if not metric["suffix"]:
                metric["suffix"] = "full"

            file.write(
                f"\n{metric['ml_model']:<11}{structure_composition:<11}{metric['mp_id']:<12}"
                f"{metric['displacement']:<18.2f}{metric['benchmark_phonon_rmse']:<12.5f}"
                f"{metric['ml_imaginary_modes']!s:<16}{metric['dft_imaginary_modes']!s:<16}"
                f"{metric['suffix']!s:<16}{pretty_hyper_params!s:<50}"
            )

    return Response(output=metrics)
