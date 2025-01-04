"""Utility functions for benchmarking jobs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.plotter import PhononBSPlotter


def compute_bandstructure_benchmark_metrics(
    ml_model: str,
    structure: Structure,
    mp_id: str,
    ml_phonon_bs: PhononBandStructureSymmLine,
    dft_phonon_bs: PhononBandStructureSymmLine,
    ml_imag_modes: bool,
    dft_imag_modes: bool,
    atomwise_regularization_parameter: float,
    soap_dict: dict,
    suffix: str,
    displacement: float = 0.01,
):
    """
    Compute phonon band-structure benchmark metrics and generate associated plots.

    Parameters
    ----------
    ml_model: str
        ML model to be used. Default is GAP.
    structure : .Structure
        A structure object.
    mp_id:
        Materials Project ID.
    ml_phonon_bs: PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object.
    dft_phonon_bs: PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object.
    ml_imag_modes: bool
        Whether the ML-based phonon band structure shows imaginary modes.
    dft_imag_modes: bool
        Whether the DFT-based phonon band structure shows imaginary modes.
    displacement: float
        Displacement distance for phonons
    atomwise_regularization_parameter: float
        Regularization value for the atom-wise force components.
    suffix: str
        GAP potential file suffix.
    soap_dict: dict
        Dictionary containing SOAP parameters.


    Returns
    -------
    dict including
       Overall root mean squared error between DFT and ML phonon band-structure.
    """
    # might fail if band structures are not the same
    # TODO: Robust alternative would be a mesh computation
    try:
        # compute overall root mean squared error
        overall_rmse = get_rmse(ml_bs=ml_phonon_bs, dft_bs=dft_phonon_bs)

        # saves rmse k-dependent plot
        file_name = f"{structure.composition.reduced_formula}_rmse_phonons.pdf"
        _ = rmse_qdep_plot(
            ml_bs=ml_phonon_bs,
            dft_bs=dft_phonon_bs,
            which_q_path=2,
            file_name=file_name,
            img_format="pdf",
        )

        # saves DFT and ML phonon band-structure overlay plot
        file_name = f"{structure.composition.reduced_formula}_band_comparison.pdf"
        _ = compare_plot(
            ml_model=ml_model,
            ml_bs=ml_phonon_bs,
            dft_bs=dft_phonon_bs,
            file_name=file_name,
        )
    except ValueError:
        overall_rmse = None
    return {
        "benchmark_phonon_rmse": overall_rmse,
        "dft_imaginary_modes": dft_imag_modes,
        "ml_imaginary_modes": ml_imag_modes,
        "ml_model": ml_model,
        "mp_id": mp_id,
        "structure": structure,
        "displacement": displacement,
        "atomwise_regularization_parameter": atomwise_regularization_parameter,
        "soap_dict": soap_dict,
        "suffix": suffix,
    }


def get_rmse(
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    q_dependent_rmse: bool = False,
) -> float | list[float]:
    """
    Compute root mean squared error (rmse) between DFT and ML phonon band-structure.

    Parameters
    ----------
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object
    q_dependent_rmse : bool.
        If true, method returns k-dependent rmse between the band-structures

    Returns
    -------
    float or list[float]
      Root mean squared error between DFT and ML phonon band-structure
    """
    diff = np.sort(ml_bs.bands, axis=0) - np.sort(dft_bs.bands, axis=0)

    rmse = np.sqrt(np.mean(diff**2))

    if q_dependent_rmse:
        diff_here = np.transpose(diff)
        rmse = [np.sqrt(np.mean(diff_here[i] ** 2)) for i in range(len(diff_here))]

    return rmse


def rmse_qdep_plot(
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    which_q_path=1,
    file_name="rms.pdf",
    img_format="pdf",
) -> plt:
    """
    Save q dependent root mean squared error plot between DFT and ML phonon band-structure.

    Parameters
    ----------
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object.
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object.
    which_q_path : int.
        If set 1, use ML band-structure as reference, if 2, uses DFT band-structure as reference.
    file_name: str.
        Name of the saved plot
    img_format: str
        File extension of plot to be saved, default is pdf.

    Returns
    -------
    matplotlib.plt
        A matplotlib figure with q-dependent RMSE.
    """
    rmse_qp = get_rmse(ml_bs=ml_bs, dft_bs=dft_bs, q_dependent_rmse=True)
    if which_q_path == 1:
        plotter = PhononBSPlotter(bs=ml_bs)
    elif which_q_path == 2:
        plotter = PhononBSPlotter(bs=dft_bs)

    distances = []
    for element in plotter.bs_plot_data()["distances"]:
        distances.extend(element)

    plt.close("all")
    plt.plot(distances, rmse_qp)
    plt.xticks(
        ticks=plotter.bs_plot_data()["ticks"]["distance"],
        labels=plotter.bs_plot_data()["ticks"]["label"],
    )
    plt.xlabel("Wave vector")
    plt.ylabel("Phonons RMS (THz)")
    plt.savefig(file_name, format=img_format)

    return plt


def compare_plot(
    ml_model: str,
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    file_name: str = "band_comparison.pdf",
) -> Figure:
    """
    Save DFT and ML phonon band-structure overlay plot for visual comparison.

    Parameters
    ----------
    ml_model: str
        ML model to be used. Default is GAP.
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object
    file_name: str.
        Name of the saved plot

    Returns
    -------
    matplotlib.plt
        A matplotlib figure with DFT and ML generated phonon band-structure overlay
    """
    plotter = PhononBSPlotter(bs=ml_bs)
    plotter2 = PhononBSPlotter(bs=dft_bs)
    new_plotter = plotter.plot_compare(
        other_plotter={"DFT": plotter2}, self_label=ml_model
    )

    new_plotter.figure.savefig(fname=file_name)

    return new_plotter.figure
