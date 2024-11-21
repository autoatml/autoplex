import json
import pytest
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_ph_bs_symm_line
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from autoplex.benchmark.phonons.utils import compute_bandstructure_benchmark_metrics


def test_compute_bandstructure_benchmark_metrics_dummy(test_dir, clean_dir):
    import os
    from pathlib import Path

    # test if same band structure gives out expected 0.0 rmse
    dummy_bs_file_path = test_dir / "benchmark" / "NaCl.json"

    with open(dummy_bs_file_path, "r") as file:
        dummy_bs_dict = json.load(file)

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")
    df_bs = PhononBandStructureSymmLine.from_dict(dummy_bs_dict)
    ml_bs = PhononBandStructureSymmLine.from_dict(dummy_bs_dict)

    result = compute_bandstructure_benchmark_metrics(
        ml_model="GAP", structure=df_bs.structure, ml_phonon_bs=ml_bs, dft_phonon_bs=df_bs, ml_imag_modes=False,
        dft_imag_modes=False, mp_id="mp-22905", atomwise_regularization_parameter=0.01, soap_dict={}, suffix=""
    )

    assert result["benchmark_phonon_rmse"] == pytest.approx(0.0)

    # get list of generated plot files
    test_files_dir = Path(test_dir / "benchmark").resolve()
    path_to_plot_files = list(test_files_dir.glob("NaCl*.pdf"))

    # ensure two plots are generated
    assert len(path_to_plot_files) == 2

    # remove the plot files from directory
    for file in path_to_plot_files:
        file.unlink()

    os.chdir(parent_dir)


def test_compute_bandstructure_benchmark_metrics(test_dir, clean_dir):
    import os
    from pathlib import Path

    # test wih two different band-structures
    dft_bs_file_path = test_dir / "benchmark" / "DFT_phonon_band_structure.yaml"
    ml_bs_file_path = test_dir / "benchmark" / "GAP_phonon_band_structure.yaml"

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")
    df_bs = get_ph_bs_symm_line(bands_path=dft_bs_file_path)
    ml_bs = get_ph_bs_symm_line(bands_path=ml_bs_file_path)

    result = compute_bandstructure_benchmark_metrics(
        ml_model="GAP", structure=df_bs.structure, ml_phonon_bs=ml_bs, dft_phonon_bs=df_bs, ml_imag_modes=False,
        dft_imag_modes=False, mp_id="mp-22905", atomwise_regularization_parameter=0.01, soap_dict={}, suffix=""
    )

    assert result["benchmark_phonon_rmse"] == pytest.approx(
        0.5716963823412201, abs=0.3  # TODO check results
    )

    # get list of generated plot files
    test_files_dir = Path(test_dir / "benchmark").resolve()
    path_to_plot_files = list(test_files_dir.glob("LiCl*.pdf"))

    # ensure two plots are generated
    assert len(path_to_plot_files) == 2
    # remove the plot files from directory
    for file in path_to_plot_files:
        file.unlink()

    os.chdir(parent_dir)


def test_write_benchmark_metrics(test_dir, clean_dir):
    import os
    from jobflow import run_locally

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")

    path_to_struct = test_dir / "benchmark" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    metric_vals = [
        [{'benchmark_phonon_rmse': 0.87425, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False},
         {'benchmark_phonon_rmse': 0.63839, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}],
        [{'benchmark_phonon_rmse': 0.55506, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}],
        [{'benchmark_phonon_rmse': 0.43216, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}],
        [{'benchmark_phonon_rmse': 0.54584, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}],
        [{'benchmark_phonon_rmse': 0.43216, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}],
        [{'benchmark_phonon_rmse': 0.36478, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False},
         {'benchmark_phonon_rmse': 0.38100, 'dft_imaginary_modes': False, 'ml_imaginary_modes': False}]
    ]

    soap_dict = [  # unit tests for checking correct default soap_dict in tests/auto/test_auto_flows.py
        {'n_sparse': 3000, 'delta': 1.0},
        {'n_sparse': 4000, 'delta': 1.0},
        {'n_sparse': 5000, 'delta': 1.0},
        {'n_sparse': 6000, 'delta': 1.0},
        {'n_sparse': 6000, 'delta': 1.0},
        {'n_sparse': 3000, 'delta': 1.0},
        {'n_sparse': 5000, 'delta': 1.0},
        {'n_sparse': 6000, 'delta': 1.0},
    ]

    suffixes = ["", '_wo_sigma', '_phonon', '_rand_struc']

    metrics = []

    suffix_index = 0

    for metric_group in metric_vals:
        for metric, soap in zip(metric_group, soap_dict):
            fused_dict = {
                'benchmark_phonon_rmse': metric['benchmark_phonon_rmse'],
                'dft_imaginary_modes': metric['dft_imaginary_modes'],
                'ml_imaginary_modes': metric['ml_imaginary_modes'],
                'ml_model': 'GAP',
                'mp_id': 'mp-22905',
                'structure': structure,
                'displacement': 0.01,
                'atomwise_regularization_parameter': 0.1,
                'soap_dict': soap,
                'suffix': suffixes[suffix_index]
            }
            metrics.append(fused_dict)
        suffix_index = min(suffix_index + 1, len(suffixes) - 1)

    write_metrics_job = write_benchmark_metrics(
        benchmark_structures=[structure],
        metrics=[metrics],
    )

    run_locally(write_metrics_job, create_folders=False, ensure_success=True)

    # get list of generated txt file
    path_to_ref_txt_file = test_dir / "benchmark" / "results_LiCl_ref.txt"
    path_to_txt_file = test_dir / "benchmark" / "results_LiCl.txt"

    with open(path_to_ref_txt_file, "r") as ref:
        ref_txt = ref.readlines()

    with open(path_to_txt_file, "r") as org:
        org_txt = org.readlines()

    assert ref_txt == org_txt

    # remove generated file as part of test
    path_to_txt_file.unlink()

    os.chdir(parent_dir)
