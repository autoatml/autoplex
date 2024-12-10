import pytest
from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker


def test_benchmark(test_dir, clean_dir):
    import os
    from pathlib import Path
    from jobflow import run_locally
    from monty.serialization import loadfn
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

    # test with two different band-structures

    dft_data = loadfn(test_dir / "benchmark" / "PhononBSDOSDoc_LiCl.json")
    ml_data = loadfn(test_dir / "benchmark" / "PhononBSDOSMLDoc_LiCl.json")
    dft_doc: PhononBSDOSDoc = dft_data["output"]
    ml_doc: PhononBSDOSDoc = ml_data["output"]

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")

    benchmark_flow = PhononBenchmarkMaker().make(
        ml_model="GAP",
        structure=dft_doc.structure,
        ml_phonon_task_doc=ml_doc,
        dft_phonon_task_doc=dft_doc,
        benchmark_mp_id="test",
        displacement=0.01,
        atomwise_regularization_parameter=0.01,
        soap_dict=None, suffix="",
    )

    responses = run_locally(benchmark_flow, create_folders=False, ensure_success=True)

    assert responses[benchmark_flow.output.uuid][1].output["benchmark_phonon_rmse"] == pytest.approx(0.03660647131610)

    # get list of generated plot files
    test_files_dir = Path(test_dir / "benchmark").resolve()
    path_to_plot_files = list(test_files_dir.glob("LiCl*.pdf"))

    # ensure two plots are generated
    assert len(path_to_plot_files) == 2
    # remove the plot files from directory
    for file in path_to_plot_files:
        file.unlink()

    os.chdir(parent_dir)
