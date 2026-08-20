# Iterative DFT vs MLIP benchmark workflow for phonons

We will not perform VASP calculations in realtime for this tutorial, but rather mock vasp runs.
Thus, it is necessary to set folders with pre-computed VASP output files for execution in the notebook.



```python
# Please note that I am reusing the same relaxations here in several steps.
# This is only to save storage on our repo. It has influence on the result.
ref_paths = {
    "tight relax 1_mp-117_0.94_pre1": "tutorial_data/tight_relax_1_mp-117_0.94_0_42",
    "tight relax 1_mp-117_0.94_0": "tutorial_data/tight_relax_1_mp-117_0.94_0_42",
    "tight relax 1_mp-117_0.94_1": "tutorial_data/tight_relax_1_mp-117_0.94_0_42",
    "dft tight relax 1_mp-117_0.94_0": "tutorial_data/tight_relax_1_mp-117_0.94_0_42",
    "tight relax 2_mp-117_0.94_pre1": "tutorial_data/tight_relax_2_mp-117_0.94_0_43",
    "tight relax 2_mp-117_0.94_0": "tutorial_data/tight_relax_2_mp-117_0.94_0_43",
    "tight relax 2_mp-117_0.94_1": "tutorial_data/tight_relax_2_mp-117_0.94_0_43",
    "dft tight relax 2_mp-117_0.94_0": "tutorial_data/tight_relax_2_mp-117_0.94_0_43",
    "dft static_mp-117_0.94_0": "tutorial_data/tight_relax_2_mp-117_0.94_0_43",
    "tight relax 1_mp-117_1.0_pre1": "tutorial_data/tight_relax_1_mp-117_1.0_0_47",
    "tight relax 1_mp-117_1.0_0": "tutorial_data/tight_relax_1_mp-117_1.0_0_47",
    "tight relax 1_mp-117_1.0_1": "tutorial_data/tight_relax_1_mp-117_1.0_0_47",
    "dft tight relax 1_mp-117_1.0_0": "tutorial_data/tight_relax_1_mp-117_1.0_0_47",
    "dft tight relax 1_mp-117_1.0_1": "tutorial_data/tight_relax_1_mp-117_1.0_0_47",
    "tight relax 2_mp-117_1.0_pre1": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "tight relax 2_mp-117_1.0_0": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "tight relax 2_mp-117_1.0_1": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "dft tight relax 2_mp-117_1.0_0": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "dft tight relax 2_mp-117_1.0_1": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "dft static_mp-117_1.0_0": "tutorial_data/tight_relax_2_mp-117_1.0_0_48",
    "tight relax 1_mp-117_1.06_pre1": "tutorial_data/tight_relax_1_mp-117_1.06_0_52",
    "tight relax 1_mp-117_1.06_0": "tutorial_data/tight_relax_1_mp-117_1.06_0_52",
    "tight relax 1_mp-117_1.06_1": "tutorial_data/tight_relax_1_mp-117_1.06_0_52",
    "dft tight relax 1_mp-117_1.06_0": "tutorial_data/tight_relax_1_mp-117_1.06_0_52",
    "dft tight relax 1_mp-117_1.06_1": "tutorial_data/tight_relax_1_mp-117_1.06_0_52",
    "tight relax 2_mp-117_1.06_pre1": "tutorial_data/tight_relax_2_mp-117_1.06_0_53",
    "tight relax 2_mp-117_1.06_0": "tutorial_data/tight_relax_2_mp-117_1.06_0_53",
    "tight relax 2_mp-117_1.06_1": "tutorial_data/tight_relax_2_mp-117_1.06_0_53",
    "dft tight relax 2_mp-117_1.06_0": "tutorial_data/tight_relax_2_mp-117_1.06_0_53",
    "dft static_mp-117_1.06_0": "tutorial_data/tight_relax_2_mp-117_1.06_0_53",
    "Sn-stat_iso_atom_0": "tutorial_data/Sn-stat_iso_atom_0_25",
    "Sn-stat_iso_atom_1": "tutorial_data/Sn-stat_iso_atom_0_25",
    "dft rattle static 1/1_mp-117_0.94_0": "tutorial_data/dft_rattle_static_1_1_mp-117_0.94_0_63",
    "dft rattle static 1/1_mp-117_0.94_1": "tutorial_data/dft_rattle_static_1_1_mp-117_0.94_1_257",
    "dft rattle static 1/1_mp-117_1.0_0": "tutorial_data/dft_rattle_static_1_1_mp-117_1.0_0_65",
    "dft rattle static 1/1_mp-117_1.0_1": "tutorial_data/dft_rattle_static_1_1_mp-117_1.0_1_255",
    "dft rattle static 1/1_mp-117_1.06_0": "tutorial_data/dft_rattle_static_1_1_mp-117_1.06_0_67",
    "dft rattle static 1/1_mp-117_1.06_1": "tutorial_data/dft_rattle_static_1_1_mp-117_1.06_1_259",
    "dft phonon static 1/1_mp-117_0.94_0": "tutorial_data/dft_phonon_static_1_1_mp-117_0.94_0_193",
    "dft phonon static 1/1_mp-117_1.0_0": "tutorial_data/dft_phonon_static_1_1_mp-117_1.0_0_189",
    "dft phonon static 1/1_mp-117_1.06_0": "tutorial_data/dft_phonon_static_1_1_mp-117_1.06_0_191",
}
```


```python
import warnings

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator
from jobflow import Flow, run_locally
from mock_vasp import TEST_DIR, mock_vasp
from pymatgen.core.structure import Structure

from autoplex.auto.phonons.flows import (
    CompleteDFTvsMLBenchmarkWorkflow,
    IterativeCompleteDFTvsMLBenchmarkWorkflow,
)

warnings.filterwarnings("ignore")
```

First, we define all relevant Makers for the workflow, used to train and finetune ML potentials for phonons. We will pre-relax the structures before starting the workflow.
We will now define the relax maker, a displacement maker (static maker), static energy maker
and a static energy maker for isolated atoms.


```python
phonon_bulk_relax_maker = DoubleRelaxMaker.from_relax_maker(
    TightRelaxMaker(
        run_vasp_kwargs={"handlers": ()},
        input_set_generator=TightRelaxSetGenerator(
            user_incar_settings={
                "GGA": "PE",
                "ISPIN": 1,
                "KSPACING": 0.1,
                "ALGO": "Normal",
                "LAECHG": False,
                "ISMEAR": 1,
                "ENCUT": 700,
                "IBRION": 1,
                "ISYM": 0,
                "SIGMA": 0.05,
                "LCHARG": False,
                "LWAVE": False,
                "LVTOT": False,
                "LORBIT": None,
                "LOPTICS": False,
                "LREAL": False,
                "ISIF": 4,
                "NPAR": 4,
            }
        ),
    )
)
```


```python
phonon_displacement_maker = PhononDisplacementMaker(
    name="dft phonon static",
    run_vasp_kwargs={"handlers": ()},
    input_set_generator=StaticSetGenerator(
        user_incar_settings={
            "GGA": "PE",
            "IBRION": -1,
            "ISPIN": 1,
            "ISMEAR": 1,
            "ISIF": 3,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "LREAL": False,
            "ALGO": "Normal",
            "NSW": 0,
            "LCHARG": False,
            "LWAVE": False,
            "LVTOT": False,
            "LORBIT": None,
            "LOPTICS": False,
            "SIGMA": 0.05,
            "ISYM": 0,
            "KSPACING": 0.1,
            "NPAR": 4,
        },
        auto_ispin=False,
    ),
)
```


```python
phonon_static_energy_maker = phonon_displacement_maker

static_isolated_atom_maker = StaticMaker(
    run_vasp_kwargs={"handlers": ()},
    input_set_generator=StaticSetGenerator(
        user_kpoints_settings={"reciprocal_density": 1},
        user_incar_settings={
            "GGA": "PE",
            "ALGO": "Normal",
            "ISPIN": 1,
            "LAECHG": False,
            "ISMEAR": 0,
            "LCHARG": False,
            "LWAVE": False,
            "LVTOT": False,
            "LORBIT": None,
            "LOPTICS": False,
            "NPAR": 4,
        },
    ),
)
```

First, collect a number of structures and then optimize them in advance of the workflow. One can also perform subsequent optimizations with different k-point settings, for example.


```python
job_list = []


structure_list = []
benchmark_structure_list = []
start_mpids = ["mp-117"]
start_poscars = [TEST_DIR / "tutorial_data/POSCAR-mp-117"]

mpids = []
for mpid, start_poscar in zip(start_mpids, start_poscars):
    for scale in [0.94, 1.0, 1.06]:
        structure = Structure.from_file(start_poscar)
        volume = structure.copy().volume
        structure = structure.scale_lattice((scale**3) * volume)  # added the cube
        job_opt = phonon_bulk_relax_maker.make(structure)
        job_opt.append_name("_" + mpid + "_" + str(scale) + "_pre1")
        job_list.append(job_opt)
        structure_list.append(job_opt.output.structure)
        mpids.append(mpid + "_" + str(scale))


mpbenchmark = mpids
benchmark_structure_list = structure_list
```


```python
iteration_flow = IterativeCompleteDFTvsMLBenchmarkWorkflow(
    max_iterations=2, # with the current test data, you can switch between 1 and 2
    rms_max=0.2,
    complete_dft_vs_ml_benchmark_workflow_0=CompleteDFTvsMLBenchmarkWorkflow(
        symprec=1e-3,
        apply_data_preprocessing=True,
        add_dft_rattled_struct=True,
        add_dft_phonon_struct=True,
        volume_custom_scale_factors=[1.0],
        rattle_type=0,
        distort_type=0,
        rattle_std=0.1,  #
        benchmark_kwargs={"relax_maker_kwargs": {"relax_cell": False}},
        supercell_settings={
            "min_length": 10,
            "max_length": 15,
            "min_atoms": 10,
            "max_atoms": 300,
            "fallback_min_length": 9,
        },
        # settings that worked with a GAP
        split_ratio=0.33,
        regularization=False,
        separated=False,
        num_processes_fit=48,
        displacement_maker=phonon_displacement_maker,
        phonon_bulk_relax_maker=phonon_bulk_relax_maker,
        phonon_static_energy_maker=phonon_static_energy_maker,
        rattled_bulk_relax_maker=phonon_bulk_relax_maker,
        isolated_atom_maker=static_isolated_atom_maker,
    ),
    complete_dft_vs_ml_benchmark_workflow_1=CompleteDFTvsMLBenchmarkWorkflow(
        symprec=1e-3,
        apply_data_preprocessing=True,
        add_dft_phonon_struct=False,
        add_dft_rattled_struct=True,
        volume_custom_scale_factors=[1.0],
        rattle_type=0,
        distort_type=0,
        rattle_std=0.1,  # maybe 0.1
        benchmark_kwargs={"relax_maker_kwargs": {"relax_cell": False}},
        supercell_settings={
            "min_length": 10,
            "max_length": 15,
            "min_atoms": 10,
            "max_atoms": 300,
            "fallback_min_length": 9,
        },
        # settings that worked with a GAP
        split_ratio=0.33,
        regularization=False,
        separated=False,
        num_processes_fit=48,
        displacement_maker=phonon_displacement_maker,
        phonon_bulk_relax_maker=phonon_bulk_relax_maker,
        phonon_static_energy_maker=phonon_static_energy_maker,
        rattled_bulk_relax_maker=phonon_bulk_relax_maker,
        isolated_atom_maker=static_isolated_atom_maker,
    ),
).make(
    structure_list=structure_list,
    mp_ids=mpids,
    benchmark_structures=benchmark_structure_list,
    benchmark_mp_ids=mpbenchmark,
    rattle_seed=0,
    fit_kwargs_list=[
        {
            "soap": {
                "delta": 1.0,
                "l_max": 12,
                "n_max": 10,
                "atom_sigma": 0.5,
                "zeta": 4,
                "cutoff": 5.0,
                "cutoff_transition_width": 1.0,
                "central_weight": 1.0,
                "n_sparse": 6000,
                "f0": 0.0,
                "covariance_type": "dot_product",
                "sparse_method": "cur_points",
            },
            "general": {
                "two_body": True,
                "three_body": False,
                "soap": True,
                "default_sigma": "{0.001 0.05 0.05 0.0}",
                "sparse_jitter": 1.0e-8,
            },
        }
    ],
)

job_list.append(iteration_flow)
autoplex_flow = Flow(jobs=job_list, output=iteration_flow.output)
```

Now, we are mocking the VASP execution. If you would like leave all folders from the run, set `clean_folders` to False!


```python
with mock_vasp(ref_paths=ref_paths, clean_folders=False) as mf:
    run_locally(
        autoplex_flow,
        create_folders=True,
        ensure_success=True,
        raise_immediately=True,
    )
```
