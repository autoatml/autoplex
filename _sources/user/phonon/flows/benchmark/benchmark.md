(benchmark)=

# Benchmark of ML-based phonon structure

This tutorial will help you understand all the `autoplex` benchmark specifications.

## General settings

For the benchmark, you do not have to worry about a lot of settings. The crucial part here is the number of benchmark structures you are interested in.
All benchmark harmonic phonon runs will always be generated with a displacement of 0.01 even though the fitting procedure can also include different displacements.

```python
from mp_api.client import MPRester
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.serialization import loadfn

mpr = MPRester(api_key='YOUR_MP_API_KEY')
structure_list = []
benchmark_structure_list = []
mpids = ["mp-22905"]
mpbenchmark = ["mp-22905"]
dft_data = loadfn("/path/to/DFT/ref/data/PhononBSDOSDoc_mp_22905.json")
dft_reference: PhononBSDOSDoc = dft_data["output"]
for mpid in mpids:
    structure = mpr.get_structure_by_material_id(mpid)
    structure_list.append(structure)
for mpbm in mpbenchmark:
    bm_structure = mpr.get_structure_by_material_id(mpbm)
    benchmark_structure_list.append(bm_structure)

complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    apply_data_preprocessing=True,
).make(
    structure_list=structure_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark, 
    dft_references=[dft_reference])
```
In case you have pre-existing DFT calculations, you can pass them as a list via the `dft_references` parameters. Make sure, it is the same order as in the benchmark MP-IDs and structures.
It is important to provide the pre-existing DFT data in form of a `PhononBSDOSDoc` task document object (from `atomate2`). Without any DFT reference calculations given, `autoplex` will automatically execute the VASP calculations. A mix of pre-existing and missing DFT references is not supported.

## Error metrics
`autoplex` automatically provides you with a phonon bandstructure comparison plot, a q-point wise RMSE plot and an overall RMSE value (`results_XY.txt` file). For examples see [here](../flows.md#output-and-results).

## Run a benchmark with a pre-existing DFT calculation and GAP potential

If you want to run or repeat the benchmark with your pre-existing DFT calculation and your pre-existing GAP potential, you can use the following Python script as a template.
It is important to provide the pre-existing DFT data in form of a `PhononBSDOSDoc` task document object (from `atomate2`).

```python
#!/usr/bin/env python

import os
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from mp_api.client import MPRester
from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from monty.serialization import loadfn
from jobflow import SETTINGS
from jobflow import run_locally

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

store = SETTINGS.JOB_STORE
# connect to the job store
store.connect()

mpr = MPRester(api_key = 'YOUR_API_KEY')
mpid = "mp-22905"
structure =  mpr.get_structure_by_material_id(mpid)
dft_data = loadfn("/path/to/DFT/ref/data/PhononBSDOSDoc_mp_22905.json")
dft_reference: PhononBSDOSDoc = dft_data["output"]
potential_filename = "/path/to/GAP/file/gap_file.xml"

phojob = PhononMaker(
        bulk_relax_maker=ForceFieldRelaxMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename}, 
        relax_cell=True, relax_kwargs={"interval": 500, "fmax": 0.00001}, steps=10000,
        force_field_name="GAP",
),
        phonon_displacement_maker=ForceFieldStaticMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename},
        force_field_name="GAP",
),
        static_energy_maker=ForceFieldStaticMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename}),
        store_force_constants=False, min_length=18,
        generate_frequencies_eigenvectors_kwargs={"units": "THz"}, force_field_name="GAP",
).make(structure=structure)
        
bm = PhononBenchmarkMaker(name="Benchmark").make(
    structure=structure, benchmark_mp_id = "mp-22905",
    displacement=0.01, atomwise_regularization_parameter=0.1,
    soap_dict={'n_sparse': 6000, 'delta': 0.5}, suffix="",  # exemplary values
    ml_phonon_task_doc = phojob.output, dft_phonon_task_doc = dft_reference)

comp_bm = write_benchmark_metrics(
            benchmark_structures=[structure],
            metrics=[[bm.output]],
        )

run_locally([phojob, bm, comp_bm], create_folders=True, store=store)
```
If you use another [`ForceFieldRelaxMaker` and `ForceFieldStaticMaker`](https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/jobs.py), you can switch from GAP to one of the other 
[MLIP potentials](../fitting/fitting.md#fitting-phonon-accurate-potentials). 

You can extract a JSON file containing your pre-existing VASP DFT run from your MongoDB with the following script:
```python
from jobflow import SETTINGS
from monty.json import jsanitize
from monty.serialization import dumpfn

store = SETTINGS.JOB_STORE
store.connect()

result = store.query( {'name': 'generate_frequencies_eigenvectors'}, load=True)
phononbsdosdoc = store.query({'uuid': 'put the MongoDB UUID here'}, load=True)
for i in phononbsdosdoc:
    del i["_id"]
    monty_encoded_json_doc = jsanitize(i, allow_bson=True, strict=True, enum_values=True)
    
    dumpfn(monty_encoded_json_doc, 'PhononBSDOSDoc_mp_22905.json')
```

And check if it contains the correct output with:
```python
from monty.serialization import loadfn

data = loadfn('PhononBSDOSDoc_mp_22905.json')
data['output'].structure  
```

Your output for `structure` should look like:
```bash
Structure Summary
Lattice
    abc : 5.061019144638489 5.061019144638489 5.061019144638489
 angles : 90.0 90.0 90.0
 volume : 129.63251308285152
      A : 5.061019144638489 -0.0 3e-16
      B : 8e-16 5.061019144638489 3e-16
      C : 0.0 -0.0 5.061019144638489
    pbc : True True True
PeriodicSite: Li (0.0, 0.0, 0.0) [0.0, 0.0, 0.0]
PeriodicSite: Li (4e-16, 2.531, 2.531) [0.0, 0.5, 0.5]
PeriodicSite: Li (2.531, 0.0, 2.531) [0.5, 0.0, 0.5]
PeriodicSite: Li (2.531, 2.531, 3e-16) [0.5, 0.5, 0.0]
PeriodicSite: Cl (2.531, 0.0, 1.5e-16) [0.5, 0.0, 0.0]
PeriodicSite: Cl (2.531, 2.531, 2.531) [0.5, 0.5, 0.5]
PeriodicSite: Cl (0.0, 0.0, 2.531) [0.0, 0.0, 0.5]
PeriodicSite: Cl (4e-16, 2.531, 1.5e-16) [0.0, 0.5, 0.0]
```


