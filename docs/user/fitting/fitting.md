(fitting)=

*Tutorials written by Christina Ertural ([christina.ertural@bam.de](mailto:christina.ertural@bam.de)).*

# Fitting potentials

This tutorial will show you how to control the MLIP fit settings with the `autoplex` workflow. 
The choice of the correct fit setup and hyperparameter settings has a significant influence on the final result.
Please note that the fitting might need nodes with very large memory requirements (1 TB) in some cases.

## General settings

There are two categories of fit settings that you can change. The first type concerns the general fit setup, 
that will affect the fit regardless of the chosen MLIP method, and e.g. changes database specific settings 
(like the split-up into training and test data). The other type of settings influences the MLIP specific setup 
like e.g. the choice of hyperparameters.

In case of the general settings, you can pass the MLIP model you want to use with the `ml_models` parameter list.
You can set the maximum force threshold `f_max` for filtering the data ("distillation") in the MLIP fit preprocess step.
In principle, the distillation step can be turned off by passing `"distillation": False` in the `fit_kwargs` keyword arguments,
but it is strongly advised to filter out too high force data points.
The hyperparameters and further parameters can be passed in the `make` call (see below) or using `fit_kwargs` (or `**{...}`),
like e.g. you can set the `split_ratio` to split the database up into a training and a test set,
or adjust the number of processes `num_processes_fit`.
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["GAP", "MACE"], ...,
).make(..., preprocessing_data=True,
    f_max=40.0,
    fit_kwargs={
        "split_ratio": 0.4,
        "num_processes_fit": 48,
    },
    ...  # put the other hyperparameter commands here as shown below
)
```

The MLIP model specific settings and hyperparameters setup varies from model to model and is demonstrated in the next 
sections. 
> ℹ️ Note that `autoplex` provides the most comprehensive features for **GAP**, and more features for the other models will 
follow in future versions.  

## GAP

There are several overall settings for the GAP fit that will change the mode in which `autoplex` runs.
When `hyper_para_loop` is set to `True`, `autoplex` wil automatically iterate through a set of several hyperparameters 
(`atomwise_regularization_list`, `soap_delta_list` and `n_sparse_list`) and repeat the GAP fit for each combination.
More information on the atom-wise regularization parameter can be found in [J. Chem. Phys. 153, 044104 (2020)](https://pubs.aip.org/aip/jcp/article/153/4/044104/1056348/Combining-phonon-accuracy-with-high) 
and a comprehensive list GAP hyperparameters can be found in the [QUIP/GAP user guide](https://libatoms.github.io/GAP/gap_fit.html#command-line-example).
The other keywords to change `autoplex`'s mode are `glue_xml` (use glue.xml core potential instead of 2b/3b terms), 
`regularization` (use a sigma regularization) and `separated` (repeat the GAP fit for the combined database and each 
separated subset).
The parameter `atom_wise_regularization` can turn the atom-wise regularization on and off, 
`atomwise_regularization_parameter` is the value that shall be set and `f_min` is the lower bound cutoff of forces 
taken into account for the atom-wise regularization or otherwise be replaced by the f_min value.
`auto_delta` let's you decide if you want to pass a fixed delta value for the 2b, 3b and SOAP terms or let `autoplex` 
automatically determine a suitable delta value based on the database's energies.
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["GAP"], ...,
    hyper_para_loop=True, 
    atomwise_regularization_list=[0.01, 0.1], 
    soap_delta_list=[0.5, 1.0, 1.5], 
    n_sparse_list=[1000, 3000, 6000, 9000]
).make(..., 
    structure_list=[structure],
    mp_ids=["mpid"],
    benchmark_mp_ids=["mpid"],
    benchmark_structures=[structure],
    preprocessing_data=True,
    atom_wise_regularization=True, 
    atomwise_regularization_parameter=0.1, 
    f_min=0.01, 
    auto_delta=False,
    ...,
    **{...,
     "glue_xml": False,
     "regularization": False,
     "separated": False,
     "general": {"default_sigma": "{0.001 0.05 0.05 0.0}", {"two_body": True, "three_body": False,"soap": False},...},
     "twob": {"cutoff": 5.0,...},
     "threeb": {"cutoff": 3.25,...},
     "soap": {"delta": 1.0, "l_max": 12, "n_max": 10,...},
    }
)
```
`autoplex` provides a JSON dict file containing default GAP fit settings in 
*autoplex/fitting/common/gap-defaults.json*, 
that can be overwritten using the fit keyword arguments as demonstrated in the code snippet.

`autoplex` follows a certain convention for naming files and labelling the data 
(see *autoplex/fitting/common/gap-defaults.json*).
```json
  "general": {
    "at_file": "train.extxyz",
    "energy_parameter_name": "REF_energy",
    "force_parameter_name": "REF_forces",
    "virial_parameter_name": "REF_virial",
    "gp_file": "gap_file.xml"
  },
```
You can either adapt to the `autoplex` conventions or change by passing your preferred names and label to the fit keyword arguments.



## ACE

For fitting and validating ACE potentials, one needs to install **julia** as `autoplex` relies on 
[ACEpotentials.jl](https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/) which support fitting of linear ACE. Currently no python package exists for the same.

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["J-ACE"], ...,
).make(..., 
    structure_list=[structure],
    mp_ids=["mpid"],
    benchmark_mp_ids=["mpid"],
    benchmark_structures=[structure],
    preprocessing_data=True,
    order=3,
    totaldegree=6,
    cutoff=2.0,
    solver="BLR",
    ...)
```
The ACE fit hyperparameters can be passed in the `make` call with its distinct commands. 
Because there is no respective MLPhononMaker in `atomate2` for J-ACE, the functionalities for `autoplex` are limited to
the data generation and ML fit in this case.

## Nequip

The Nequip fit procedure can be controlled by fit hyperparameters in the `make` call.

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["Nequip"], ...,
).make(...,
    structure_list=[structure],
    mp_ids=["mpid"],
    benchmark_mp_ids=["mpid"],
    benchmark_structures=[structure],
    preprocessing_data=True,
    r_max=4.0,
    num_layers=4,
    l_max=2,
    num_features=32,
    num_basis=8,
    invariant_layers=2,
    invariant_neurons=64,
    batch_size=5,
    learning_rate=0.005,
    max_epochs=10000,  
    default_dtype="float32",
    device="cuda",
    ...
)
```

## M3GNet

In a similar way, the M3GNet fit hyperparameters can be passed using `make` as well.

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["M3GNet"], ...,
).make(...,
    structure_list=[structure],
    mp_ids=["mpid"],
    benchmark_mp_ids=["mpid"],
    benchmark_structures=[structure],
    preprocessing_data=True,
    cutoff=5.0,
    threebody_cutoff=4.0,
    batch_size=10,
    max_epochs=1000,
    include_stresses=True,
    hidden_dim=128,
    num_units=128,
    max_l=4,
    max_n=4,
    device="cuda",
    test_equal_to_val=True,
    ...,
    )
```

## MACE

Here again, you can pass the MACE fit hyperparameters to `make`.

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ml_models=["MACE"], ...,
).make(...,
    structure_list=[structure],
    mp_ids=["mpid"],
    benchmark_mp_ids=["mpid"],
    benchmark_structures=[structure],
    preprocessing_data=True,
    model="MACE",
    config_type_weights='{"Default":1.0}',
    hidden_irreps="128x0e + 128x1o",
    r_max=5.0,
    batch_size=10,
    max_num_epochs=1500,
    start_swa=1200,
    ema_decay=0.99,
    correlation=3,
    loss="huber",
    default_dtype="float32",
    device="cuda",
    ...
)
```
### Finetuning MACE-MP-0

It is also possible to finetune MACE-MP-0. To do so, you need to install MACE-torch 0.3.7. 
Currently, this can only be done by cloning the git-repo and installing it from there: []()

It is now important that you switch off the default settings for the fitting procedure (use_defaults_fitting=False).
If one wants to perform very low-data fine-tuning (less than 4 additional structures), please also think about switching off `preprocessing_data` 
as this will disable the test/train split. 

The following workflow `CompleteDFTvsMLBenchmarkWorkflowMPSettings` uses Materials Project default settings slightly adapted to phonon runs (more accurate convergence, ALGO=Normal).
It can also be used without finetuning option. To finetune optimally, please adapt the MACE fitting parameters yourself.

```python
complete_workflow_mace = CompleteDFTvsMLBenchmarkWorkflowMPSettings(
        ml_models=["MACE"],
        volume_custom_scale_factors=[0.95,1.00,1.05], rattle_type=0, distort_type=0,
        ...
    ).make(
        structure_list=[structure],
        mp_ids=["mpid"],
        benchmark_mp_ids=["mpid"],
        benchmark_structures=[structure],
        preprocessing_data=True,
        use_defaults_fitting=False,
        model="MACE",
        name="MACE_final",
        foundation_model="large",
        multiheads_finetuning=False,
        r_max=6,
        loss="huber",
        energy_weight=1000.0,
        forces_weight=1000.0,
        stress_weight=1.0,
        compute_stress=True,
        E0s="average",
        scaling="rms_forces_scaling",
        batch_size=1,
        max_num_epochs=200,
        ema=True,
        ema_decay=0.99,
        amsgrad=True,
        default_dtype="float64",
        restart_latest=True,
        lr=0.0001,
        patience=20,
        device="cpu",
        save_cpu=True,
        seed=3,
    )
    


## Example script for `autoplex` workflow using GAP to fit and benchmark a Si database

The following code snippet will demonstrate, how you can submit an `autoplex` workflow for an automated SOAP-only GAP fit 
and DFT benchmark for a Si allotrope database. The GAP fit parameters are taken from [J. Chem. Phys. 153, 044104 (2020)](https://pubs.aip.org/aip/jcp/article/153/4/044104/1056348/Combining-phonon-accuracy-with-high).
In this example we will also use `hyper_para_loop=True` to loop through a set of given GAP fit convergence parameter 
and hyperparameters set as provided by the lists `atomwise_regularization_list`, `soap_delta_list` and `n_sparse_list`.
In this example script, we are using `jobflow_remote` to submit the jobs to a remote cluster.

```python
from jobflow_remote import submit_flow
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow
from mp_api.client import MPRester

mpr = MPRester(api_key='YOUR_MP_API_KEY')
struc_list = []
benchmark_structure_list = []
mpids = ["mp-149"]  # add all the Si structure mpids you are interested in
mpbenchmark = ["mp-149"]  # add all the Si structure mpids you are interested in
for mpid in mpids:
    struc = mpr.get_structure_by_material_id(mpid)
    struc_list.append(struc)
for mpbm in mpbenchmark:
    bm_struc = mpr.get_structure_by_material_id(mpbm)
    benchmark_structure_list.append(bm_struc)

autoplex_flow = CompleteDFTvsMLBenchmarkWorkflow(
    n_structures=50, symprec=0.1,
    volume_scale_factor_range=[0.95, 1.05], rattle_type=0, distort_type=0,
    hyper_para_loop=True, atomwise_regularization_list=[0.1, 0.01],
    soap_delta_list=[0.5], n_sparse_list=[7000, 8000, 9000]).make(
    structure_list=struc_list, mp_ids=mpids, benchmark_structures=benchmark_structure_list,
    benchmark_mp_ids=mpbenchmark, preprocessing_data=True,
    **{
        "split_ratio": 0.33,
        "regularization": False,
        "separated": True,
        "num_processes_fit": 48,
        "GAP": {"soap": {"delta": 1.0, "l_max": 12, "n_max": 10,
                         "atom_sigma": 0.5, "zeta": 4, "cutoff": 5.0,
                         "cutoff_transition_width": 1.0,
                         "central_weight": 1.0, "n_sparse": 9000, "f0": 0.0,
                         "covariance_type": "dot_product",
                         "sparse_method": "cur_points"},
                "general": {"two_body": False, "three_body": False, "soap": True,
                            "default_sigma": "{0.001 0.05 0.05 0.0}", "sparse_jitter": 1.0e-8, }}},
)

autoplex_flow.name = "autoplex_wf"

resources = {...}

print(submit_flow(autoplex_flow, worker="autoplex_worker", resources=resources, project="autoplex"))
```


## Running a MLIP fit only

The following script shows an example of how you can run a sole GAP fit with `autoplex` using `run_locally` from 
`jobflow` for the job management.

```python
#!/usr/bin/env python

from jobflow import SETTINGS
from autoplex.fitting.common.flows import MLIPFitMaker
import os
from jobflow import run_locally

os.environ["OMP_NUM_THREADS"] = "48" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 

os.chdir("/path/to/destination/directory")
store = SETTINGS.JOB_STORE
# connect to the job store
store.connect()


fit_input_dict = {
        "mp-id": {  # put a mp-id or another kind of data marker here
            "rand_struc_dir": [[
                (
                    "/path/to/randomized/supercell/structure/calculation"
                ),
                (
                    "/path/to/randomized/supercell/structure/calculation"
                ),
            ]],
            "phonon_dir": [[
                (
                    "/path/to/phonopy/supercell/structure/calculation"
                ),
            ]],
            "phonon_data": [],
        },
        "isolated_atom": {"iso_atoms_dir": [[
                (
                    "/path/to/isolated/atom/calculation"
                ),
            ]]
        }
    }
    
    
mlip_fit = MLIPFitMaker(mlip_type="GAP", ...,).make(
        species_list=["Li", "Cl"],
        isolated_atoms_energy=[-0.28649227, -0.25638457],
        fit_input=fit_input_dict,
        pre_xyz_files=["vasp_ref.extxyz"],
        pre_database_dir="/path/to/pre_database",
        auto_delta = True,
        glue_xml = False,
        **{...}       
        )

run_locally(mlip_fit, create_folders=True, store=store)
```
Additional fit settings can again be passed using `fit_kwargs` or `**{...}`.

> ℹ️ Note that in the current setup of `autoplex`, you need to pass a `fit_input_dict` to the `MLIPFitMaker`
> containing at least one entry for "rand_struc_dir", "phonon_dir" and "isolated_atom" **VASP** calculations, 
> otherwise the code will not finish successfully.
            
