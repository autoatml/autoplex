[![Testing Linux](https://github.com/JaGeo/autoplex/actions/workflows/python-package.yml/badge.svg)](https://github.com/JaGeo/autoplex/actions/workflows/python-package.yml)
# autoplex

`autoplex` is a software for generating and benchmarking machine learning (ML) based interatomic potentials. The aim of `autoplex` is to provide a fully automated solution for creating high-quality ML potentials. The software is interfaced to multiple different ML potential fitting frameworks and to the atomate2 and ase environment for efficient high-throughput computations. The vision of this project is to allow a wide community of researchers to create accurate and reliable ML potentials for materials simulations.

`autoplex` is developed jointly by two research groups at BAM Berlin and the University of Oxford.

`autoplex` is an evolving project and **contributions are very welcome**! To ensure that the code remains of high quality, please raise a pull request for any contributions, which will be reviewed before integration into the main branch of the code. In the beginning, Janine will take care of the reviews.

# Setup

In order to setup the mandatory prerequisites to be able to use `autoplex`, please follow the [installation guide of atomate2](https://materialsproject.github.io/atomate2/user/install.html).

After setting up `atomate2`, make sure to add `VASP_INCAR_UPDATES: {"NPAR": number}` in your ~/atomate2/config/atomate2.yaml file.
Set a number that is a divisor of the number of tasks you use for the VASP calculations.

# Installation
autoplex relies on the atomate2 and ase frameworks.

Developers installation: Navigate to autoplex and install it by `pip install -e .[docs,strict,dev]`.

Alternatively, the dependencies atomate2 and ase can be installed individually via
```bash
pip install atomate2[strict]

pip install ase@git+https://gitlab.com/ase/ase.git@aae51d57721847624cea569f3a2d4bb6aa5032b4
```

Additionally, to be able to fit and validate `ACEpotentials`, one also needs to install julia as autoplex relies on [ACEpotentials](https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/) which support fitting of linear ACE and currently no python package exists for the same.
Please run following commands to enable ACEpotentials fitting and functionality.

Install julia
`curl -fsSL https://install.julialang.org | sh`

Once installed, in terminal run `julia` and then run the following commands

```jl
using Pkg
Pkg.activate(".")
Pkg.Registry.add("General")  # only needed when installing Julia for the first time
Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add("ACEpotentials")
Pkg.add("DataFrames")
Pkg.add("CSV")
```

Additionally, `buildcell` as a part of `AIRSS` needs to be installed:

```bash
curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz; tar -xf airss-0.9.3.tgz; rm airss-0.9.3.tgz; cd airss; make ; make install ; make neat; cd ..
```

# Contributing guidelines

Please follow the [contributing guidelines](docs/dev/contributing.md)!

# Workflow overview

The following [Mermaid](https://mermaid.live/) diagram will give you an overview of the flows and jobs in the default autoplex workflow:
```mermaid
flowchart TD
    f831581e-1d20-4fa8-aa7d-773ae45a78aa(external) -->|VASP output files info| 25f1b412-6e80-4ea0-a669-126b1d2eefdc(data preprocessing)
    f831581e-1d20-4fa8-aa7d-773ae45a78aa(external) -->|VASP output files info| 75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark)
    e99258a7-6717-4cc9-b629-709bee881cfa(external) -->|'phonon calc dirs', 'phonon data'| 25f1b412-6e80-4ea0-a669-126b1d2eefdc(data preprocessing)
    e99258a7-6717-4cc9-b629-709bee881cfa(external) -->|'phonon calc dirs', 'phonon data'| 75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark)
    38349844-bee1-4869-839f-74ccd753524e(external) -->|'isolated atoms calc dirs'| 25f1b412-6e80-4ea0-a669-126b1d2eefdc(data preprocessing)
    38349844-bee1-4869-839f-74ccd753524e(external) -->|'energies', 'species'| 0a11a48c-3d9b-454a-9959-f7732967b49f(machine learning fit)
    38349844-bee1-4869-839f-74ccd753524e(external) -->|'isolated atoms calc dirs'| 75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark)
    25f1b412-6e80-4ea0-a669-126b1d2eefdc(data preprocessing) -->|formatted database| 0a11a48c-3d9b-454a-9959-f7732967b49f(machine learning fit)
    0a11a48c-3d9b-454a-9959-f7732967b49f(machine learning fit) -->|'path to MLIP file'| 75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark)
    75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark) -->|benchmark output info| d5b02fd6-806d-43f4-9f3f-d9de5f0f28e3(write benchmark metrics)
    subgraph 2bc86ca5-f4bd-47dc-aa9d-45f72d0ab527 [autoplex workflow]
        subgraph 821b6198-a8c5-45c5-939f-8ff0edd9f5b0 [data generation]
            f831581e-1d20-4fa8-aa7d-773ae45a78aa(rattled DFT structures)
        end
        subgraph 75368ebe-fe58-48a9-aeba-6e81ca9169d6 [data generation]
            e99258a7-6717-4cc9-b629-709bee881cfa(phonopy based structures)
        end
        38349844-bee1-4869-839f-74ccd753524e(isolated atoms calcs)
        subgraph cdcce0a3-83fe-4590-993c-0b6e3ff5adcb [fit ML potentials]
            25f1b412-6e80-4ea0-a669-126b1d2eefdc(data preprocessing)
            0a11a48c-3d9b-454a-9959-f7732967b49f(machine learning fit)
        end
        75cee155-2708-4dcf-b8b3-d184d450ed4f(benchmark)
        d5b02fd6-806d-43f4-9f3f-d9de5f0f28e3(write benchmark metrics)
    end
```
The workflow starts with three flows that generate data for our database:
* The first flow is preparing the VASP calculation for the isolated atoms (`isolated atoms calcs`).
* A second flow is preparing the `phonopy` calculations to collect the VASP data from the single-atom displaced supercells (`phonopy based structures`).
* The third flow is constructing rattled supercells by rattling the atoms, i.e. displacing all atoms' positions (in the default setup), preparing the VASP calculations and collecting the data for the MLIP fit (`rattled DFT structures`).

After a few data preprocessing steps (`data preprocessing`) to filter out data with too strong force values, the MLIP fit (`machine learning fit`) is run and the resulting potential is used for the benchmark against DFT data (`benchmark`).
Finally, the result metrics are collected in form of output plots and files (`write benchmark metrics`).
