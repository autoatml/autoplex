[![Testing Linux](https://github.com/JaGeo/autoplex/actions/workflows/python-package.yml/badge.svg)](https://github.com/JaGeo/autoplex/actions/workflows/python-package.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/autoatml/autoplex/main.svg)](https://results.pre-commit.ci/latest/github/autoatml/autoplex/main) [![DOI](https://zenodo.org/badge/671124251.svg)](https://doi.org/10.5281/zenodo.14105049) ![supported python versions](https://img.shields.io/pypi/pyversions/autoplex) [![PyPI version](https://badge.fury.io/py/autoplex.svg)](https://badge.fury.io/py/autoplex)

<img src="docs/_static/autoplex_logo.png" width="66%">

<div style="border: 1px solid #2e3191; padding: 5px; position: relative;">
    <div style="background-color: #2e3191; color: #ffffff; padding: 0px; position: absolute; top: 0; left: 0; right: 0; text-align: center;">
        <strong>Disclaimer</strong>
    </div>
<br>

`autoplex` is still under very active development and larger modifications to the source code should be expected.
</div>

<br>

`autoplex` is a software for generating and benchmarking machine learning (ML)-based interatomic potentials. The aim of `autoplex` is to provide a fully automated solution for creating high-quality ML potentials. The software is interfaced to multiple different ML potential fitting frameworks and to the atomate2 and ase environments for efficient high-throughput computations. The vision of this project is to allow a wide community of researchers to create accurate and reliable ML potentials for materials simulations.

`autoplex` is developed jointly by two research groups at BAM Berlin and the University of Oxford.

`autoplex` is an evolving project and **contributions are very welcome**! To ensure that the code remains of high quality, please raise a pull request for any contributions, which will be reviewed before integration into the main branch of the code. Initially, [@JaGeo](https://github.com/JaGeo) will handle the reviews.

# Workflow overview

We currently have two different types of automation workflows available:

* Workflow to use random-structure searches for the systematic construction of interatomic potentials: [arXiv: 10.48550/arXiv.2412.16736](https://arxiv.org/abs/2412.16736).
  The implementation automates ideas from the following articles: [*Phys. Rev. Lett.* **120**, 156001 (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.156001) and [*npj Comput. Mater.* **5**, 99 (2019)](https://www.nature.com/articles/s41524-019-0236-6).
* Workflow to train accurate interatomic potentials for harmonic phonon properties. The implementation automates the ideas from the following article: [*J. Chem. Phys.* **153**, 044104 (2020)](https://pubs.aip.org/aip/jcp/article/153/4/044104/1056348/Combining-phonon-accuracy-with-high).


## Documentation

You can find the `autoplex` documentation [here](https://autoatml.github.io/autoplex/index.html)!
The documentation also contains tutorials that teach you how to use `autoplex` for different use cases for the RSS and phonon workflows and individual modules therein.

## What to cite?

Please cite our preprint on the random-structure searches (RSS) performed with `autoplex`:

[Y. Liu, J. D. Morrow, C. Ertural, N. L. Fragapane, J. L. A. Gardner, A. A. Naik, Y. Zhou, J. George, V. L. Deringer, 2024, DOI 10.48550/arXiv.2412.16736](https://arxiv.org/abs/2412.16736).


Please additionally cite all relevant software we rely on within `autoplex` and specific workflows. Please take a look below and check out the corresponding links.

# Before you start using `autoplex`

We expect the general user of `autoplex` to be familiar with the [Materials Project](https://github.com/materialsproject) framework software tools and related
packages for (high-throughput) workflow submission and management.
This involves the following software packages:
- [pymatgen](https://github.com/materialsproject/pymatgen) for input and output handling of computational materials science software,
- [atomate2](https://github.com/materialsproject/atomate2) for providing a library of pre-defined computational materials science workflows,
- [jobflow](https://github.com/materialsproject/jobflow) for processes, job and workflow handling,
- [jobflow-remote](https://github.com/Matgenix/jobflow-remote) or [FireWorks](https://github.com/materialsproject/fireworks) for workflow and database (MongoDB) management,
- [MongoDB](https://www.mongodb.com/) as the database (we recommend installing the MongoDB community edition). More help regarding the MongoDB installation can be found [here](https://materialsproject.github.io/fireworks/installation.html#install-mongodb).

All of these software tools provide documentation and tutorials. Please take your time and check everything out! Please also cite the software packages if you are using them!

## Setup

To set up the mandatory prerequisites for using `autoplex,` please follow the [installation guide of atomate2](https://materialsproject.github.io/atomate2/user/install.html).

After setting up `atomate2`, make sure to add `VASP_INCAR_UPDATES: {"NPAR": number}` in your `~/atomate2/config/atomate2.yaml` file.
Set a number that is a divisor of the number of tasks you use for the VASP calculations.

# Installation

## Python version

Before the installation, please make sure that you are using one of the supported Python versions (see [pyproject.toml](https://github.com/autoatml/autoplex/blob/main/pyproject.toml))

## Standard installation

Please install `autoplex` using
```
pip install autoplex[strict]
```
This will install all the Python packages and dependencies needed for MLIP fits.

> ℹ️ To fit and validate `ACEpotentials`, one also needs to install Julia, as `autoplex` relies on [ACEpotentials](https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/), which supports fitting of linear ACE. Currently, no Python package exists for the same.
Please run the following commands to enable the `ACEpotentials` fitting options and further functionality.

Install Julia v1.9.2

```bash
curl -fsSL https://install.julialang.org | sh -s -- default-channel 1.9.2
```

Once installed in the terminal, run the following commands to get Julia ACEpotentials dependencies.

```bash
julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add(Pkg.PackageSpec(;name="ACEpotentials", version="0.6.7")); Pkg.add("DataFrames"); Pkg.add("CSV")'
```

> ℹ️ To fit and validate NEP potentials, one requires an Nvidia GPU card with compute capability no less than 3.5 and CUDA toolkit 9.0 or newer. This potential can only be trained on GPU only and currently interface to NEP potential training is provided via [calorine](https://calorine.materialsmodeling.org/) package that uses `nep` executable from the [GPUMD](https://gpumd.org/index.html) package. To get this executable please follow the compilation instructions [here](https://gpumd.org/installation.html) and add this executable to the system path.

## Enabling RSS workflows

Additionally, `buildcell` as a part of `AIRSS` needs to be installed if one wants to use the RSS functionality:

> ℹ️ To be able to build the AIRSS utilities one needs `gcc` and `gfortran` version 5 and above. Other compiler families (such as ifort) are not supported.
> These compilers are usually available on HPCs and one can simply load them if needed. On Ubuntu/Debian systems, one can install the necessary compilers with the following command:
````bash
apt install -y build-essential gfortran
````

```bash
curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz; tar -xf airss-0.9.3.tgz; rm airss-0.9.3.tgz; cd airss; make ; make install ; make neat; cd ..
```

Please find out about licenses and citation requirements here: [https://airss-docs.github.io/](https://airss-docs.github.io/)

## LAMMPS installation

You only need to install LAMMPS, if you want to use J-ACE as your MLIP.
Recipe for compiling lammps-ace including the download of the `libpace.tar.gz` file:

```
git clone -b stable_29Aug2024_update1 https://github.com/lammps/lammps.git
cd lammps
mkdir build
cd build
wget -O libpace.tar.gz https://github.com/wcwitt/lammps-user-pace/archive/main.tar.gz

cmake  -C ../cmake/presets/clang.cmake -D BUILD_SHARED_LIBS=on -D BUILD_MPI=yes \
-DMLIAP_ENABLE_PYTHON=yes -D PKG_PYTHON=on -D PKG_KOKKOS=yes -D Kokkos_ARCH_ZEN3=yes \
-D PKG_PHONON=yes -D PKG_MOLECULE=yes -D PKG_MANYBODY=yes \
-D Kokkos_ENABLE_OPENMP=yes -D BUILD_OMP=yes -D LAMMPS_EXCEPTIONS=yes \
-D PKG_ML-PACE=yes -D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}') \
-D CMAKE_INSTALL_PREFIX=$LAMMPS_INSTALL -D CMAKE_EXE_LINKER_FLAGS:STRING="-lgfortran" \
../cmake

make -j 16
make install-python
```

$LAMMPS_INSTALL is the conda environment for installing the lammps-python interface.
Use `BUILD_MPI=yes` to enable MPI for parallelization.

After the installation, enter the following commands in the Python environment.
If you get the same output, it means the installation was successful.

```python
from lammps import lammps; lmp = lammps()
```
```bash
LAMMPS (29 Aug 2024 - Update 1)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task
>>>
```
It is very important to have it compiled with Python (`-D PKG_PYTHON=on`) and
LIB PACE flags (`-D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}')`).

Please find out about licenses and citation requirements here: [https://www.lammps.org/](https://www.lammps.org/)

# Contributing guidelines / Developer's installation

A short guide to contributing to autoplex can be found [here](https://autoatml.github.io/autoplex/dev/contributing.html). Additional information for developers can be found [here](https://autoatml.github.io/autoplex/dev/dev_install.html).
