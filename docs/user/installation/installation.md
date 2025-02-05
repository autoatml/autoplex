(installation)=

# Installation guide

## Before you start using `autoplex`

We expect the general user of `autoplex` to be familiar with the [Materials Project](https://github.com/materialsproject) framework software tools and related 
packages for (high-throughput) workflow submission and management. 
This involves the following software packages:
- [pymatgen](https://github.com/materialsproject/pymatgen) for input and output handling of computational materials science software,
- [atomate2](https://github.com/materialsproject/atomate2) for providing a library of pre-defined computational materials science workflows,
- [jobflow](https://github.com/materialsproject/jobflow) for processes, job and workflow handling, 
- [jobflow-remote](https://github.com/Matgenix/jobflow-remote) or [FireWorks](https://github.com/materialsproject/fireworks) for workflow and database (MongoDB) management,
- [MongoDB](https://www.mongodb.com/) as the database (we recommend installing the MongoDB community edition). 


We are also referring the user to the [installation guide of atomate2](https://materialsproject.github.io/atomate2/user/install.html) in order to setup the mandatory prerequisites to 
be able to use `autoplex`.

After setting up `atomate2`, make sure to add `VASP_INCAR_UPDATES: {"NPAR": number}` in your ~/atomate2/config/atomate2.yaml file. 
Set a number that is a divisor of the number of tasks you use for the VASP calculations.


## Installation Documentation and Guides of the Dependencies

The first step you need to do is to set up a [MongoDB](https://www.mongodb.com/) database. Help and tips regarding the MongoDB installation 
can be found [here](https://materialsproject.github.io/fireworks/installation.html#install-mongodb). 
We recommend installing the [MongoDB community edition](https://www.mongodb.com/docs/manual/administration/install-community/).
MongoDB also provides lots of [installation guides](https://www.mongodb.com/docs/manual/administration/install-on-linux/#std-label-install-mdb-community-edition-linux) 
and [tutorials](https://www.mongodb.com/docs/manual/administration/self-managed-configuration-and-maintenance/)
to setup and manage your database. For a Kick-start with MongoDB, we also provide a [MongoDB tutorial](../mongodb.md). 
Also consider asking your IT administration for help.

The next step you need do is to install a workflow manager. There are currently two options: [jobflow-remote](https://github.com/Matgenix/jobflow-remote) or [FireWorks](https://github.com/materialsproject/fireworks).
There are also documentation and tutorials available for [FireWorks](https://materialsproject.github.io/fireworks/) and [jobflow-remote](https://matgenix.github.io/jobflow-remote/).
We recommend using `jobflow-remote` and provide a more comprehensive `jobflow-remote` tutorial [here](../jobflowremote.md).

Please take your time and check out all the documentation and tutorials!

When you have completed all these preparation steps, it's time to install `autoplex`!

You can install `autoplex` simply by:

``` 
pip install autoplex[strict]
```
This will install all the Python packages and dependencies needed for MLIP fits. 

Additionally, to fit and validate `ACEpotentials`, one also needs to install Julia, as `autoplex` relies on [ACEpotentials](https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/), which supports fitting of linear ACE. Currently, no Python package exists for the same.
Please run the following commands to enable the `ACEpotentials` fitting options and further functionality.

Install Julia v1.9.2

```bash
curl -fsSL https://install.julialang.org | sh -s -- default-channel 1.9.2
```

Once installed in the terminal, run the following commands to get Julia ACEpotentials dependencies.

```bash
julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add(Pkg.PackageSpec(;name="ACEpotentials", version="0.6.7")); Pkg.add("DataFrames"); Pkg.add("CSV")'
```

### Enabling RSS workflows

Additionally, `buildcell` as a part of `AIRSS` needs to be installed if one wants to use the RSS functionality:

```bash
curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz; tar -xf airss-0.9.3.tgz; rm airss-0.9.3.tgz; cd airss; make ; make install ; make neat; cd ..
```

### LAMMPS installation

You only need to install LAMMPS, if you want to use J-ACE as your MLIP.
Recipe for compiling lammps-ace including the download of the `libpace.tar.gz` file:

```
git clone -b release https://github.com/lammps/lammps
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

After the installation is completed, enter the following commands in the Python environment.
If you get the same output, it means the installation was successful.

```
from lammps import lammps; lmp = lammps()
LAMMPS (27 Jun 2024)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task
Total wall time: 0:02:22
```
It is very important to have it compiled with Python (`-D PKG_PYTHON=on`) and 
LIB PACE flags (`-D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}')`).

As `autoplex` heavily relies on `atomate2`, it is strongly recommended to also make yourself familiar with the [atomate2 documentation](https://materialsproject.github.io/atomate2/).


For a more advanced installation, you can also follow the [developer installation guide](../../dev/dev_install.md).

## Workflow management

You can manage your `autoplex` workflow using [`FireWorks`](https://materialsproject.github.io/fireworks/) or [`jobflow-remote`](https://matgenix.github.io/jobflow-remote/). 
Please follow the installation and setup instructions on the respective guide website.
Both packages rely on the [MongoDB](https://www.mongodb.com/) database manager for data storage.

We recommend using `jobflow-remote` as it is more flexible to use, especially on clusters where users cannot store their
own MongoDB. You can find a more comprehensive `jobflow-remote` tutorial [here](../jobflowremote.md).

> ℹ️ These workflow managers are not included in the standard installation by default, to install autoplex with workflow managers please install `autoplex` using `pip install autoplex[strict,workflow-managers]`

> ℹ️ If using fireworks to manage your jobs, additionally please update `pymongo` package to v4.11, using
`pip install --upgrade pymongo==4.11` 

Submission using `FireWorks`:
```python
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

...

autoplex_flow = ...

wf = flow_to_workflow(autoplex_flow)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
```

Submission using `jobflow-remote`:
```python
from jobflow_remote import submit_flow, set_run_config

...

autoplex_flow = ...

# setting different job setups in the submission script directly:
resources = {"nodes": N, "partition": "name", "time": "01:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}
            # put your slurm submission keywords as needed
            # you can add "qverbatim": "#SBATCH --get-user-env" in case your conda env is not activated automatically

resources_phon = {"nodes": N, "partition": "name", "time": "05:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

resources_ratt = {"nodes": N, "partition": "micro", "time": "03:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

resources_mlip = {"nodes": N, "partition": "name", "time": "02:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

autoplex_flow = set_run_config(autoplex_flow, name_filter="dft phonon static", resources=resources_phon)

autoplex_flow = set_run_config(autoplex_flow, name_filter="dft rattle static", resources=resources_ratt)

autoplex_flow = set_run_config(autoplex_flow, name_filter="machine_learning_fit", resources=resources_mlip)

# submit the workflow to jobflow-remote
print(submit_flow(autoplex_flow, worker="autoplex_worker", resources=resources, project="autoplex"))
```
