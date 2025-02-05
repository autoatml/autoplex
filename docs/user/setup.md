Quick-start
================

This guide assumes that you have all the [Materials Project](https://github.com/materialsproject) framework software tools as well as a working [MongoDB](https://www.mongodb.com/) 
database setup and have experience using [atomate2](https://github.com/materialsproject/atomate2).

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

## Enabling RSS workflows

Additionally, `buildcell` as a part of `AIRSS` needs to be installed if one wants to use the RSS functionality:

> ℹ️ To be able to build the AIRSS utilities one needs gcc and gfortran version 5 and above. Other compiler families (such as ifort) are not supported.


```bash
curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz; tar -xf airss-0.9.3.tgz; rm airss-0.9.3.tgz; cd airss; make ; make install ; make neat; cd ..
```

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

After the installation is completed, enter the following commands in the Python environment.
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


A more comprehensive `autoplex` installation guide can be found [here](installation/installation.md) and for a even more advanced 
installation, you can also follow the [developer installation guide](../dev/dev_install.md).


## Workflow management

You can manage your `autoplex` workflow using [`FireWorks`](https://materialsproject.github.io/fireworks/) or [`jobflow-remote`](https://matgenix.github.io/jobflow-remote/). 
Please follow the installation and setup instructions on the respective guide website.
Both packages rely on the [MongoDB](https://www.mongodb.com/) database manager for data storage.

We recommend using `jobflow-remote` as it is more flexible to use, especially on clusters where users cannot store their
own MongoDB. You can find a more comprehensive `jobflow-remote` tutorial [here](jobflowremote.md).

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
