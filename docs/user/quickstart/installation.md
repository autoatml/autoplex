(installation)=

*Tutorials written by Christina Ertural ([christina.ertural@bam.de](mailto:christina.ertural@bam.de)).*

# Installation guide

## Before you start using `autoplex`

We expect the general user of `autoplex` to be familiar with the [Materials Project](https://github.com/materialsproject) framework software tools and related 
packages for (high-throughput) workflow submission and management. 
This involves the following software packages:
- [pymatgen](https://github.com/materialsproject/pymatgen) for input and output handling of computational materials science software,
- [atomate2](https://github.com/materialsproject/atomate2) for providing a library of pre-defined computational materials science workflows,
- [jobflow](https://github.com/materialsproject/jobflow) for processes, job and workflow handling, 
- [jobflow-remote](https://github.com/Matgenix/jobflow-remote) or [FireWorks](https://github.com/materialsproject/fireworks) for workflow and database (MongoDB) management,
- [MongoDB](https://www.mongodb.com/) as the database (we recommend installing the MongoDB community version). More help regarding the MongoDB installation can be found [here](https://materialsproject.github.io/fireworks/installation.html#install-mongodb).

All of these software tools provide documentation and tutorials. Please take your time and check everything out!

## Workflow management

You can manage your `autoplex` workflow using [`FireWorks`](https://materialsproject.github.io/fireworks/) or [`jobflow-remote`](https://matgenix.github.io/jobflow-remote/). 
Please follow the installation and setup instructions on the respective guide website.
Both packages rely on the [MongoDB](https://www.mongodb.com/) database manager for data storage.

We recommend using `jobflow-remote` as it is more flexible to use, especially on clusters where users cannot store their
own MongoDB. You can find a more comprehensive `jobflow-remote` tutorial [here](../jobflowremote.md).

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