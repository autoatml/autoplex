---
orphan: true
---

# Jobflow-remote setup

This will result in a setup for automation where 
1. We will add/submit job to DB on your local machine.
2. Jobs will be executed on your remote custer.

# Installation

## on your local machine
1. Create a new env > `conda create -n autoplex python=3.10`. (You can choose any other env name.)
2. Activate your env > `conda activate autoplex` 
3. Install jobflow-remote by `pip install git+https://github.com/Matgenix/jobflow-remote@v0.1.4`.
4. You can check for the latest release of [jobflow-remote](https://github.com/Matgenix/jobflow-remote/releases).
5. Install autoplex > In your local autoplex directory: `pip install -e .[strict]`. 
6. Run `jf project generate --full YOUR_PROJECT_NAME`. 
Choose a sensible project name.
This will generate an empty project config file in your home directory. 
You can find this file inside `~/.jfremote` 
(This is optional, a config file is provided here: [test_project.yaml](test_project.yaml), 
you can simply copy this file to the  `~/.jfremote` directory. 
You will need to create `~/.jfremote` directory in your home if it's not created automatically.
Creat the log, tmp and daemon subfolders if they are not created automatically.)


## on your remote cluster
7. Repeat step 1,2,3,4 and 5 on your remote cluster.
8. Now setup atomate2 config as usual.
Just `atomate2/config/atomate2.yaml`. (We do not need to set up jobflow.yaml in atomate2/config)

Below is an example `atomate2.yaml` config file
```yaml
VASP_CMD:  your hpc vasp_std cmd
VASP_GAMMA_CMD:  your hpc vasp_gam cmd
LOBSTER_CMD: your hpc lobster cmd
```

9. Add environment variable to your ~/.bashrc `export ATOMATE2_CONFIG_FILE="/path/to/atomate2/config/atomate2.yaml"`

## Changes to be done in the config file - on your local machine
1. Set paths to base, tmp, log, daemon dir. Best would be, simply creating empty dirs in your `~/.jfremote` directory. 
Use the paths as provided in sample config file for reference.
2. Under the `workers` section of the yaml, change worker name from `example_worker` to your liking, set `work_dir` 
(directory where calcs will be run), set `pre_run` command (use to activate the environment before job execution), 
set `user` (this your username on your remote cluster)  
3. In `queue` section, just change details as per your MongoDB (admin username password, host, port, name)


# Check if your setup works correctly

> Note: If you have any password protected key in your `~/.ssh` directory worker might fail to start. To overcome this, temporarily move your passphrase procted keys from `~/.ssh` directory to some other directory before starting the runner.

1. `jf project check -w example_worker` 
(If everything is setup correctly, you will get asked for password and OTP (one-time password) for MFA (multifactor
authentication) login and will exit with a green tick in few secs.)
2. `jf project check --errors` this will check all even your MongoDB connection is proper or not. 
If anything fails, please check the config file.


# Getting started

1. Run `jf admin reset` (Do not worry, this will reset your DB, necessary to do only once. 
You can skip this if you want to keep the data in your DB.)
2. `jf runner start -s -i`

You can type `jf runner start -h` for help and more information:
```bash
(conda_env) user@local_host:~$ jf runner start -h
The selected project is test_pproject from config file /home/user/.jfremote/test_project.yaml
                                                                                                                                                                               
 Usage: jf runner start [OPTIONS]                                                                                                                                              
                                                                                                                                                                               
 Start the Runner as a daemon                                                                                                                                                  
                                                                                                                                                                               
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --transfer             -t        INTEGER                  The number of processes dedicated to completing jobs [default: 1]                                                     │
│ --complete             -com      INTEGER                  The number of processes dedicated to completing jobs [default: 1]                                                     │
│ --single               -s  <---                           Use a single process for the runner                                                                                   │
│ --log-level            -log      [error|warn|info|debug]  Set the log level of the runner [default: info]                                                                       │
│ --connect-interactive  -i  <---                           Wait for the daemon to start and manually log in the connection for interactive remote host. Requires --single.       │
│ --help                 -h                                 Show this message and exit.                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
We need to use the options `-s -i` for the interactive mode to use the MFA login with OTP. 
The two options are highlighted by `<---` in the above example.

You will be prompted with a question "Do you want to open the connection for the host of the XXX worker?" 
Answer "y". And then you should be prompted for password and OTP.
After that you can quit the interactive mode with ctrl+c. The runner should now be working fine until the connection drops.
 
During the starting of the runner, you will probably see a few error/warnings. 
First, a warning that the password may be echoed. Ignore it, it should not.  

3. `jf runner status` (this should return status of runner as `running`, if everything is set up correctly)


# Example job scripts to test (Add/Submit jobs to DB from your local machine)

## Simple Python job

```python
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow

job1 = add(1, 2)
job2 = add(job1.output, 2)

flow = Flow([job1, job2])

resources = {"nodes": N, "partition": "name", "time": "01:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

print(submit_flow(flow, worker="example_worker", resources=resources, project="test_project")) 
# Do not forget to change worker and project name to what you se tup in the jobflow remote config file.
```
> NOTE: We are using [Simple Linux Utility for Resource Management (SLURM)](https://matgenix.github.io/qtoolkit/api/qtoolkit.io.slurm.html) specific keywords in our examples.
> For [Portable Batch System (PBS)](https://matgenix.github.io/qtoolkit/api/qtoolkit.io.pbs.html) specific commands, see [here](https://matgenix.github.io/qtoolkit/api/qtoolkit.io.pbs.html).

## VASP relax job using atomate2 workflow

```python
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow
from mp_api.client import MPRester
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.powerups import update_user_incar_settings 


mpid = "mp-22862"
mr = MPRester(api_key='YOUR_MP_API_KEY')
struct = mr.get_structure_by_material_id(mpid)

# we use the same structure (mp-22862) here and instantiate the workflow
relax_job = DoubleRelaxMaker().make(structure=struct)

relax_job = update_user_incar_settings(relax_job, {"NPAR": 4})

# You can also pass exe_config for the worker using exe_config in submit flow. Below is an example 
# exec_config={"pre_run": "source activate autoplex \n module load slurm_setup \n module load vasp/6.1.2"}

resources = {"nodes": N, "partition": "name", "time": "01:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

print(submit_flow(relax_job, worker="example_worker", resources=resources, project="test_project"))
```
It is crucial to set `"qverbatim": "#SBATCH --get-user-env"` to make sure the same environment is used on your remote cluster.

# Setting different workers for different job types

This is very much similar to how we do in atomate2, jobflow-remote provides a specific utility for this.
```python
from jobflow_remote import set_run_config
```
An example use case can be found [here](https://matgenix.github.io/jobflow-remote/user/tuning.html#jobconfig)

# Querying completed jobs from DB using jobflow-remote Python API

```python
from jobflow_remote import get_jobstore

js = get_jobstore(project_name='YOUR_PROJECT_NAME') 
js.connect()
result = js.query(criteria={"name": "generate_frequencies_eigenvectors"},load=True) 
# example query for completed phonon workflow runs
# the query methods are the same as in atomate2 basically, 
for i in result:
    print(i['output']["phonon_bandstructure"]) 
    # get phonon banstructure pymatgen object
```

# Updating failed jobs time limit or execution config
```python
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='YOUR_PROJECT_NAME') # initialize a job controller

job_docs = jc.get_jobs_doc(db_ids='214') # query job docs based on different criteria 
# (Check documentation to see all available options https://github.com/Matgenix/jobflow-remote/blob/967e7c512f230105b1a82c2227fb101d8d4acb3d/src/jobflow_remote/jobs/jobcontroller.py#L467)

# get your existing resources
resources = job_docs[0].resources

# update time limit in the retrieved dict (you can update any other keys like partition/ nodes etc as well)
resources["time"] = '8:00:00'

jc.rerun_job(db_id=job_docs[0].db_id, force=True) # important for jobs that are in failed state to reset them first
jc.set_job_run_properties(db_ids=[job_docs[0].db_id], resources=resources) # this will update the DB entry
```

> IMPORTANT: When you restart VASP calculations, make sure to move the old VASP files somewhere else, 
> because jobflow-remote will restart your calculation in the same directory and that leads to some clash of old and new files.

# Update pre-exsiting job input parameters in the DB

```python
# Note that this way is bit involved and you need to find exact structure of your nested DB entry based on type of maker used

# Following is an example for failed vasp job where NPAR and ALGO tags in DB entry are updated
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='YOUR_PROJECT_NAME')

job_collection = jc.db.jobs # get jobs collection from mongoDB

for i in job_collection.find({'db_id': '214'}):
    job_dict = i['_id'] # get object id in mongodb (this is used to as filter)
    incar_settings = i['job']['function']['@bound']['input_set_generator']['user_incar_settings'] # get existing user incar settings

incar_settings.update({'NPAR': 2, 'ALGO': 'FAST'}) # now update incar settings here as per requirement
job_collection.update_one({'_id': job_dict}, {'$set': {'job.function.@bound.input_set_generator.user_incar_settings' : incar_settings}})

print(jc.get_jobs_doc(db_ids='214')[0].job.maker.input_set_generator.user_incar_settings) # check if entries are updated
```
> IMPORTANT: When you restart VASP calculations, make sure to move the old VASP files somewhere else, 
> because jobflow-remote will restart your calculation in the same directory and that leads to some clash of old and new files.

# Some useful commands

1. `jf job list` (list jobs in the DB)
2. `jf flow list` (list of flows in the DB)
3. `jf job info jobid` (provides some info of job like workdir, error info if it failed)
4. `jf flow delete -did db_id` (deletes flow from DB)
5. `jf job list -s STATE`, e.g. `jf job list -s FAILED`
6. `jf job rerun -s STATE`, e.g. `jf job rerun -s FAILED`
7. `jf job retry -s STATE`, e.g. `jf job retry -s REMOTE_ERROR`
8. `jf flow info flow_id` (shows the jobs of a certain flow)
9. `jf flow -h` or `jf job -h` for checking other options

In case your remote cluster connection is broken or lost, it is crucial to restart the 
jobflow-remote runner [(first stop/kill, then start)](#getting-started). Job states that are `REMOTE_ERROR`, `FAILED` or
inconsistent because of this, can be fixed be force-rerunning the respective jobs by `jf job rerun -f -s STATE`.

# Some useful links

1. Check slurm.py for finding different available options you can set for resources dict [here](https://github.com/Matgenix/qtoolkit/tree/develop/src/qtoolkit/io)
2. More details on project config and settings can be found [here](https://matgenix.github.io/jobflow-remote/user/projectconf.html)
3. Details on different setup options [here](https://matgenix.github.io/jobflow-remote/user/install.html)

# Download (all) the data of a flow from your remote cluster

Once, your workflows have finished, you can download the data you need for further analysis.
You can first check the jobs of your workflow via jobflow-remote as `jf flow info db_id/flow_id` 
to check the names of the jobs whose data you want to retrieve, like e.g.:
```bash
jf flow info 12345
The selected project is autoplex from config file /home/user/.jfremote/auto.yaml
                             Flow: LiCl - f45374e9-f095-4c8a-a0e5-be3e50e15e21 - COMPLETED                             
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ DB id ┃ Name                                         ┃ State     ┃ Job id  (Index)                           ┃ Worker      ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 43076 │ run_phonon_displacements_mp-22905            │ COMPLETED │ 0074c8d0-ce6a-49fd-a2a8-ffd00c14982e  (1) │ auto_worker │
│ 43126 │ store_inputs_mp-22905                        │ COMPLETED │ 0074c8d0-ce6a-49fd-a2a8-ffd00c14982e  (2) │ auto_worker │
│ 43106 │ gap phonon static 1/20_mp-22905              │ COMPLETED │ 025a49ba-a45d-493f-8761-021f506da5c4  (1) │ auto_worker │
│ 43020 │ tight relax 1_mp-22905                       │ COMPLETED │ 07f2f80d-df47-4cf1-b4e8-0d5e00f1a74e  (1) │ auto_worker │
│ 43070 │ run_phonon_displacements_mp-22905            │ COMPLETED │ 083c4c17-d1cd-4ef7-b1bc-8c6be8305f0f  (1) │ auto_worker │
│ 43069 │ generate_phonon_displacements_mp-22905       │ COMPLETED │ 0b6bffe3-10a8-47b4-b15d-c948337c0dae  (1) │ auto_worker │
│ 43017 │ generate_randomized_structures_mp-22905      │ COMPLETED │ 198b30d8-470c-4357-893e-a6ce98dcea3f  (1) │ auto_worker │
│ 43047 │ dft phonon static 9/20_mp-22905              │ COMPLETED │ 20bdb7b3-875f-45cb-91a3-dbe11dfa5100  (1) │ auto_worker │
│ 43036 │ dft rattle static 10/11_mp-22905             │ COMPLETED │ 24d92e7b-4aff-4f7a-9883-4f2b31d3f09c  (1) │ auto_worker │
│ 43077 │ generate_frequencies_eigenvectors_mp-22905   │ COMPLETED │ 2829ea98-220b-4bdb-9c2f-11c19d95557d  (1) │ auto_worker │
│ 43008 │ single-atom displaced supercells_mp-22905    │ COMPLETED │ 337cf0ee-c003-4b0f-a988-01b903c416ca  (1) │ auto_worker │
│ 43062 │ Benchmark_mp-22905                           │ COMPLETED │ 3d456713-a88e-4b19-a9c4-970413631078  (1) │ auto_worker │
│ 43010 │ machine_learning_fit                         │ COMPLETED │ 72665b79-bcdf-4455-a330-b2b5bca76b30  (1) │ auto_worker │
│ 43011 │ complete_benchmark_mp-22905                  │ COMPLETED │ d4c6c8fb-b8b0-41ee-a794-d13f1cc874ef  (1) │ auto_worker │
│ 43005 │ reduce_supercell_size_job                    │ COMPLETED │ f87e9be8-94f5-46a4-a7a8-d6ba2f4b0577  (1) │ auto_worker │
│ ..... │ ...                                          │ COMPLETED │ ...                                   (1) │ auto_worker │
│ ..... │ ...                                          │ COMPLETED │ ...                                   (1) │ auto_worker │
│ 43084 │ store_inputs_mp-22905                        │ COMPLETED │ faf20068-67bc-426b-aa11-e9f2c9e11ea7  (2) │ auto_worker │
└───────┴──────────────────────────────────────────────┴───────────┴───────────────────────────────────────────┴─────────────┘
```
Depending on the data you need, you can then use the job names in the following scripts for filtering.

You can download your data in one batch from your remote cluster using this script:
```python
import os
import subprocess
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='project-name') # initialize a job controller

flow_query = jc.get_flows_info(flow_ids="the-flow-id")  # make sure to use the long flow_id and not the db_id
job_docs = jc.get_jobs_doc(db_ids=flow_query[0].db_ids)
os.chdir("/your/desired/destination/directory")
user = "your remote cluster user name"
remote_tmp_dir = f"/home/path/to/{user}/tmp_folder"

try:
    subprocess.run(["ssh", f"{user}@remote_cluster", f"mkdir -p {remote_tmp_dir}"], check=True)
    print(f"Created temporary folder on remote cluster: {remote_tmp_dir}")
except subprocess.CalledProcessError as e:
    print(f"Failed to create temporary folder on remote cluster: {e}")
    exit(1)

for i in job_docs:
    base_dir_name = i.job.name.replace(' ', '_').replace('/', '_')
    dir_name = base_dir_name
    counter = 1

    while os.path.exists(dir_name):
        dir_name = f"{base_dir_name}_{counter}"
        counter += 1

    # filter out the data you don't need
    if 'displacements' not in base_dir_name and 'store_inputs' not in base_dir_name and 'reduce_supercell_size' not in base_dir_name and 'relax' not in base_dir_name:
        remote_target_dir = f"{remote_tmp_dir}/{dir_name}"

        remote_dir = f"{i.run_dir}/*"
        scp_to_tmp_command = ["ssh", f"{user}@remote_cluster", f"mkdir -p {remote_target_dir} && cp -r {remote_dir} {remote_target_dir}"]
        try:
            subprocess.run(scp_to_tmp_command, check=True)
            print(f"Copied files from {remote_dir} to {remote_target_dir} on the remote cluster")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy files to remote temporary folder: {e}")

try:
    scp_final_command = ["scp", "-r", f"{user}@remote_cluster:{remote_tmp_dir}", "."]
    subprocess.run(scp_final_command, check=True)
    print(f"Copied {remote_tmp_dir} to the local machine")
except subprocess.CalledProcessError as e:
    print(f"Failed to copy the temporary folder to the local machine: {e}")

try:
    subprocess.run(["ssh", f"{user}@remote_cluster", f"rm -rf {remote_tmp_dir}"], check=True)
    print(f"Cleaned up temporary folder on remote cluster: {remote_tmp_dir}")
except subprocess.CalledProcessError as e:
    print(f"Failed to clean up temporary folder on remote cluster: {e}")
```

Alternatively, you can use this script to download the data of a whole flow:

```python
import os
import subprocess
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='project-name') # initialize a job controller

flow_query = jc.get_flows_info(flow_ids="the-flow-id")  # make sure to use the long flow_id and not the db_id
job_docs = jc.get_jobs_doc(db_ids=flow_query[0].db_ids)
os.chdir("/your/desired/destination/directory")
user = "your remote cluster user name"

for i in job_docs:
    dir_name = i.job.name.replace(' ', '_').replace('/', '_')
    # filter out the data you don't need
    if 'displacements' not in dir_name and 'store_inputs' not in dir_name and 'reduce_supercell_size' not in dir_name and 'relax' not in dir_name:
        base_dir_name = dir_name
        counter = 1

        while os.path.exists(dir_name):
            dir_name = f"{base_dir_name}_{counter}"
            counter += 1

        os.mkdir(dir_name)
        remote_dir = f"{user}@remote_cluster:{i.run_dir}/*"
        scp_command = ["scp", "-r", remote_dir, dir_name]
        try:
            subprocess.run(scp_command, check=True)
            print(f"Copied files from {remote_dir} to {dir_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy files from {remote_dir}: {e}")
```
