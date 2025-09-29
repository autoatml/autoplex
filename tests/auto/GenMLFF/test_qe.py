from jobflow import Flow, job
from jobflow_remote import submit_flow, set_run_config
from autoplex.auto.GenMLFF.QuantumEspressoSCF import qe_params_from_config, QEstaticLabelling

#Define resources 
parallel_gpu_resources = {
    "account": "IscrB_CNT-HARV", 
    "partition": "boost_usr_prod",
    "qos": "boost_qos_dbg",
    "time": "00:30:00",
    "nodes": 1,
    "ntasks_per_node": 2,
    "cpus_per_task": 8,
    "gres": "gpu:2",
    "mem": "240000",
    "job_name": "qe_auto",
    "qerr_path": "JOB.err",
    "qout_path": "JOB.out",
    }


#Define QE test parameters
qe_test_params = {
    "qe_run_cmd": "mpirun -np 1 pw.x",
    "num_qe_workers": 2, #Number of workers to use for the calculations. If None setp up 1 worker per scf
    "kspace_resolution": 0.25, #k-point spacing in 1/Angstrom
    "koffset": [False, False, True], #k-point offset
    "fname_pwi_template": "/leonardo_work/EUHPC_A04_113/Alberto/GenMLFF-progect/autoplex/tests/auto/GenMLFF/reference.pwi", #Path to file containing the template QE input
}

# Update QE default parameters
qe_params = qe_params_from_config(qe_test_params)

# Define QEscf job
fname_structures_to_be_computed = ""
qe_scf_maker = QEstaticLabelling(**qe_params, fname_structures=fname_structures_to_be_computed)
qe_scf_job = qe_scf_maker.make()

# Define flow
flow = Flow([qe_scf_job])
set_run_config(flow, name_filter="run_qe_worker", exec_config="qe_config", worker="QuantumEspresso", resources=parallel_gpu_resources)

# Submit flow
submit_flow(flow, worker="local_worker", resources={}, project="GenMLFF")