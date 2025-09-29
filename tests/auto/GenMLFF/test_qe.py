from jobflow_remote import submit_flow, set_run_config
from autoplex.auto.GenMLFF.QuantumEspressoSCF import qe_params_from_config, QEstaticLabelling

if __name__ == "__main__":

    # Resources for QE
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


    # QE job parameters (test)
    qe_test_params = {
        "qe_run_cmd": "mpirun -np 2 pw.x -nk 2 ", #Command to run QE scf calculation
        "num_qe_workers": 2, #Number of workers to use for the calculations. If None setp up 1 worker per scf
        "kspace_resolution": 0.25, #k-point spacing in 1/Angstrom
        "koffset": [False, False, True], #k-point offset
        "fname_pwi_template": "/leonardo_work/EUHPC_A04_113/Alberto/GenMLFF-progect/autoplex/tests/auto/GenMLFF/reference.pwi", #Path to file containing the template QE input
        "fname_structures": "/leonardo_work/EUHPC_A04_113/Alberto/GenMLFF-progect/Test-QE/initial_dataset.extxyz", #Path to file containing the structures to be computed
    }

    # Define QE scf workflow
    qe_params = qe_params_from_config(qe_test_params)
    qe_workflow = QEstaticLabelling(**qe_params).make()

    # Update flow config
    set_run_config(qe_workflow, name_filter="run_qe_worker", exec_config="qe_config", worker="schedule_worker", resources=parallel_gpu_resources)

    # Submit flow
    submit_flow(qe_workflow, worker="local_worker", resources={}, project="GenMLFF")