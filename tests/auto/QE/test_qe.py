from jobflow_remote import submit_flow, set_run_config
from autoplex.misc.qe import QEStaticMaker, QeRunSettings, QeKpointsSettings

# --------- QE namelists dictionaries ---------
control_dict = {
    "calculation" : "scf",
    "restart_mode" : "from_scratch",
    "prefix" : "FeCOH",
    "tprnfor" : True,
    "tstress" : True,
    "outdir" : "./OUT/",
    "disk_io" : 'none',
    "pseudo_dir" : '/leonardo/home/userexternal/apacini0/PSEUDO',
    "max_seconds" : 86000
}

system_dict = {
    "ibrav" : 0,
    "nat" : 506,
    "ntyp" : 4,
    "ecutwfc" : 60,
    "ecutrho" : 480,
    "occupations" : "smearing",
    "smearing" : "gaussian",
    "degauss" : 0.00735,
    "nosym" : True,
    "vdw_corr" : "dft-d3",
    "nspin" : 2,
    "starting_magnetization(1)" : 0.4
}

electrons_dict = {
    "diagonalization" : "david",
    "mixing_beta" : 0.15,
    "electron_maxstep" : 150,
    "mixing_mode" : "local-TF",
    "mixing_ndim" : 16,
    "conv_thr" : 1.0e-6,
}    

pseudo_dict = {
    "Fe": "Fe.pbe-sp-van.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
}

# --------- Resources ---------
parallel_gpu_resources = {
    "account": "IscrB_CNT-HARV", 
    "partition": "boost_usr_prod",
    "qos": "boost_qos_dbg",
    "time": "00:30:00",
    "nodes": 1,
    "ntasks_per_node": 4,
    "cpus_per_task": 8,
    "gres": "gpu:4",
    "mem": "480000",
    "job_name": "qe_auto",
    "qerr_path": "JOB.err",
    "qout_path": "JOB.out",
    }



if __name__ == "__main__":
    # QE command
    qe_command = "mpirun -np 4 pw.x -nk 2"

    # QE run seetings (computational parameters namelist)
    qe_run_settings = QeRunSettings(
        control=control_dict,
        system=system_dict,
        electrons=electrons_dict,
    )

    # QE KPOINTS settings
    k_points_settings = QeKpointsSettings(
        kspace_resolution=0.25,  # angstrom^-1
        koffset=[False, False, True],
    )

    # Instance of QEStaticMaker
    qe_maker = QEStaticMaker(
        name="static_qe",
        command=qe_command,
        template_pwi=None,  # Optional if run_settings
        structures="/leonardo_work/EUHPC_A04_113/Alberto/GenMLFF-progect/Test-QE/initial_dataset.extxyz",
        workdir=None,  # default <cwd>/qe_static
        run_settings=qe_run_settings,
        kpoints=k_points_settings,
        pseudo=pseudo_dict,  
    )    

    # Define QE scf workflow
    qe_workflow = qe_maker.make()

    # Update flow config
    set_run_config(qe_workflow, name_filter="static_qe", exec_config="qe_config", worker="schedule_worker", resources=parallel_gpu_resources)

    # Submit flow
    submit_flow(qe_workflow, worker="local_worker", resources={}, project="GenMLFF")