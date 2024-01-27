from jobflow import run_locally, Flow
from jobflow import Response, job
import sys
from autoplex.structure import random_structure
from autoplex.selection import boltz_cur
from autoplex.vasp import VASP_static, VASP_collect_data
from autoplex.mlip_fitting import data_preprocessing, mlip_fit
from autoplex.sampler import do_rss

@job
def initial_RSS():

    ###### 0th-genertation potential
    job1 = random_structure(struct_number=10000, tag='LiPNO',buildcell_options=['VARVOL=15-20',
                                    'SPECIES=Li%NUM=1,P%NUM=1,O%NUM=1,N%NUM=1',
                                    'NFORM=2-10',
                                    'SYMMOPS=1-8',
                                    'SLACK=0.25',
                                    'OVERLAP=0.1',
                                    'COMPACT',
                                    'MINSEP=2.5 Li-O=1.8 P-O=1.5 O-N=1.25 Li-Li=2.8 Li-N=1.8 O-N=1.5 O-O=2.8 Li-P=2.8',]).make()
    job2 = boltz_cur(selection_method='cur',num_of_cur=100).make(dir=job1.output)
    job3 = VASP_static(structures=job2.output, e0_spin=False, isolated_atom=True, dimer=True)
    job4 = VASP_collect_data(vasp_ref_file='vasp_ref.extxyz', gap_rss_group='initial').make(vasp_dirs=job3.output)
    job5 = data_preprocessing(split_ratio=0.1, regularization=True, distillation=True, f_max=40).make(vasp_ref_dir=job4.output['vasp_ref_dir'], pre_database_dir=None)
    job6 = mlip_fit(mlip_type='ACE').make(database_dir=job5.output, gap_para={'two_body':True,'three_body':True}, ace_para={'energy_name':"REF_energy", 'force_name':"REF_forces", 'virial_name':"REF_virial", 'order':4, 'totaldegree':12, 'cutoff':5.0, 'solver':'BLR'},isol_es=job4.output['isol_es'])
    job_list = [job1, job2, job3, job4, job5, job6]

    return Response(replace=Flow(job_list),
                    output={'test_error':job6.output['test_error'],
                            'pre_database_dir':job5.output,
                            'mlip_path':job6.output['mlip_path'],
                            'isol_es':job4.output['isol_es'],
                            'current_iter':0,
                            'kt':0.6},)
                            
@job
def do_RSS_iterations(input={'test_error': None,
                             'pre_database_dir':None,
                             'mlip_path': None,
                             'isol_es': None,
                             'current_iter': None,
                             'kt':0.6},
                    stop_criterion = 0.01,
                    max_iteration_number = 10):

    if input['test_error'] > stop_criterion and input['current_iter'] < max_iteration_number:
        if input['current_iter'] % 2 == 0:
            if input['kt'] > 0.15:
                kt = input['kt']-0.1
            else:
                kt = 0.1
        else:
            kt = input['kt']
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print('The error of' + str(current_iter) + 'st interation:', input['test_error'])
        flag1 = random_structure(struct_number=10000, tag='LiPNO',buildcell_options=['VARVOL=15-20',
                                    'SPECIES=Li%NUM=1,P%NUM=1,O%NUM=1,N%NUM=1',
                                    'NFORM=2-10',
                                    'SYMMOPS=1-8',
                                    'SLACK=0.15',
                                    'OVERLAP=0.1',
                                    'COMPACT',
                                    'MINSEP=2.5 Li-O=1.8 P-O=1.5 O-N=1.25 Li-Li=2.8 Li-N=1.8 O-N=1.5 O-O=2.8 Li-P=2.8']).make()
        flag2 = boltz_cur(selection_method='cur',num_of_cur=1000).make(dir=flag1.output)
        flag3 = do_rss(mlip_type='ACE',iteration_index=str(current_iter) + 'st').make(mlip_path=input['mlip_path'], structure=flag2.output)
        flag4 = boltz_cur(selection_method='boltzhist_CUR',kT=kt,num_of_cur=100).make(traj_info=flag3.output, isol_es=input["isol_es"])
        flag5 = VASP_static(structures=flag4.output, isolated_atom=False, dimer=False)
        flag6 = VASP_collect_data(vasp_ref_file='vasp_ref.extxyz', gap_rss_group='rss').make(vasp_dirs=flag5.output)
        flag7 = data_preprocessing(split_ratio=0.1, regularization=True, distillation=True, f_max=40).make(vasp_ref_dir=flag6.output['vasp_ref_dir'], pre_database_dir=input['pre_database_dir'])
        flag8 = mlip_fit(mlip_type='ACE').make(database_dir=flag7.output, gap_para={'two_body':True,'three_body':True}, ace_para={'energy_name':"REF_energy", 'force_name':"REF_forces", 'virial_name':"REF_virial", 'order':4, 'totaldegree':12, 'cutoff':5.0, 'solver':'BLR'},isol_es=input["isol_es"])

        flag9 = do_RSS_iterations({'test_error':flag8.output['test_error'],
                            'pre_database_dir':flag7.output,
                            'mlip_path':flag8.output['mlip_path'],
                            'isol_es':input["isol_es"],
                            'current_iter': current_iter,
                            'kt': kt})
        job_list = [flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8, flag9]

        return Response(detour=job_list,
                        output=flag9.output)
    else:
        return input
    