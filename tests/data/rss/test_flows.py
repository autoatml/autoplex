from autoplex.data.rss.flows import BuildMultiRandomizedStructure
from jobflow import run_locally, Flow


def test_build_multi_randomized_structure(memory_jobstore, clean_dir):
    from autoplex.data.common.utils import flatten
    bcur_params = {'soap_paras': {'l_max': 3,
                                  'n_max': 3,
                                  'atom_sigma': 0.5,
                                  'cutoff': 4.0,
                                  'cutoff_transition_width': 1.0,
                                  'zeta': 4.0,
                                  'average': True,
                                  'species': True,
                                  },
                   }
    generate_structure = BuildMultiRandomizedStructure(tag="Si",
                                                       generated_struct_numbers=[50, 50],
                                                       buildcell_options=[{'VARVOL': 20,
                                                                           'VARVOL_RANGE': '0.75 1.25',
                                                                           'NATOM': '{6,8,10,12,14,16,18,20,22,24}',
                                                                           'NFORM': '1'},
                                                                          {'SYMMOPS': '1-2',
                                                                           'NATOM': '{7,9,11,13,15,17,19,21,23}',
                                                                           'NFORM': '1'}],
                                                       num_processes=8,
                                                       initial_selection_enabled=True,
                                                       selected_struct_numbers=[8, 2],
                                                       bcur_params=bcur_params,
                                                       random_seed=None).make()

    job_rss = Flow(generate_structure, output=generate_structure.output)
    responses = run_locally(job_rss,
                            ensure_success=True,
                            create_folders=True,
                            store=memory_jobstore)

    structures = job_rss.output.resolve(memory_jobstore)

    n_atoms = [struct.num_sites for struct in flatten(structures, recursive=False)]

    assert max(n_atoms) < 25

    even_count = sum(1 for n in n_atoms if n % 2 == 0)
    odd_count = sum(1 for n in n_atoms if n % 2 != 0)

    assert even_count == 8
    assert odd_count == 2
