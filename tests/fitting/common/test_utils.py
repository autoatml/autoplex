import os.path
from ase.io import read
from autoplex import MLIP_HYPERS
from autoplex.fitting.common.utils import (
    gap_hyperparameter_constructor,
    check_convergence,
    data_distillation,
    prepare_fit_environment,
    calculate_delta_2b,
    stratified_dataset_split,
    compute_num_of_descriptor
)

def test_stratified_split(test_dir):
    atoms=read(test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz",':')

    train, test=stratified_dataset_split(atoms,0.33, energy_label="REF_energy")

    assert len(train) == 7
    assert len(test) == 3

def test_gap_hyperparameter_constructor():

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )
    # test if string for all possible args true
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap add_species=T "
        "l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=False,
        include_two_body=True,
    )
    # test if string for include_soap==False
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T}",
    ]

    assert ref_list == gap_input_list

    # test if returned string is changed if passed in dict is updated
    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter.update_parameters({"twob": {"cutoff": 8},
                                                                 "threeb": {"cutoff": 8.0,
                                                                            "n_sparse": 100},
                                                                 "soap": {"delta": 1.5,
                                                                          "zeta": 2}})
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    gap_input_list_updated = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )

    ref_list_exp = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=8 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=8.0 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=2 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list_exp == gap_input_list_updated

    # check disable three_body and two_body

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    # three_body_disable

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=True,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    # disable two_body

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter = MLIP_HYPERS.GAP.model_copy(deep=True)
    gap_hyper_parameter_dict = gap_hyper_parameter.model_dump(by_alias=True)

    # check with only soap enabled

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "e0_offset=0.0",
        "gap={soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=6000 delta=1.0 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list


def test_check_convergence():
    check_convergence(0.002)

    assert True


def test_data_distillation(test_dir):
    atoms = data_distillation((test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz"), 35.0, force_label="REF_forces")

    for atom in atoms:
        if (atom.symbols == "Li32Cl32").any() or (atom.symbols == "Li").any() or (atom.symbols == "Cl").any():
            assert True


def test_calculate_delta_2b(test_dir):
    atoms=read(test_dir / "fitting" / "rss_training_dataset"/ "train.extxyz",':')
    
    delta = calculate_delta_2b(atoms_db=atoms, cutoff=4.0, e_name='REF_energy')
    assert (delta-0.7810536) < 1e-6
    
    
def test_calculate_num_of_descriptor(test_dir):
    atoms=read(test_dir / "fitting" / "rss_training_dataset"/ "train.extxyz",':')
    num_2b = sum([compute_num_of_descriptor(atom=at, nb=2, cutoff=4) for at in atoms])
    num_3b = sum([compute_num_of_descriptor(atom=at, nb=3, cutoff=3.25) for at in atoms])
    assert num_2b == 2370
    assert num_3b == 2573


def test_prepare_fit_environment(test_dir, clean_dir):
    prepare = prepare_fit_environment(
        database_dir=(test_dir / "fitting" / "ref_files"),
        mlip_path=(test_dir / "fitting" / "test_mlip_path"),
        glue_xml=False,
    )

    assert os.path.isdir(prepare)
