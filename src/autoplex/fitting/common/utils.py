"""Utility functions for fitting jobs."""

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import ase
import lightning as pl
import matgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.data import chemical_symbols
from ase.io import read, write
from ase.io.extxyz import XYZError
from ase.neighborlist import NeighborList, natural_cutoffs
from atomate2.utils.path import strip_hostname
from calorine.nep import read_loss, write_nepfile, write_structures
from dgl.data.utils import split_dataset
from matgl.apps.pes import Potential
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from monty.dev import requires
from monty.serialization import dumpfn
from nequip.ase import NequIPCalculator
from numpy import ndarray
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pytorch_lightning.loggers import CSVLogger
from scipy.spatial import ConvexHull
from scipy.special import comb

from autoplex import (
    GAP_HYPERS,
    JACE_HYPERS,
    M3GNET_HYPERS,
    MACE_HYPERS,
    NEP_HYPERS,
    NEQUIP_HYPERS,
)
from autoplex.data.common.utils import (
    data_distillation,
    plot_energy_forces,
    rms_dict,
    stratified_dataset_split,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def gap_fitting(
    db_dir: Path,
    species_list: list | None = None,
    hyperparameters: GAP_HYPERS = GAP_HYPERS,
    num_processes_fit: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
    glue_file_path: str = "glue.xml",
    fit_kwargs: dict | None = None,  # pylint: disable=E3701
) -> dict:
    """
    Perform the GAP (Gaussian approximation potential) model fitting.

    Parameters
    ----------
    db_dir: str or path
        Path to database directory.
    species_list: list
        List of element names (strings)
    hyperparameters: MLIP_HYPERS.GAP
        Fit hyperparameters.
    num_processes_fit: int
        Number of processes used for gap_fit
    auto_delta: bool
        Automatically determine delta for 2b, 3b and soap terms.
    glue_xml: bool
        use the glue.xml core potential instead of fitting 2b terms.
    ref_energy_name: str
        Reference energy name.
    ref_force_name : str
        Reference force name.
    ref_virial_name: str
        Reference virial name.
    train_name: str
        Name of the training set file.
    test_name: str
        Name of the test set file.
    glue_file_path: str
        Name of the glue.xml file path.
    fit_kwargs: dict
        Additional keyword arguments for GAP fitting with keys same as
        those in gap-defaults.json.

    Returns
    -------
    dict
        A dictionary with train_error, test_error, path_to_mlip

    """
    hyperparameters = hyperparameters.model_copy(deep=True)
    # keep additional pre- and suffixes
    gap_file_xml = train_name.replace("train", "gap_file").replace(".extxyz", ".xml")
    quip_train_file = train_name.replace("train", "quip_train")
    quip_test_file = test_name.replace("test", "quip_test")
    mlip_path: Path = prepare_fit_environment(
        db_dir, Path.cwd(), glue_xml, train_name, test_name, glue_file_path
    )

    db_atoms = ase.io.read(os.path.join(db_dir, train_name), index=":")
    train_data_path = os.path.join(db_dir, train_name)

    test_data_path = os.path.join(db_dir, test_name)

    hyperparameters.update_parameters(
        {
            "general": {
                "gp_file": gap_file_xml,
                "energy_parameter_name": ref_energy_name,
                "force_parameter_name": ref_force_name,
                "virial_parameter_name": ref_virial_name,
            }
        }
    )

    if fit_kwargs:
        hyperparameters.update_parameters(fit_kwargs)

    gap_default_hyperparameters = hyperparameters.model_dump(by_alias=True)

    include_two_body = gap_default_hyperparameters["general"]["two_body"]
    include_three_body = gap_default_hyperparameters["general"]["three_body"]
    include_soap = gap_default_hyperparameters["general"]["soap"]

    if include_two_body:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_2b, num_triplet = calculate_delta(db_atoms, ref_energy_name)
            gap_default_hyperparameters["twob"].update({"delta": delta_2b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
        )

        run_gap(num_processes_fit, fit_parameters_list)

        run_quip(num_processes_fit, train_data_path, gap_file_xml, quip_train_file)

    if include_three_body:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_3b = energy_remain(quip_train_file)
            delta_3b = delta_3b / num_triplet
            gap_default_hyperparameters["threeb"].update({"delta": delta_3b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
        )

        run_gap(num_processes_fit, fit_parameters_list)
        run_quip(num_processes_fit, train_data_path, gap_file_xml, quip_train_file)

    if glue_xml:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        gap_default_hyperparameters["general"].update({"core_param_file": "glue.xml"})
        gap_default_hyperparameters["general"].update({"core_ip_args": "{IP Glue}"})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=False,
            include_three_body=False,
        )

        run_gap(num_processes_fit, fit_parameters_list)
        run_quip(
            num_processes_fit,
            train_data_path,
            gap_file_xml,
            quip_train_file,
            glue_xml,
        )

    if include_soap:
        delta_soap = (
            energy_remain(quip_train_file)
            if include_two_body or include_three_body
            else 1
        )
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            gap_default_hyperparameters["soap"].update({"delta": delta_soap})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
            include_soap=include_soap,
        )

        run_gap(num_processes_fit, fit_parameters_list)
        run_quip(
            num_processes_fit,
            train_data_path,
            gap_file_xml,
            quip_train_file,
            glue_xml,
        )

    # Calculate training error

    train_error = energy_remain(quip_train_file)
    logging.info(f"Training error of MLIP (eV/at.): {round(train_error, 7)}")

    # Calculate testing error
    run_quip(num_processes_fit, test_data_path, gap_file_xml, quip_test_file, glue_xml)
    test_error = energy_remain(quip_test_file)
    logging.info(f"Testing error of MLIP (eV/at.): {round(test_error, 7)}")

    if not glue_xml and species_list:
        try:
            plot_energy_forces(
                title="Data error metrics",
                energy_limit=0.005,
                force_limit=0.1,
                species_list=species_list,
                train_name=train_name,
                test_name=test_name,
            )
        except (ValueError, XYZError) as e:
            logging.warning(f"Skipped fit error metrics plot because of: \n{e}")

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": mlip_path,
    }


@requires(
    (
        subprocess.run(
            'julia -e "using Pkg; println(haskey(Pkg.dependencies(), '
            'Base.UUID(\\"3b96b61c-0fcc-4693-95ed-1ef9f35fcc53\\")))"',
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    == "true",
    "J-ACE fitting requires the executable 'julia' and ACEPotentials.jl v0.6.7 library to be in PATH. "
    "Please follow the instructions in the autoplex documentation to install the required julia dependencies "
    "and add them to PATH. Link to the documentation:"
    " https://autoatml.github.io/autoplex/user/index.html#standard-installation",
)
def jace_fitting(
    db_dir: str | Path,
    hyperparameters: JACE_HYPERS = JACE_HYPERS,
    isolated_atom_energies: dict | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    num_processes_fit: int = 32,
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Perform the ACE (Atomic Cluster Expansion) potential fitting.

    This function sets up and executes a Julia script to perform ACE fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the ACE model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: str or Path
        directory containing the training and testing data files.
    hyperparameters: MLIP_HYPERS.J_ACE
        Fit hyperparameters.
    isolated_atom_energies: dict:
        mandatory dictionary mapping element numbers to isolated energies.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    num_processes_fit: int
        number of processes to use for parallel computation.
    fit_kwargs: dict.
        optional dictionary with parameters for ace fitting with keys same as
        mlip-rss-defaults.json.

    Keyword Arguments
    -----------------
    order: int
        order of ACE.
    totaldegree: int
        total degree of the polynomial terms in the ACE model.
    cutoff: float
        cutoff distance for atomic interactions in the ACE model.
    solver: str
        solver to be used for fitting the ACE model. Default is "BLR" (Bayesian Linear Regression).
        For very large-scale parameter estimation problems, using "LSQR" solver.

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    Raises
    ------
    - ValueError: If the `isolated_atom_energies` dictionary is empty or not provided when required.
    """
    hyperparameters = hyperparameters.model_copy(deep=True)
    train_atoms = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    source_file_path = os.path.join(db_dir, "test.extxyz")
    shutil.copy(source_file_path, ".")
    isolated_atom_energies_update = {}

    if isolated_atom_energies:
        for e_num, e_energy in isolated_atom_energies.items():
            isolated_atom_energies_update[chemical_symbols[int(e_num)]] = e_energy
    else:
        raise ValueError("isolated_atom_energies parameter is empty or not defined!")

    formatted_isolated_atom_energies = (
        "["
        + ", ".join(
            [
                f":{key} => {value}"
                for key, value in isolated_atom_energies_update.items()
            ]
        )
        + "]"
    )
    formatted_species = (
        "["
        + ", ".join([f":{key}" for key, value in isolated_atom_energies_update.items()])
        + "]"
    )

    train_ace = [
        at for at in train_atoms if "IsolatedAtom" not in at.info["config_type"]
    ]
    ase.io.write("train_ace.extxyz", train_ace, format="extxyz")

    if fit_kwargs:
        hyperparameters.update_parameters(fit_kwargs)

    jace_hypers = hyperparameters.model_dump(by_alias=True)

    order = jace_hypers["order"]
    totaldegree = jace_hypers["totaldegree"]
    cutoff = jace_hypers["cutoff"]
    solver = jace_hypers["solver"]

    ace_text = f"""using ACEpotentials
using LinearAlgebra: norm, Diagonal
using CSV, DataFrames
using Distributed
addprocs({num_processes_fit - 1}, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

data_file = "train_ace.extxyz"
data = read_extxyz(data_file)
test_data_file = "test.extxyz"
test_data = read_extxyz(test_data_file)
data_keys = (energy_key = "{ref_energy_name}", force_key = "{ref_force_name}", virial_key = "{ref_virial_name}")

model = acemodel(elements={formatted_species},
                order={order},
                totaldegree={totaldegree},
                rcut={cutoff},
                Eref={formatted_isolated_atom_energies})

weights = Dict(
            "crystal" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
            "RSS" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "amorphous" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "liquid" => Dict("E" => 10.0, "F" => 0.5 , "V" => 0.25 ),
            "RSS_initial" => Dict("E" => 1.0, "F" => 0.5 , "V" => 0.1 ),
            "dimer" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 )
            )

P = smoothness_prior(model; p = 4)

solver = ACEfit.{solver}()

acefit!(model, data; solver=solver, weights=weights, prior = P, data_keys...)

@info("Training Error Table")
ACEpotentials.linear_errors(data, model; data_keys...)

@info("Testing Error Table")
ACEpotentials.linear_errors(test_data, model; data_keys...)

@info("Manual RMSE Test")
potential = model.potential
train_energies = [ JuLIP.get_data(at, "{ref_energy_name}") / length(at) for at in data]
model_energies_train = [energy(potential, at) / length(at) for at in data]
rmse_energy_train = norm(train_energies - model_energies_train) / sqrt(length(data))
test_energies = [ JuLIP.get_data(at, "{ref_energy_name}") / length(at) for at in test_data]
model_energies_pred = [energy(potential, at) / length(at) for at in test_data]
rmse_energy_test = norm(test_energies - model_energies_pred) / sqrt(length(test_data))

df = DataFrame(rmse_energy_train = rmse_energy_train, rmse_energy_test = rmse_energy_test)
CSV.write("rmse_energies.csv", df)

save_potential("acemodel.json", model)
export2lammps("acemodel.yace", model)
    """

    with open("ace.jl", "w") as file:
        file.write(ace_text)

    os.system(f"export OMP_NUM_THREADS={num_processes_fit} && julia ace.jl")

    energy_err = pd.read_csv("rmse_energies.csv")
    train_error = energy_err["rmse_energy_train"][0]
    test_error = energy_err["rmse_energy_test"][0]

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": Path.cwd(),
    }


def nep_fitting(
    db_dir: str | Path,
    hyperparameters: NEP_HYPERS = NEP_HYPERS,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    species_list: list | None = None,
    gpu_identifier_indices: list[int] = list[0],
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Perform the NEP (Neural evolution Potential) model fitting.

    Parameters
    ----------
    db_dir: Path
        Directory containing the training and testing data files.
    path_to_hyperparameters : str or Path.
        Path to JSON file containing the M3GNet hyperparameters.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    species_list: list
        List of element names (strings)
    gpu_identifier_indices: list[int]
        Indices that identifies the GPU that NEP should be run with
    fit_kwargs: dict.
        optional dictionary with parameters for NEP fitting with keys same as
        mlip-rss-defaults.json.

    Keyword Arguments
    -----------------
    version: int
        NEP model version to train can be 3 or 4. Default is 4.
    type: list[int, str]
        Number of atom types and list of chemical species. Number
        of atom types must be an integer, followed by chemical
        symbols of species as in periodic table for which model
        needs to be trained, separated by comma.
        Default is [1, "X"] as a placeholder. Example:
        [2, "Pb", "Te"].
    type_weight: float
        Weights for different chemical species. Default is 1.0
    model_type: int
        Type of model that is being trained. Can be 0 (potential),
        1 (dipole), 2 (polarizability). Default is 0.
    prediction: int
        Mode of NEP run. Set 0 for training and 1 for inference.
        Default is 0.
    cutoff: list[int, int]
        Radial and angular cutoff. First element is for radial cutoff and
        second element is for angular cutoff. Default is [6, 5].
    n_max: list[int, int]
        Number of radial and angular descriptors. First element is for radial
        and second element is for angular. Default is [4, 4].
    basis_size: list[int, int]
        Number of basis functions that are used to build the radial and angular
        descriptor. First element is for radial descriptor and
        second element is for angular descriptor. Default is [8, 8].
    l_max: list[int, int, int]
       The maximum expansion order for the angular terms. First element is for
       three-body, second element is for four-body and third element is for five-body.
       Default is [4, 2, 1].
    neuron: int
        Number of neurons in the hidden layer. Default is 80.
    lambda_1: float
        Weight for L1 regularization. Default is 0.
    lambda_e: float
        Weight for energy loss. Default is 1.
    lambda_f: float
        Weight for force loss. Default is 1.
    lambda_v: float
        Weight for virial loss. Default is 0.1.
    force_delta: float
        Sets bias the on the loss function to put more emphasis on obtaining
        accurate predictions for smaller forces. Default is 0.
    batch: int
        Batch size for training. Default is 1000.
    population: int
        Size of the population used by the SNES algorithm. Default is 50.
    generation: bool
        Sets the max number of generations for SNES algorithm.
    zbl : float
        Cutoff to use in universal ZBL potential at short distances.
        Acceptable values are in range 1 to 2.5. Default is 2.

    References
    ----------
    * GPUMD & NEP: https://doi.org/10.1063/5.0106617.
    * SNES : https://doi.org/10.1145/2001576.2001692.
    * Parameter defaults taken from SI: https://doi.org/10.1038/s41467-024-54554-x.

    Returns
    -------
    dict[str, float]
        A dictionary mapping 'train_error', 'test_error', and 'mlip_path'.
    """
    hyperparameters = hyperparameters.model_copy(deep=True)

    train_data = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    test_data = ase.io.read(os.path.join(db_dir, "test.extxyz"), index=":")

    try:
        train_nep = [
            at for at in train_data if "IsolatedAtom" not in at.info["config_type"]
        ]
        test_nep = [
            at for at in test_data if "IsolatedAtom" not in at.info["config_type"]
        ]
    except KeyError:
        train_nep = train_data
        test_nep = test_data

    # Use the SinglePointCalculator to set the energy, forces, and virial
    # Step required to generate NEP compatible xyz file using write_structures from calorine
    for at in train_data:
        at.calc = SinglePointCalculator(
            at, energy=at.info[ref_energy_name], forces=at.arrays[ref_force_name]
        )
        at.info["virial"] = at.info[ref_virial_name]
        del at.info[ref_energy_name]
        del at.info[ref_virial_name]
        del at.arrays[ref_force_name]

    for at in test_data:
        at.calc = SinglePointCalculator(
            at, energy=at.info[ref_energy_name], forces=at.arrays[ref_force_name]
        )
        at.info["virial"] = at.info[ref_virial_name]
        del at.info[ref_energy_name]
        del at.info[ref_virial_name]
        del at.arrays[ref_force_name]

    write_structures(outfile="train.xyz", structures=train_nep)
    write_structures(outfile="test.xyz", structures=test_nep)

    if fit_kwargs:
        hyperparameters.update_parameters(fit_kwargs)

    nep_hypers = hyperparameters.model_dump(by_alias=True)

    nep_hypers["type"] = [len(species_list), *species_list]
    nep_hypers["type_weight"] = [1.0] * len(species_list)

    write_nepfile(parameters=nep_hypers, dirname=".")
    run_nep(gpu_identifier_indices=gpu_identifier_indices)

    metrics_df = read_loss("loss.out")

    return {
        "train_error": metrics_df.RMSE_E_train.values[-1],
        "test_error": metrics_df.RMSE_E_test.values[-1],
        "mlip_path": Path.cwd(),
    }


def nequip_fitting(
    db_dir: Path,
    hyperparameters: NEQUIP_HYPERS = NEQUIP_HYPERS,
    isolated_atom_energies: dict | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    fit_kwargs: dict | None = None,
    device: str = "cuda",
) -> dict:
    """
    Perform the NequIP potential fitting.

    This function sets up and executes a python script to perform NequIP fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the NequIP model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: Path
        directory containing the training and testing data files.
    hyperparameters: MLIP_HYPERS.NEQUIP
        Fit hyperparameters.
    isolated_atom_energies: dict
        mandatory dictionary mapping element numbers to isolated energies.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    device: str
        specify device to use cuda or cpu
    fit_kwargs: dict.
        optional dictionary with parameters for nequip fitting with keys same as
        mlip-rss-defaults.json.

    Keyword Arguments
    -----------------
    r_max: float
        cutoff radius in length units
    num_layers: int
        number of interaction blocks
    l_max: int
        maximum irrep order (rotation order) for the network's features
    num_features: int
        multiplicity of the features
    num_basis: int
        number of basis functions used in the radial basis
    invariant_layers: int
        number of radial layers
    invariant_neurons: int
        number of hidden neurons in radial function
    batch_size: int
        batch size
    learning_rate: float
        learning rate
    default_dtype: str
        type of float to use, e.g. float32 and float64

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    Raises
    ------
    - ValueError: If the `isolated_atom_energies` dictionary is empty or not provided when required.
    """
    """
    [TODO] train Nequip on virials
    """
    hyperparameters = hyperparameters.model_copy(deep=True)

    train_data = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    train_nequip = [
        at for at in train_data if "IsolatedAtom" not in at.info["config_type"]
    ]
    ase.io.write("train_nequip.extxyz", train_nequip, format="extxyz")

    test_data = ase.io.read(os.path.join(db_dir, "test.extxyz"), index=":")
    num_of_train = len(train_nequip)
    num_of_val = len(test_data)

    isolated_atom_energies_update = ""
    ele_syms = []
    if isolated_atom_energies:
        for e_num in isolated_atom_energies:
            element_symbol = "  - " + chemical_symbols[int(e_num)] + "\n"
            isolated_atom_energies_update += element_symbol
            ele_syms.append(chemical_symbols[int(e_num)])
    else:
        raise ValueError("isolated_atom_energies is empty or not defined!")

    nequip_config_updates = {
        "dataset_key_mapping": {
            f"{ref_energy_name}": "total_energy",
            f"{ref_force_name}": "forces",
        },
        "validation_dataset_key_mapping": {
            f"{ref_energy_name}": "total_energy",
            f"{ref_force_name}": "forces",
        },
        "chemical_symbols": ele_syms,
        "dataset_file_name": "./train_nequip.extxyz",
        "validation_dataset_file_name": f"{db_dir}/test.extxyz",
        "n_train": num_of_train,
        "n_val": num_of_val,
    }
    hyperparameters.update_parameters(nequip_config_updates)

    if fit_kwargs:
        hyperparameters.update_parameters(fit_kwargs)

    nequip_hypers = hyperparameters.model_dump(by_alias=True)

    dumpfn(nequip_hypers, "nequip.yaml")

    run_nequip("nequip-train nequip.yaml", "nequip_train")
    run_nequip(
        "nequip-deploy build --train-dir results/autoplex ./deployed_nequip_model.pth",
        "nequip_deploy",
    )

    calc = NequIPCalculator.from_deployed_model(
        model_path="deployed_nequip_model.pth",
        device=device,
        species_to_type_name={s: s for s in ele_syms},
        set_global_options=False,
    )

    ener_out_train = []
    for at in train_nequip:
        at.calc = calc
        ener_out_train.append(at.get_potential_energy() / len(at))

    ener_in_train = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in train_nequip
    ]

    train_error = rms_dict(ener_in_train, ener_out_train)["rmse"]

    ener_out_test = []
    for at in test_data:
        at.calc = calc
        ener_out_test.append(at.get_potential_energy() / len(at))

    ener_in_test = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in test_data
    ]

    test_error = rms_dict(ener_in_test, ener_out_test)["rmse"]

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": Path.cwd(),
    }


def m3gnet_fitting(
    db_dir: Path,
    hyperparameters: M3GNET_HYPERS = M3GNET_HYPERS,
    device: str = "cuda",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    test_equal_to_val: bool = True,
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Perform the M3GNet potential fitting.

    Parameters
    ----------
    db_dir: Path
        Directory containing the training and testing data files.
    hyperparameters: MLIP_HYPERS.M3GNET
        Fit hyperparameters.
    device: str
        Device on which the model will be trained, e.g., 'cuda' or 'cpu'.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    test_equal_to_val: bool
        If True, the testing dataset will be the same as the validation dataset.
    fit_kwargs: dict.
        optional dictionary with parameters for M3GNET fitting.

    Keyword Arguments
    -----------------
    exp_name: str
        Name of the experiment, used for saving model checkpoints and logs.
    results_dir: str
        Directory to store the training results and fitted model.
    cutoff: float
        Cutoff radius for atomic interactions in length units.
    threebody_cutoff: float
        Cutoff radius for three-body interactions in length units.
    batch_size: int
        Number of structures per batch during training.
    max_epochs: int
        Maximum number of training epochs.
    include_stresses: bool
        If True, includes stress tensors in the model predictions and training process.
    dim_node_embedding: int
         Dimension of node embedding.
    dim_edge_embedding: int
        Dimension of edge embeddings.
    units: int
        Number of units in each dense layer of the model.
    max_l: int
        Maximum degree of spherical harmonics.
    max_n: int
        Maximum radial function degree.

    Returns
    -------
    dict[str, float]
        A dictionary containing keys such as 'train_error', 'test_error', and 'path_to_fitted_model',
        representing the training error, test error, and the location of the saved model, respectively.

    References
    ----------
    *    Title: Tutorials of Materials Graph Library (MatGL)
    *    Author: Tsz Wai Ko, Chi Chen and Shyue Ping Ong
    *    Version: 1.1.3
    *    Date 7/8/2024
    *    Availability: https://matgl.ai/tutorials%2FTraining%20a%20M3GNet%20Potential%20with%20PyTorch%20Lightning.html
    *    License: BSD 3-Clause License
    """
    hyperparameters = hyperparameters.model_copy(deep=True)

    if fit_kwargs:
        hyperparameters.update_parameters(fit_kwargs)

    m3gnet_hypers = hyperparameters.model_dump(by_alias=True)

    exp_name = m3gnet_hypers["exp_name"]
    results_dir = m3gnet_hypers["results_dir"]

    os.makedirs(os.path.join(results_dir, exp_name), exist_ok=True)

    with open("output.txt", "w") as f:
        # Backup original stdout stream.
        original_stdout = sys.stdout

        # Set stdout to the file object.
        sys.stdout = f

        # Print something (it goes to the file).
        print("This line will be written to the file.")

        # Restore original stdout stream.
        sys.stdout = original_stdout

    with open("m3gnet.log", "w") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log_file
        sys.stderr = log_file

        train_data = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
        train_m3gnet = [
            at
            for at in train_data
            if "IsolatedAtom" not in at.info["config_type"]
            and "dimer" not in at.info["config_type"]
        ]

        # prepare train dataset
        (
            train_structs,
            train_energies,
            train_forces,
            train_stresses,
        ) = convert_xyz_to_structure(
            train_m3gnet,
            include_forces=True,
            include_stresses=m3gnet_hypers.get("include_stresses"),
            ref_energy_name=ref_energy_name,
            ref_force_name=ref_force_name,
            ref_virial_name=ref_virial_name,
        )

        train_labels = {
            "energies": train_energies,
            "forces": train_forces,
            "stresses": train_stresses,
        }
        train_element_types = get_element_list(train_structs)

        print(
            train_element_types
        )  # this print has to stay as the stdout is written to the file
        train_converter = Structure2Graph(
            element_types=train_element_types, cutoff=m3gnet_hypers.get("cutoff")
        )
        train_datasets = MGLDataset(
            threebody_cutoff=m3gnet_hypers.get("threebody_cutoff"),
            structures=train_structs,
            converter=train_converter,
            labels=train_labels,
            include_line_graph=True,
            filename="dgl_graph_train.bin",
            filename_lattice="lattice_train.pt",
            filename_line_graph="dgl_line_graph_train.bin",
            filename_state_attr="state_attr_train.pt",
            filename_labels="labels_train.json",
            save_dir=os.path.join(results_dir, exp_name),
        )

        if os.path.exists(os.path.join(db_dir, "test.extxyz")):
            test_data = ase.io.read(os.path.join(db_dir, "test.extxyz"), index=":")
            # prepare test dataset
            (
                test_structs,
                test_energies,
                test_forces,
                test_stresses,
            ) = convert_xyz_to_structure(
                test_data,
                include_forces=True,
                include_stresses=m3gnet_hypers.get("include_stresses"),
                ref_energy_name=ref_energy_name,
                ref_force_name=ref_force_name,
                ref_virial_name=ref_virial_name,
            )

            test_labels = {
                "energies": test_energies,
                "forces": test_forces,
                "stresses": test_stresses,
            }
            test_element_types = get_element_list(test_structs)
            test_converter = Structure2Graph(
                element_types=test_element_types, cutoff=m3gnet_hypers.get("cutoff")
            )
            test_dataset = MGLDataset(
                threebody_cutoff=m3gnet_hypers.get("threebody_cutoff"),
                structures=test_structs,
                converter=test_converter,
                labels=test_labels,
                include_line_graph=True,
                filename="dgl_graph_test.bin",
                filename_lattice="lattice_test.pt",
                filename_line_graph="dgl_line_graph_test.bin",
                filename_state_attr="state_attr_test.pt",
                filename_labels="labels_test.json",
                save_dir=os.path.join(results_dir, exp_name),
            )

        if test_equal_to_val:
            train_dataset = train_datasets
            val_dataset = test_dataset
        else:
            if os.path.exists(os.path.join(db_dir, "test.extxyz")):
                train_dataset, val_dataset, _ = split_dataset(
                    train_datasets,
                    frac_list=[0.9, 0.1, 0],  # to guarantee train:valid=9:1
                    shuffle=True,
                    random_state=42,
                )
            else:
                train_dataset, val_dataset, test_dataset = split_dataset(
                    train_datasets,
                    frac_list=[0.8, 0.1, 0.1],  # to guarantee train:valid:test=8:1:1
                    shuffle=True,
                    random_state=42,
                )

        my_collate_fn = partial(
            collate_fn_pes, include_line_graph=True
        )  # Set all include_line_graph to False will disable three-body interactions
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=my_collate_fn,
            batch_size=m3gnet_hypers.get("batch_size"),
            num_workers=1,
        )
        # train from scratch
        if not m3gnet_hypers["foundation_model"]:  # train from scratch
            model = M3GNet(
                element_types=train_element_types,
                is_intensive=m3gnet_hypers.get("is_intensive"),
                cutoff=m3gnet_hypers.get("cutoff"),
                threebody_cutoff=m3gnet_hypers.get("threebody_cutoff"),
                dim_node_embedding=m3gnet_hypers.get("dim_node_embedding"),
                dim_edge_embedding=m3gnet_hypers.get("dim_edge_embedding"),
                units=m3gnet_hypers.get("units"),
                max_l=m3gnet_hypers.get("max_l"),
                max_n=m3gnet_hypers.get("max_n"),
                nblocks=m3gnet_hypers.get("nblocks"),
            )
            lit_module = PotentialLightningModule(
                model=model,
                element_refs=m3gnet_hypers.get("element_refs"),
                include_line_graph=m3gnet_hypers.get("include_line_graph"),
                allow_missing_labels=m3gnet_hypers.get("allow_missing_labels"),
                energy_weight=m3gnet_hypers.get("energy_weight"),
                force_weight=m3gnet_hypers.get("force_weight"),
                lr=m3gnet_hypers.get("lr"),
                loss=m3gnet_hypers.get("loss"),
                loss_params=m3gnet_hypers.get("loss_params"),
                stress_weight=m3gnet_hypers.get("stress_weight"),
                magmom_weight=m3gnet_hypers.get("magmom_weight"),
                data_mean=m3gnet_hypers.get("data_mean"),
                data_std=m3gnet_hypers.get("data_std"),
                decay_alpha=m3gnet_hypers.get("decay_alpha"),
                decay_steps=m3gnet_hypers.get("decay_steps"),
                sync_dist=m3gnet_hypers.get("sync_dist"),
                magmom_target=m3gnet_hypers.get("magmom_target"),
                optimizer=m3gnet_hypers.get("optimizer"),
                scheduler=m3gnet_hypers.get("scheduler"),
            )
        else:  # finetune a foundation model (pretrained model)
            logging.info(
                f"Finetuning foundation model: {m3gnet_hypers['foundation_model']}"
            )
            m3gnet_nnp = matgl.load_model(m3gnet_hypers["foundation_model"])
            model = m3gnet_nnp.model
            property_offset = (
                m3gnet_nnp.element_refs.property_offset
                if m3gnet_hypers["use_foundation_model_element_refs"]
                else None
            )
            lit_module = PotentialLightningModule(
                model=model,
                element_refs=property_offset,
                include_line_graph=m3gnet_hypers.get("include_line_graph"),
                allow_missing_labels=m3gnet_hypers.get("allow_missing_labels"),
                energy_weight=m3gnet_hypers.get("energy_weight"),
                force_weight=m3gnet_hypers.get("force_weight"),
                lr=m3gnet_hypers.get("lr"),
                loss=m3gnet_hypers.get("loss"),
                loss_params=m3gnet_hypers.get("loss_params"),
                stress_weight=m3gnet_hypers.get("stress_weight"),
                magmom_weight=m3gnet_hypers.get("magmom_weight"),
                data_mean=m3gnet_hypers.get("data_mean"),
                data_std=m3gnet_hypers.get("data_std"),
                decay_alpha=m3gnet_hypers.get("decay_alpha"),
                decay_steps=m3gnet_hypers.get("decay_steps"),
                sync_dist=m3gnet_hypers.get("sync_dist"),
                magmom_target=m3gnet_hypers.get("magmom_target"),
                optimizer=m3gnet_hypers.get("optimizer"),
                scheduler=m3gnet_hypers.get("scheduler"),
            )

        logger = CSVLogger(name=exp_name, save_dir=os.path.join(results_dir, "logs"))
        # Inference mode = False is required for calculating forces, stress in test mode and prediction mode
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                torch.cuda.set_device(torch.device(f"cuda:{gpu_id}"))
                trainer = pl.Trainer(
                    max_epochs=m3gnet_hypers.get("max_epochs"),
                    accelerator="gpu",
                    logger=logger,
                    inference_mode=False,
                )
            else:
                raise ValueError("CUDA is not available.")
        else:
            trainer = pl.Trainer(
                max_epochs=m3gnet_hypers.get("max_epochs"),
                accelerator="cpu",
                logger=logger,
                inference_mode=False,
            )
        # Again loggers ...
        print("Start training...")
        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of val_loader: {len(val_loader)}")
        print(f"Length of test_loader: {len(test_loader)}")
        trainer.fit(
            model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        # test the model, remember to set inference_mode=False in trainer (see above)
        print("Train error:")
        trainer.test(dataloaders=train_loader)
        print("Valid error:")
        trainer.test(dataloaders=val_loader)
        print("Test error:")
        trainer.test(dataloaders=test_loader)

        # save trained model
        model_export_path = os.path.join(results_dir, exp_name)
        # model.save(model_export_path)
        potential = Potential(model=model)
        potential.save(model_export_path)

        sys.stdout = original_stdout
        sys.stderr = original_stderr

    for fn in (
        "dgl_graph_train.bin",
        "lattice_train.pt",
        "dgl_line_graph_train.bin",
        "state_attr_train.pt",
        "labels_train.json",
        "dgl_graph_test.bin",
        "lattice_test.pt",
        "dgl_line_graph_test.bin",
        "state_attr_test.pt",
        "labels_test.json",
    ):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(results_dir, exp_name, fn))

    sections = {
        "Train error:": {
            "train_Energy_RMSE": "test_Energy_RMSE",
            "train_Force_RMSE": "test_Force_RMSE",
        },
        "Valid error:": {
            "val_Energy_RMSE": "test_Energy_RMSE",
            "val_Force_RMSE": "test_Force_RMSE",
        },
        "Test error:": {
            "test_Energy_RMSE": "test_Energy_RMSE",
            "test_Force_RMSE": "test_Force_RMSE",
        },
    }

    extracted_values = {}
    with open("m3gnet.log") as file:
        content = file.read()

        for section, metrics in sections.items():
            start_index = content.find(section)
            if start_index != -1:
                next_index = min(
                    [
                        content.find(sec, start_index + 1)
                        for sec in sections
                        if content.find(sec, start_index + 1) != -1
                    ],
                    default=len(content),
                )
                section_content = content[start_index:next_index]
                for key, metric in metrics.items():
                    for line in section_content.split("\n"):
                        if metric in line:
                            if metric in line.split()[0]:
                                extracted_values[key] = float(line.split()[1])
                            else:
                                extracted_values[key] = float(line.split()[3])

    for key, value in extracted_values.items():
        print(f"{key}: {value}")

    """
    !!![Note] The RMSE directly outputted from Torch is not strictly the RMSE of the full datasets;
    it is related to the batch size. It only becomes a strict RMSE when the batch size is larger
    than the size of the dataset. The output here can be considered as an approximate result.
    [TODO] Switch it to the strict RMSE.
    """
    mlip_path = Path.cwd() / model_export_path

    return {
        "train_error": extracted_values["train_Energy_RMSE"],
        "test_error": extracted_values["test_Energy_RMSE"],
        "mlip_path": mlip_path,
    }


def mace_fitting(
    db_dir: Path,
    hyperparameters: MACE_HYPERS = MACE_HYPERS,
    device: str = "cuda",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Perform the MACE potential fitting.

    This function sets up and executes a python script to perform MACE fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the NequIP model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: Path
        directory containing the training and testing data files.
    hyperparameters: MLIP_HYPERS.MACE
        Fit hyperparameters.
    device: str
        specify device to use cuda or cpu.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    fit_kwargs: dict.
        optional dictionary with parameters for mace fitting with keys same as
        mlip-rss-defaults.json.

    Keyword Arguments
    -----------------
    model: str
        type of model to be trained
    config_type_weights: str
        weights of config types
    hidden_irreps: str
        control the model size
    r_max: float
        cutoff radius controls the locality of the model
    batch_size: int
        batch size (note that batch size cannot be larger than the size of training datasets)
    start_swa: str
        if the keyword --swa is enabled, the energy weight of the loss is increased
        for the last ~20% of the training epochs (from --start_swa epochs)
    correlation: int
        correlation order corresponds to the order that MACE induces at each layer
    loss: str
        loss functions
    default_dtype: str
        type of float to use, e.g. float32 and float64

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    """
    hyperparameters = hyperparameters.model_copy(deep=True)

    if ref_virial_name is not None:
        atoms = read(f"{db_dir}/train.extxyz", index=":")
        mace_virial_format_conversion(
            atoms=atoms, ref_virial_name=ref_virial_name, out_file_name="train.extxyz"
        )

    hyperparameters.update_parameters(fit_kwargs)

    mace_hypers = hyperparameters.model_dump(by_alias=True, exclude_none=True)

    boolean_hypers = [
        "distributed",
        "pair_repulsion",
        "amsgrad",
        "swa",
        "stage_two",
        "keep_checkpoint",
        "save_all_checkpoints",
        "restart_latest",
        "save_cpu",
        "wandb",
        "compute_statistics",
        "foundation_model_readout",
        "ema",
    ]
    boolean_str_hypers = [
        "compute_avg_num_neighbors",
        "compute_stress",
        "compute_forces",
        "multi_processed_test",
        "pin_memory",
        "foundation_filter_elements",
        "multiheads_finetuning",
        "keep_isolated_atoms",
        "shuffle",
    ]

    hypers = []
    for hyper in mace_hypers:
        if hyper in boolean_hypers:
            if mace_hypers[hyper] is True:
                hypers.append(f"--{hyper}")
        elif hyper in boolean_str_hypers:
            hypers.append(f"--{hyper}={mace_hypers[hyper]}")
        elif hyper in ["train_file", "test_file"]:
            logging.info("Train and test files have default names.")
        elif hyper in ["energy_key", "virial_key", "forces_key", "device"]:
            logging.info("energy_key, virial_key and forces_key have default names.")
        else:
            hypers.append(f"--{hyper}={mace_hypers[hyper]}")

    hypers.append(f"--train_file={db_dir}/train.extxyz")
    hypers.append(f"--valid_file={db_dir}/test.extxyz")

    if ref_energy_name is not None:
        hypers.append(f"--energy_key={ref_energy_name}")
    if ref_force_name is not None:
        hypers.append(f"--forces_key={ref_force_name}")
    if ref_virial_name is not None:
        hypers.append(f"--virials_key={ref_virial_name}")
    if device is not None:
        hypers.append(f"--device={device}")

    run_mace(hypers)

    try:
        with open("./logs/MACE_model_run-123.log") as file:
            log_data = file.read()
    except FileNotFoundError:
        # to cover finetuning
        try:
            with open(f"./logs/{fit_kwargs['name']}_run-123.log") as file:
                log_data = file.read()
        except FileNotFoundError:
            try:
                with open("./logs/MACE_final_run-3.log") as file:
                    log_data = file.read()
            except FileNotFoundError:
                with open(f"./logs/{fit_kwargs['name']}_run-3.log") as file:
                    log_data = file.read()

    tables = re.split(r"\+-+\+\n", log_data)
    # if tables:
    last_table = tables[-2]
    try:
        matches = re.findall(
            r"\|\s*(train_default|valid_default)\s*\|\s*([\d\.]+)\s*\|", last_table
        )

        return {
            "train_error": float(matches[0][1]),
            "test_error": float(matches[1][1]),
            "mlip_path": Path.cwd(),
        }
    except IndexError:
        # to ensure backward compatibility to mace 0.3.4
        matches = re.findall(r"\|\s*(train|valid)\s*\|\s*([\d\.]+)\s*\|", last_table)

        return {
            "train_error": float(matches[0][1]),
            "test_error": float(matches[1][1]),
            "mlip_path": Path.cwd(),
        }


def check_convergence(test_error: float) -> bool:
    """
    Check the convergence of the fit.

    Parameters
    ----------
    test_error:
        The error of the test data.

    Returns
    -------
    The convergence bool.
    """
    convergence = False
    if test_error < 0.01:
        convergence = True

    return convergence


def gap_hyperparameter_constructor(
    gap_parameter_dict: dict,
    include_two_body: bool = False,
    include_three_body: bool = False,
    include_soap: bool = False,
) -> list:
    """
    Construct a list of arguments needed to execute gap potential from the parameters' dict.

    Parameters
    ----------
    gap_parameter_dict : dict.
        dictionary with gap hyperparameters.
    include_two_body : bool.
        bool indicating whether to include two-body hyperparameters
    include_three_body : bool.
        bool indicating whether to include three-body hyperparameters
    include_soap : bool.
        bool indicating whether to include soap hyperparameters

    Returns
    -------
        list
           gap fit input parameter string.
    """
    dict_wo_term_name = gap_parameter_dict.copy()
    if "two_body" in dict_wo_term_name["general"]:
        del dict_wo_term_name["general"]["two_body"]
    if "three_body" in dict_wo_term_name["general"]:
        del dict_wo_term_name["general"]["three_body"]
    if "soap" in dict_wo_term_name["general"]:
        del dict_wo_term_name["general"]["soap"]

    general = [f"{key}={value}" for key, value in dict_wo_term_name["general"].items()]

    two_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in dict_wo_term_name["twob"].items()
            if include_two_body is True
        ]
    )

    three_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in dict_wo_term_name["threeb"].items()
            if include_three_body is True
        ]
    )
    soap_params = " ".join(
        [
            f"{key}={value}"
            for key, value in dict_wo_term_name["soap"].items()
            if include_soap is True
        ]
    )
    # add separator between the arg types
    if include_two_body and include_three_body and include_soap:
        three_body_params = " :" + three_body_params
        soap_params = " :soap " + soap_params
    elif include_two_body and include_three_body and not include_soap:
        three_body_params = " :" + three_body_params
    elif (include_two_body or include_three_body) and include_soap:
        soap_params = " :soap " + soap_params
    elif include_soap and not include_three_body and not include_two_body:
        soap_params = "soap " + soap_params

    gap_hyperparameters = f"gap={{{two_body_params}{three_body_params}{soap_params}}}"

    return [*general, gap_hyperparameters]


def get_list_of_vasp_calc_dirs(flow_output) -> list[str]:
    """
    Return a list of vasp_calc_dirs from PhononDFTMLDataGenerationFlow output.

    Parameters
    ----------
    flow_output: dict.
        PhononDFTMLDataGenerationFlow output

    Returns
    -------
    list.
        A list of vasp_calc_dirs
    """
    list_of_vasp_calc_dirs: list[str] = []
    for output in flow_output.values():
        for output_type, dirs in output.items():
            if output_type != "phonon_data" and isinstance(dirs, list):
                if output_type == "rattled_dir":
                    flat_dirs = [[item for sublist in dirs for item in sublist]]
                    list_of_vasp_calc_dirs.extend(*flat_dirs)
                else:
                    list_of_vasp_calc_dirs.extend(*dirs)

    return list_of_vasp_calc_dirs


def vaspoutput_2_extended_xyz(
    path_to_vasp_static_calcs: list,
    config_types: list[str] | None = None,
    data_types: list[str] | None = None,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    regularization: float = 0.1,
    f_min: float = 0.01,  # unit: eV Å-1
    atom_wise_regularization: bool = True,
) -> None:
    """
    Parse all VASP output files (vasprun.xml/OUTCAR) and generates a vasp_ref.extxyz.

    Uses ase.io.read to parse the OUTCARs
    Adapted from https://lipai.github.io/scripts/ml_scripts/outcar2xyz.html

    Parameters
    ----------
    path_to_vasp_static_calcs : list.
        List of VASP static calculation directories.
    config_types: list[str] or None
            list of config_types.
    data_types: list[str] or None
            track the data type (phonon or random).
    ref_energy_name : str
        Reference energy name in xyz file.
    ref_force_name : str
        Reference force name in xyz file.
    ref_virial_name : str
        Reference virial name in xyz file.
    regularization: float
        regularization value for the atom-wise force components.
    f_min: float
        minimal force cutoff value for atom-wise regularization.
    atom_wise_regularization: bool
        for including atom-wise regularization.
    """
    counter = 0
    if config_types is None:
        config_types = ["bulk"] * len(path_to_vasp_static_calcs)
    if data_types is None:
        data_types = ["other"] * len(path_to_vasp_static_calcs)

    for path, config_type, data_type in zip(
        path_to_vasp_static_calcs, config_types, data_types
    ):
        # strip hostname if it exists in the path
        path_without_hostname = Path(strip_hostname(path)).joinpath("vasprun.xml.gz")
        path_without_hostname2 = Path(strip_hostname(path)).joinpath(
            "final_atoms_object.xyz"
        )
        try:
            # read the vasp output
            if path_without_hostname.exists():
                file = read(path_without_hostname, index=":")
            elif path_without_hostname2.exists():
                file = read(path_without_hostname2, index=":")
            for i in file:
                virial_list = (
                    -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
                )
                i.info[ref_virial_name] = " ".join(map(str, virial_list.flatten()))
                del i.calc.results["stress"]
                i.arrays[ref_force_name] = i.calc.results["forces"]
                if atom_wise_regularization and (data_type == "phonon_dir"):
                    atom_forces = np.array(i.arrays[ref_force_name])
                    atom_wise_force = np.array(
                        [
                            force if force > f_min else f_min
                            for force in np.linalg.norm(atom_forces, axis=1)
                        ]
                    )
                    i.arrays["force_atom_sigma"] = regularization * atom_wise_force
                del i.calc.results["forces"]
                i.info[ref_energy_name] = i.calc.results["free_energy"]
                del i.calc.results["energy"]
                del i.calc.results["free_energy"]
                i.info["config_type"] = config_type
                i.info["data_type"] = data_type.rstrip("_dir")
                i.pbc = True

            # TODO: maybe only add isolated atoms energy if it wasn't there?

            write("vasp_ref.extxyz", file, append=True)

        except FileNotFoundError:
            counter += 1

        if counter / len(path_to_vasp_static_calcs) > 0.05:
            raise ValueError(
                "An insufficient number of data points collected. Workflow stopped."
            )


def flatten(atoms_object, recursive=False) -> list[str | bytes | Atoms] | list:
    """
    Flatten an iterable fully, but excluding Atoms objects.

    Parameters
    ----------
    atoms_object: Atoms object
    recursive: bool
        set the recursive boolean.

    Returns
    -------
    a flattened object, excluding the Atoms objects.

    """
    iteration_list: list[str | bytes | Atoms] | list = []

    if recursive:
        for element in atoms_object:
            if isinstance(element, Iterable) and not isinstance(
                element, (str, bytes, ase.atoms.Atoms, ase.Atoms)
            ):
                iteration_list.extend(flatten(element, recursive=True))
            else:
                iteration_list.append(element)
        return iteration_list

    return [item for sublist in atoms_object for item in sublist]


def gcm3_to_Vm(gcm3, mr, n_atoms=1) -> float:
    """
    Convert gcm3 to Vm.

    Parameters
    ----------
    gcm3:
        Density in grams per cubic centimeter (g/cm³).
    mr:
        Molar mass in grams per mole (g/mol).
    n_atoms:
        Number of atoms in the formula unit. Default is 1.

    Returns
    -------
    the converted unit.

    """
    return 1 / (n_atoms * (gcm3 / mr) * 6.022e23 / (1e8) ** 3)


def get_atomic_numbers(species: list) -> list[int]:
    """
    Get atomic numbers.

    Parameters
    ----------
    species:
        type of species

    Returns
    -------
    atomic_numbers:
        list of atomic numbers.

    """
    atom_numbers = []
    for atom_type in species:
        atom = Atoms(atom_type, [(0, 0, 0)])
        atom_numbers.append(int(atom.get_atomic_numbers()[0]))

    return atom_numbers


def energy_remain(in_file: str) -> float:
    """
    Plot the distribution of energy per atom on the output vs. the input.

    Parameters
    ----------
    in_file:
        input file

    Returns
    -------
    rms["rmse"]:
        distribution of energy per atom RMSE of output vs. input.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    if "config_type" in in_atoms[0].info:
        ener_in = [
            at.info["REF_energy"] / len(at.get_chemical_symbols())
            for at in in_atoms
            if at.info["config_type"] != "IsolatedAtoms"
        ]
        ener_out = [
            at.get_potential_energy() / len(at.get_chemical_symbols())
            for at in in_atoms
            if at.info["config_type"] != "IsolatedAtoms"
        ]
    else:
        ener_in = [
            at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in in_atoms
        ]
        ener_out = [
            at.get_potential_energy() / len(at.get_chemical_symbols())
            for at in in_atoms
        ]
    rms = rms_dict(ener_in, ener_out)
    return rms["rmse"]


def extract_gap_label(xml_file_path) -> str:
    """
    Extract GAP label.

    Parameters
    ----------
    xml_file_path:
        path to the GAP fit potential xml file.

    Returns
    -------
    the extracted GAP label.

    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return root.tag


def plot_convex_hull(all_points: np.ndarray, hull_points: np.ndarray) -> None:
    """
    Plot convex hull.

    Parameters
    ----------
    all_points : np.ndarray
        Array of all points to be plotted.
    hull_points : np.ndarray
        Array of points used to calculate the convex hull.
    """
    hull = ConvexHull(hull_points)

    plt.plot(all_points[:, 0], all_points[:, 1], "o", markersize=3, label="All Points")

    for i, simplex in enumerate(hull.simplices):
        if i == 0:
            plt.plot(
                hull_points[simplex, 0],
                hull_points[simplex, 1],
                "k-",
                label="Convex Hull",
            )
        else:
            plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], "k-")

    plt.xlabel("Volume")
    plt.ylabel("Energy")
    plt.title("Convex Hull with All Points")
    plt.legend()
    plt.savefig("ConvexHull.png")


def calculate_delta(atoms_db: list[Atoms], e_name: str) -> tuple[float, ndarray]:
    """
    Calculate the delta parameter and average number of triplets for gap-fitting.

    Parameters
    ----------
    atoms_db: list[Atoms]
        list of Ase atoms objects
    e_name: str
        energy_parameter_name as defined in mlip-phonon-defaults.json

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - delta parameter used for gap-fit, calculated as (es_var / avg_neigh).
        - Average number of triplets per atom.

    """
    at_ids = [atom.get_atomic_numbers() for atom in atoms_db]
    isolated_atom_energies = {
        atom.get_atomic_numbers()[0]: atom.info[e_name]
        for atom in atoms_db
        if "config_type" in atom.info and "IsolatedAtom" in atom.info["config_type"]
    }

    es_visol = np.array(
        [
            (atom.info[e_name] - sum([isolated_atom_energies[j] for j in at_ids[ct]]))
            / len(atom)
            for ct, atom in enumerate(atoms_db)
        ]
    )
    es_var = np.var(es_visol)
    avg_neigh = np.mean([compute_pairs_triplets(atom)[0] for atom in atoms_db])
    num_triplet = np.mean([compute_pairs_triplets(atom)[1] for atom in atoms_db])

    return es_var / avg_neigh, num_triplet


def compute_pairs_triplets(atoms: Atoms) -> list[float]:
    """
    Calculate the number of pairwise and triplet within a cutoff distance for a given list of atoms.

    Parameters
    ----------
    atoms : ASE atoms object

    Returns
    -------
    list[float, float]
        Returns a list of the number of pairs or triplets an atom is involved in.

    """
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(
        cutoffs=cutoffs, skin=0.15, self_interaction=False, bothways=True
    )
    neighbor_list.update(atoms)
    counts_list = [
        len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms))
    ]
    num_pair = sum(counts_list) / len(atoms)

    triplets = [comb(count, 2) for count in counts_list if count > 1]
    num_triplet = sum(triplets) / len(atoms)

    return [num_pair, num_triplet]


def run_ace(num_processes_fit: int, script_name: str) -> None:
    """
    Julia-ACE script runner.

    Parameters
    ----------
    num_processes_fit: int
        Number of threads to be used for the run.
    script_name: str
        Name of the Julia script to run.

    """
    os.environ["JULIA_NUM_THREADS"] = str(num_processes_fit)

    with (
        open("julia-ace_out.log", "w", encoding="utf-8") as file_out,
        open("julia-ace_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call(["julia", script_name], stdout=file_out, stderr=file_err)


def run_gap(num_processes_fit: int, parameters) -> None:
    """
    GAP runner.

    num_processes_fit: int
        number of threads to be used for the run.

    Parameters
    ----------
        GAP fit parameters.

    """
    os.environ["OMP_NUM_THREADS"] = str(num_processes_fit)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # blas library
    os.environ["BLIS_NUM_THREADS"] = "1"  # blas library
    os.environ["MKL_NUM_THREADS"] = "1"  # blas library
    os.environ["NETLIB_NUM_THREADS"] = "1"  # blas library

    with (
        open("std_gap_out.log", "w", encoding="utf-8") as file_std,
        open("std_gap_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call(["gap_fit", *parameters], stdout=file_std, stderr=file_err)


def run_quip(
    num_processes_fit: int,
    data_path: str,
    xml_file: str,
    filename: str,
    glue_xml: bool = False,
) -> None:
    """
    QUIP runner.

    num_processes_fit: int
        number of threads to be used for the run.
    data_path:
        Path to the data file.
    filename: str
        Name of the output file.

    """
    os.environ["OMP_NUM_THREADS"] = str(num_processes_fit)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # blas library
    os.environ["BLIS_NUM_THREADS"] = "1"  # blas library
    os.environ["MKL_NUM_THREADS"] = "1"  # blas library
    os.environ["NETLIB_NUM_THREADS"] = "1"  # blas library

    init_args = "init_args='IP Glue'" if glue_xml else ""
    quip = (
        f"quip {init_args} E=T F=T atoms_filename={data_path} param_filename={xml_file}"
    )
    command = f"{quip} | grep AT | sed 's/AT//' > {filename}"
    with (
        open("std_quip_out.log", "w", encoding="utf-8") as file_std,
        open("std_quip_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call(command, stdout=file_std, stderr=file_err, shell=True)


def run_nep(gpu_identifier_indices: list[int]) -> None:
    """
    NEP runner.

    Parameters
    ----------
    gpu_identifier_indices: list[int]
        Indices that identifies the GPU that NEP should be run with
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_identifier_indices))

    with (
        open("std_nep_out.log", "w", encoding="utf-8") as file_out,
        open("std_nep_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call("nep", stdout=file_out, stderr=file_err, env=env)


def run_nequip(command: str, log_prefix: str) -> None:
    """
    Nequip runner.

    Parameters
    ----------
    command: str
        The command to execute, along with its arguments.
    log_prefix: str
        Prefix for log file names, used to differentiate between different commands' logs.

    """
    with (
        open(f"{log_prefix}_out.log", "w", encoding="utf-8") as file_out,
        open(f"{log_prefix}_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call(command.split(), stdout=file_out, stderr=file_err)


def run_mace(hypers: list) -> None:
    """
    MACE runner.

    Parameters
    ----------
    hypers: list
        containing all hyperparameters required for the MACE model training.

    """
    with (
        open("mace_train_out.log", "w", encoding="utf-8") as file_std,
        open("mace_train_err.log", "w", encoding="utf-8") as file_err,
    ):
        subprocess.call(["mace_run_train", *hypers], stdout=file_std, stderr=file_err)


def prepare_fit_environment(
    database_dir: Path,
    mlip_path: Path,
    glue_xml: bool,
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
    glue_name: str = "glue.xml",
) -> Path:
    """
    Prepare the environment for the fit.

    Parameters
    ----------
    database_dir: Path
        Path to database directory.
    mlip_path: Path
        Path to the MLIP fit run (cwd).
    glue_xml: bool
            use the glue.xml core potential instead of fitting 2b terms.
    train_name: str
        name of the training data file.
    test_name: str
        name of the test data file.
    glue_name: str
        name of the glue.xml file or path.

    Returns
    -------
    the MLIP file path.
    """
    os.makedirs(
        os.path.join(mlip_path, train_name.replace("train.extxyz", "")), exist_ok=True
    )
    if not Path(mlip_path / test_name).exists():
        shutil.copy(
            os.path.join(database_dir, test_name),
            os.path.join(mlip_path, test_name),
        )
    if not Path(mlip_path / train_name).exists():
        shutil.copy(
            os.path.join(database_dir, train_name),
            os.path.join(mlip_path, train_name),
        )
    if glue_xml:
        # TODO: might need to be fixed for remote connection
        shutil.copy(
            Path(glue_name),
            os.path.join(mlip_path, "glue.xml"),
        )

    return Path(os.path.join(mlip_path, train_name.replace("train.extxyz", "")))


def convert_xyz_to_structure(
    atoms_list: list,
    include_forces: bool = True,
    include_stresses: bool = True,
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
) -> tuple[list[Structure], list, list[object], list[object]]:
    """
    Convert extxyz to pymatgen Structure format.

    Parameters
    ----------
    atoms_list:
        list of atoms to be converted.
    include_forces: bool
        will include forces with the Structure object.
    include_stresses: bool
        will include stresses with the Structure object.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.

    Returns
    -------
    tuple(pymatgen Structure object, energies, forces, stresses)

    """
    structures = []
    energies = []
    forces = []
    stresses = []
    for atoms in atoms_list:
        structure = AseAtomsAdaptor.get_structure(atoms)
        structures.append(structure)
        energies.append(atoms.info[ref_energy_name])
        if include_forces:
            forces.append(np.array(atoms.arrays[ref_force_name]).tolist())
        else:
            forces.append(np.zeros((len(structure), 3)).tolist())
        if include_stresses:
            # convert from eV to GPa
            virial = atoms.info[ref_virial_name] / atoms.get_volume()  # eV/Å^3
            stresses.append(np.array(virial * 160.2176565).tolist())  # eV/Å^3 -> GPa
        else:
            stresses.append(np.zeros((3, 3)).tolist())

    logging.info(f"Loaded {len(structures)} structures.")

    return structures, energies, forces, stresses


def write_after_distillation_data_split(
    distillation: bool,
    force_max: float,
    split_ratio: float,
    vasp_ref_name: str = "vasp_ref.extxyz",
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
    force_label: str = "REF_forces",
    energy_label: str = "REF_energy",
) -> None:
    """
    Write train.extxyz and test.extxyz after data distillation and split.

    Reject structures with large force components and split dataset into training and test datasets.

    Parameters
    ----------
    distillation: bool
        For using data distillation.
    force_max: float
        Maximally allowed force in the data set.
    split_ratio: float
        Parameter to divide the training set and the test set.
        A value of 0.1 means that the ratio of the training set to the test set is 9:1
    vasp_ref_name:
        name of the VASP reference data file.
    train_name:
        name of the training data file.
    test_name:
        name of the test data file.
    force_label: str
        label of the force entries.
    energy_label: str
        label of the energy entries.
    """
    # reject structures with large force components
    atoms = (
        data_distillation(vasp_ref_name, force_max, force_label)
        if distillation
        else ase.io.read(vasp_ref_name, index=":")
    )

    # split dataset into training and test datasets
    (train_structures, test_structures) = stratified_dataset_split(
        atoms=atoms, split_ratio=split_ratio, energy_label=energy_label
    )

    ase.io.write(train_name, train_structures, format="extxyz", append=True)
    ase.io.write(test_name, test_structures, format="extxyz", append=True)


def mace_virial_format_conversion(
    atoms: list[Atoms], ref_virial_name: str, out_file_name: str
) -> None:
    """
    Convert the format of virial vector (9,) into a format (3x3) recognizable by MACE.

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        input structures
    ref_virial_name: str
        virial label
    out_file_name: str
        name of output file
    """
    formatted_atoms = []
    for at in atoms:
        if ref_virial_name in at.info:
            at.info[ref_virial_name] = at.info[ref_virial_name].reshape(3, 3)
            formatted_atoms.append(at)

    write(out_file_name, formatted_atoms, format="extxyz")
