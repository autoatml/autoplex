"""Utility functions for fitting jobs."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

from collections.abc import Iterable

import ase
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.atoms import Atoms
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.data import chemical_symbols
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from atomate2.utils.path import strip_hostname
from dgl.data.utils import split_dataset
from matgl.apps.pes import Potential
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from nequip.ase import NequIPCalculator
from numpy import ndarray
from pymatgen.io.ase import AseAtomsAdaptor
from pytorch_lightning.loggers import CSVLogger
from scipy.spatial import ConvexHull
from scipy.special import comb

from autoplex.data.common.utils import (
    data_distillation,
    plot_energy_forces,
    rms_dict,
    stratified_dataset_split,
)

current_dir = Path(__file__).absolute().parent
MLIP_PHONON_DEFAULTS_FILE_PATH = current_dir / "mlip-phonon-defaults.json"
MLIP_RSS_DEFAULTS_FILE_PATH = current_dir / "mlip-rss-defaults.json"


def gap_fitting(
    db_dir: Path,
    species_list: list | None = None,
    path_to_default_hyperparameters: Path | str = MLIP_PHONON_DEFAULTS_FILE_PATH,
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
    path_to_default_hyperparameters : str or Path.
        Path to gap-defaults.json.
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
    # keep additional pre- and suffixes
    gap_file_xml = train_name.replace("train", "gap_file").replace(".extxyz", ".xml")
    mlip_path: Path = prepare_fit_environment(
        db_dir, Path.cwd(), glue_xml, train_name, test_name, glue_file_path
    )

    db_atoms = ase.io.read(os.path.join(db_dir, train_name), index=":")
    train_data_path = os.path.join(db_dir, train_name)
    test_data_path = os.path.join(db_dir, test_name)

    default_hyperparameters = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=path_to_default_hyperparameters
    )

    gap_default_hyperparameters = default_hyperparameters["GAP"]

    gap_default_hyperparameters["general"].update({"gp_file": gap_file_xml})
    gap_default_hyperparameters["general"]["energy_parameter_name"] = ref_energy_name
    gap_default_hyperparameters["general"]["force_parameter_name"] = ref_force_name
    gap_default_hyperparameters["general"]["virial_parameter_name"] = ref_virial_name

    for parameter in gap_default_hyperparameters:
        if fit_kwargs:
            for arg in fit_kwargs:
                if parameter == arg:
                    gap_default_hyperparameters[parameter].update(fit_kwargs[arg])

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
        run_quip(num_processes_fit, train_data_path, gap_file_xml, "quip_" + train_name)

    if include_three_body:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_3b = energy_remain("quip_" + train_name)
            delta_3b = delta_3b / num_triplet
            gap_default_hyperparameters["threeb"].update({"delta": delta_3b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
        )

        run_gap(num_processes_fit, fit_parameters_list)
        run_quip(num_processes_fit, train_data_path, gap_file_xml, "quip_" + train_name)

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
            "quip_" + train_name,
            glue_xml,
        )

    if include_soap:
        delta_soap = (
            energy_remain("quip_" + train_name)
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
            "quip_" + train_name,
            glue_xml,
        )

    # Calculate training error
    train_error = energy_remain("quip_" + train_name)
    print("Training error of MLIP (eV/at.):", round(train_error, 7))

    # Calculate testing error
    run_quip(
        num_processes_fit, test_data_path, gap_file_xml, "quip_" + test_name, glue_xml
    )
    test_error = energy_remain("quip_" + test_name)
    print("Testing error of MLIP (eV/at.):", round(test_error, 7))

    if not glue_xml and species_list:
        plot_energy_forces(
            title="Data error metrics",
            energy_limit=0.005,
            force_limit=0.1,
            species_list=species_list,
            train_name=train_name,
            test_name=test_name,
        )

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": mlip_path,
        "mlip_pot": mlip_path.joinpath(gap_file_xml),
    }


def jace_fitting(
    db_dir: str | Path,
    path_to_default_hyperparameters: Path | str = MLIP_RSS_DEFAULTS_FILE_PATH,
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
    path_to_default_hyperparameters : str or Path.
        Path to mlip-rss-defaults.json.
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

    default_hyperparameters = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=path_to_default_hyperparameters
    )
    jace_hypers = default_hyperparameters["J-ACE"]

    if fit_kwargs:
        for parameter in jace_hypers:
            if parameter in fit_kwargs:
                if isinstance(fit_kwargs[parameter], type(jace_hypers[parameter])):
                    jace_hypers[parameter] = fit_kwargs[parameter]
                else:
                    raise TypeError(
                        f"The type of {parameter} should be {type(jace_hypers[parameter])}!"
                    )

    order = jace_hypers["order"]
    totaldegree = jace_hypers["totaldegree"]
    cutoff = jace_hypers["cutoff"]
    solver = jace_hypers["solver"]

    ace_text = f"""using ACEpotentials
using LinearAlgebra: norm, Diagonal
using CSV, DataFrames
using Distributed
addprocs({num_processes_fit-1}, exeflags="--project=$(Base.active_project())")
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


def nequip_fitting(
    db_dir: Path,
    path_to_default_hyperparameters: Path | str = MLIP_RSS_DEFAULTS_FILE_PATH,
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
    path_to_default_hyperparameters : str or Path.
        Path to mlip-rss-defaults.json.
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

    default_hyperparameters = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=path_to_default_hyperparameters
    )

    nequip_hypers = default_hyperparameters["NEQUIP"]

    if fit_kwargs:
        for parameter in nequip_hypers:
            if parameter in fit_kwargs:
                if isinstance(fit_kwargs[parameter], type(nequip_hypers[parameter])):
                    nequip_hypers[parameter] = fit_kwargs[parameter]
                else:
                    raise TypeError(
                        f"The type of {parameter} should be {type(nequip_hypers[parameter])}!"
                    )

    r_max = nequip_hypers["r_max"]
    num_layers = nequip_hypers["num_layers"]
    l_max = nequip_hypers["l_max"]
    num_features = nequip_hypers["num_features"]
    num_basis = nequip_hypers["num_basis"]
    invariant_layers = nequip_hypers["invariant_layers"]
    invariant_neurons = nequip_hypers["invariant_neurons"]
    batch_size = nequip_hypers["batch_size"]
    learning_rate = nequip_hypers["learning_rate"]
    max_epochs = nequip_hypers["max_epochs"]
    default_dtype = nequip_hypers["default_dtype"]

    nequip_text = f"""root: results
run_name: autoplex
seed: 123
dataset_seed: 456
append: true
default_dtype: {default_dtype}

# network
r_max: {r_max}
num_layers: {num_layers}
l_max: {l_max}
parity: true
num_features: {num_features}
nonlinearity_type: gate

nonlinearity_scalars:
  e: silu
  o: tanh

nonlinearity_gates:
  e: silu
  o: tanh

num_basis: {num_basis}
BesselBasis_trainable: true
PolynomialCutoff_p: 6

invariant_layers: {invariant_layers}
invariant_neurons: {invariant_neurons}
avg_num_neighbors: auto

use_sc: true
dataset: ase
validation_dataset: ase
dataset_file_name: ./train_nequip.extxyz
validation_dataset_file_name: {db_dir}/test.extxyz

ase_args:
  format: extxyz
dataset_key_mapping:
  {ref_energy_name}: total_energy
  {ref_force_name}: forces
validation_dataset_key_mapping:
  {ref_energy_name}: total_energy
  {ref_force_name}: forces

chemical_symbols:
{isolated_atom_energies_update}
wandb: False

verbose: info
log_batch_freq: 10
log_epoch_freq: 1
save_checkpoint_freq: -1
save_ema_checkpoint_freq: -1

n_train: {num_of_train}
n_val: {num_of_val}
learning_rate: {learning_rate}
batch_size: {batch_size}
validation_batch_size: 10
max_epochs: {max_epochs}
shuffle: true
metrics_key: validation_loss
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true
report_init_validation: true

early_stopping_patiences:
  validation_loss: 50

early_stopping_lower_bounds:
  LR: 1.0e-5

loss_coeffs:
  forces: 1
  total_energy:
    - 1
    - PerAtomMSELoss

metrics_components:
  - - forces
    - mae
  - - forces
    - rmse
  - - forces
    - mae
    - PerSpecies: True
      report_per_component: False
  - - forces
    - rmse
    - PerSpecies: True
      report_per_component: False
  - - total_energy
    - mae
  - - total_energy
    - mae
    - PerAtom: True

optimizer_name: Adam
optimizer_amsgrad: true

lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 100
lr_scheduler_factor: 0.5

per_species_rescale_shifts_trainable: false
per_species_rescale_scales_trainable: false

per_species_rescale_shifts: dataset_per_atom_total_energy_mean
per_species_rescale_scales: dataset_forces_rms
    """

    with open("nequip.yaml", "w") as file:
        file.write(nequip_text)

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
    path_to_default_hyperparameters: Path | str = MLIP_RSS_DEFAULTS_FILE_PATH,
    device: str = "cuda",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Perform the M3GNet potential fitting.

    Parameters
    ----------
    db_dir: Path
        Directory containing the training and testing data files.
    path_to_default_hyperparameters : str or Path.
        Path to mlip-rss-defaults.json.
    device: str
        Device on which the model will be trained, e.g., 'cuda' or 'cpu'.
    ref_energy_name : str, optional
        Reference energy name.
    ref_force_name : str, optional
        Reference force name.
    ref_virial_name : str, optional
        Reference virial name.
    fit_kwargs: dict.
        optional dictionary with parameters for m3gnet fitting with keys same as
        mlip-rss-defaults.json.

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
    hidden_dim: int
        Dimensionality of the hidden layers in the model.
    num_units: int
        Number of units in each dense layer of the model.
    max_l: int
        Maximum degree of spherical harmonics.
    max_n: int
        Maximum radial function degree.
    test_equal_to_val: bool
        If True, the testing dataset will be the same as the validation dataset.

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
    default_hyperparameters = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=path_to_default_hyperparameters
    )

    m3gnet_hypers = default_hyperparameters["M3GNET"]

    if fit_kwargs:
        for parameter in m3gnet_hypers:
            if parameter in fit_kwargs:
                if isinstance(fit_kwargs[parameter], type(m3gnet_hypers[parameter])):
                    m3gnet_hypers[parameter] = fit_kwargs[parameter]
                else:
                    raise TypeError(
                        f"The type of {parameter} should be {type(m3gnet_hypers[parameter])}!"
                    )

    exp_name = m3gnet_hypers["exp_name"]
    results_dir = m3gnet_hypers["results_dir"]
    cutoff = m3gnet_hypers["cutoff"]
    threebody_cutoff = m3gnet_hypers["threebody_cutoff"]
    batch_size = m3gnet_hypers["batch_size"]
    max_epochs = m3gnet_hypers["max_epochs"]
    include_stresses = m3gnet_hypers["include_stresses"]
    hidden_dim = m3gnet_hypers["hidden_dim"]
    num_units = m3gnet_hypers["num_units"]
    max_l = m3gnet_hypers["max_l"]
    max_n = m3gnet_hypers["max_n"]
    test_equal_to_val = m3gnet_hypers["test_equal_to_val"]

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
            include_stresses=include_stresses,
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

        print(train_element_types)
        train_converter = Structure2Graph(
            element_types=train_element_types, cutoff=cutoff
        )
        train_datasets = MGLDataset(
            threebody_cutoff=threebody_cutoff,
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
                include_stresses=include_stresses,
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
                element_types=test_element_types, cutoff=cutoff
            )
            test_dataset = MGLDataset(
                threebody_cutoff=threebody_cutoff,
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
            batch_size=batch_size,
            num_workers=1,
        )
        model = M3GNet(
            element_types=train_element_types,
            is_intensive=False,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            dim_node_embedding=hidden_dim,
            dim_edge_embedding=hidden_dim,
            units=num_units,
            max_l=max_l,
            max_n=max_n,
        )
        lit_module = PotentialLightningModule(model=model, include_line_graph=True)
        logger = CSVLogger(name=exp_name, save_dir=os.path.join(results_dir, "logs"))
        # Inference mode = False is required for calculating forces, stress in test mode and prediction mode
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                torch.cuda.set_device(torch.device(f"cuda:{gpu_id}"))
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator="gpu",
                    logger=logger,
                    inference_mode=False,
                )
            else:
                raise ValueError("CUDA is not available.")
        else:
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="cpu",
                logger=logger,
                inference_mode=False,
            )
        # Again loggers ...
        print("Start training...")
        print("Length of train_loader: ", len(train_loader))
        print("Length of val_loader: ", len(val_loader))
        print("Length of test_loader: ", len(test_loader))
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
    path_to_default_hyperparameters: Path | str = MLIP_RSS_DEFAULTS_FILE_PATH,
    device: str = "cuda",
    ref_energy_name: str = "REF_energy",
    ref_force_name: str = "REF_forces",
    ref_virial_name: str = "REF_virial",
    use_defaults=True,
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
    path_to_default_hyperparameters : str or Path.
        Path to mlip-rss-defaults.json.
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
    if ref_virial_name is not None:
        atoms = read(f"{db_dir}/train.extxyz", index=":")
        mace_virial_format_conversion(
            atoms=atoms, ref_virial_name=ref_virial_name, out_file_name="train.extxyz"
        )

    if use_defaults:
        default_hyperparameters = load_mlip_hyperparameter_defaults(
            mlip_fit_parameter_file_path=path_to_default_hyperparameters
        )

        mace_hypers = default_hyperparameters["MACE"]
    else:
        mace_hypers = {}

    # TODO: should we do a type check? not sure
    #  as it will be a lot of work to keep it updated
    mace_hypers.update(fit_kwargs)

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
            print("Train and test files have default names.")
        elif hyper in ["energy_key", "virial_key", "forces_key", "device"]:
            print("energy_key, virial_key and forces_key have default names.")
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
        with open("./logs/MACE_final_run-3.log") as file:
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


def load_mlip_hyperparameter_defaults(mlip_fit_parameter_file_path: str | Path) -> dict:
    """
    Load gap fit default parameters from the json file.

    Parameters
    ----------
    mlip_fit_parameter_file_path : str or Path.
        Path to MLIP default parameter JSON files.

    Returns
    -------
    dict
       gap fit default parameters.
    """
    with open(mlip_fit_parameter_file_path, encoding="utf-8") as f:
        return json.load(f)


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
                if output_type == "rand_struc_dir":
                    flat_dirs = [[item for sublist in dirs for item in sublist]]
                    list_of_vasp_calc_dirs.extend(*flat_dirs)
                else:
                    list_of_vasp_calc_dirs.extend(*dirs)

    return list_of_vasp_calc_dirs


def vaspoutput_2_extended_xyz(
    path_to_vasp_static_calcs: list,
    config_types: list[str] | None = None,
    data_types: list[str] | None = None,
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
        try:
            # read the vasp output
            file = read(path_without_hostname, index=":")
            for i in file:
                virial_list = (
                    -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
                )
                i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
                del i.calc.results["stress"]
                i.arrays["REF_forces"] = i.calc.results["forces"]
                if atom_wise_regularization and (data_type == "phonon_dir"):
                    atom_forces = np.array(i.arrays["REF_forces"])
                    atom_wise_force = np.array(
                        [
                            force if force > f_min else f_min
                            for force in np.linalg.norm(atom_forces, axis=1)
                        ]
                    )
                    i.arrays["force_atom_sigma"] = regularization * atom_wise_force
                del i.calc.results["forces"]
                i.info["REF_energy"] = i.calc.results["free_energy"]
                del i.calc.results["energy"]
                del i.calc.results["free_energy"]
                i.info["config_type"] = config_type
                i.info["data_type"] = data_type.rstrip("_dir")
                i.pbc = True
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
    shutil.copy(
        os.path.join(database_dir, test_name),
        os.path.join(mlip_path, test_name),
    )
    shutil.copy(
        os.path.join(database_dir, train_name),
        os.path.join(mlip_path, train_name),
    )
    if glue_xml:
        shutil.copy(
            os.path.join(database_dir, glue_name),
            os.path.join(mlip_path, "glue.xml"),
        )

    return mlip_path


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

    print(f"Loaded {len(structures)} structures.")

    return structures, energies, forces, stresses


def write_after_distillation_data_split(
    distillation: bool,
    force_max: float,
    split_ratio: float,
    vasp_ref_name: str = "vasp_ref.extxyz",
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
    force_label: str = "REF_forces",
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
    """
    # reject structures with large force components
    atoms = (
        data_distillation(vasp_ref_name, force_max, force_label)
        if distillation
        else ase.io.read(vasp_ref_name, index=":")
    )

    # split dataset into training and test datasets
    (train_structures, test_structures) = stratified_dataset_split(atoms, split_ratio)

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
