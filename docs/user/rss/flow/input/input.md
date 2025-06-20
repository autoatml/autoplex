(input)=

# Input YAML configuration file

The YAML configuration file contains all the inputs you need to define and customize your RSS workflow. This section provides a detailed explanation of how to set up the file, using silicon (`Si`) as an example. To make it easier to understand, we will break down the YAML file into several sections based on the key components of the RSS workflow and explain each part in detail.

## General parameters

The section defines the general settings for the RSS workflow, including the system's formula, whether to start fresh, or resume from a previous state.

```yaml
# General Parameters
tag: 'Si'
train_from_scratch: false
resume_from_previous_state:
  test_error:
  pre_database_dir:
  mlip_path:
  isolated_atom_energies:
```

The `tag` identifies the elements in the system, such as `Si` in this example, and can also be passed to the parameters for random structure generation. Setting `train_from_scratch=True` indicates that the RSS workflow will start from scratch. To resume a workflow, refer to the [Resuming workflow from point of interruption](../quick_start/start.md) section.

## Buildcell parameters

The section defines the settings for generating initial random structures. These parameters control the diversity, symmetry, size, and number of the generated structures. We utilize the `buildcell` software from [AIRSS](https://airss-docs.github.io/technical-reference/buildcell-manual) for random structure generation, ensuring flexibility and efficiency in exploring configurational space.

```yaml
# Buildcell Parameters
generated_struct_numbers:
  - 8000
  - 2000
cell_seed_paths:
buildcell_options:
  - NFORM: '1'
    SYMMOPS: '1-4'
    SLACK: 0.25
    OVERLAP: 0.1
    NATOM: '{6,8,10,12,14,16,18,20,22,24}'
  - NFORM: '1'
    SYMMOPS: '1-4'
    SLACK: 0.25
    OVERLAP: 0.1
    NATOM: '{7,9,11,13,15,17,19,21,23}'
fragment_file: null
fragment_numbers: null
num_processes_buildcell: 128
```

The `buildcell_options` parameter is one of the most critical settings, 
as it determines the scope of the RSS search and directly influences the diversity of the generated structures. 
We provide multiple sets of `buildcell` input parameters. 
Using the configuration file above, a total of 8000 structures will be generated, each containing an even number of atoms. 
Additionally, 2000 structures will be created with an odd number of atoms. The symmetry operations will vary from 1 to 4. 
You can supply a single set or multiple sets as needed. 
This flexibility ensures that the initial structures have sufficient diversity. 
In principle, any parameter supported by [buildcell](https://airss-docs.github.io/technical-reference/buildcell-manual) can be used in the `buildcell_options` section. In addition to `buildcell_options`, one can also load buildcell parameters from standard `.cell` files (as described in the [buildcell](https://airss-docs.github.io/technical-reference/buildcell-manual) through `cell_seed_paths`. This allows for greater flexibility, especially for defining interface structures and solution systems, as users can rely on well-established input formats.

The `fragment_file` and `fragment_numbers` parameters are used during random structure generation to define specific fragments as the smallest building blocks. For example, you can define an H<sub>2</sub>O molecule as a fragment and use it as the basic unit for generating random structures. This allows for more customized and realistic initial configurations when working with molecular or other complex systems. The `num_processes_buildcell` parameter specifies the number of CPU cores to be used in parallel during random structure generation. Note that this parameter is limited to a single node.

> **Note**: The `generated_struct_numbers` and `buildcell_options` parameters must have the same length. Each entry in `buildcell_options` corresponds to the number of structures specified at the same position in `generated_struct_numbers`. If both `cell_seed_paths` and `buildcell_options` are set, only `cell_seed_paths` will take effect.

## Sampling parameters

The section controls the selection methods of structures during the RSS workflow. These parameters determine how initial and iteratively generated structures are selected.

```yaml
# Sampling Parameters
num_of_initial_selected_structs:
  - 80
  - 20
num_of_rss_selected_structs: 100
initial_selection_enabled: true
rss_selection_method: 'bcur2i'
bcur_params:
  soap_paras:
    l_max: 12
    n_max: 12
    atom_sigma: 0.0875
    cutoff: 10.5
    cutoff_transition_width: 1.0
    zeta: 4.0
    average: true
    species: true
  frac_of_bcur: 0.8
  bolt_max_num: 3000
random_seed: null
```

If the RSS workflow starts exploration from scratch, it consists of two stages. In the first stage, there is no pre-existing potential, so structures are directly selected from the initial random structures for fitting. The `num_of_initial_selected_structs` parameter defines how many structures are selected from the initial random structures. In the example provided, the workflow selects 80 structures from 8000 even-numbered cells and 20 structures from 2000 odd-numbered cells for fitting the initial potential. In the second stage, after obtaining the initial potential, the workflow transitions to ML-driven RSS iterations. During this stage, the `num_of_rss_selected_structs` parameter specifies the number of structures sampled in each RSS iteration.

The `initial_selection_enabled` parameter enables initial structure selection when set to `true`. In this case, the sampling method will be CUR. The `rss_selection_method` defines the strategy for selecting structures during the RSS iteration. In this example, the method `bcur2i` is used, which combines Boltzmann-weighted energy histograms and CUR sampling to select low-energy and diverse structures. The `bcur_params` section provides detailed settings for the `bcur2i` method. The `soap_paras` define the SOAP descriptors used for structure representation.

## DFT labelling parameters

The section allows you to set up VASP static calculations for accurately labeling training structures, including bulk, isolated atoms, and dimers.

```yaml
# DFT Labelling Parameters
include_isolated_atom: true
isolatedatom_box:
  - 20.0
  - 20.0
  - 20.0
e0_spin: false
include_dimer: true
dimer_box:
  - 20.0
  - 20.0
  - 20.0
dimer_range:
  - 1.0
  - 5.0
dimer_num: 21
custom_incar:
  NCORE: 16
  ISMEAR: 0
  SIGMA: 0.05
  PREC: "Accurate"
  ADDGRID: ".TRUE."
  EDIFF: 1E-7
  NELM: 250
  LWAVE: ".FALSE."
  LCHARG: ".FALSE."
  ALGO: "normal"
  AMIX: 0.1
  LREAL: ".FALSE."
  ISYM: 0
  ENCUT: 400.0
  KSPACING: 0.25
  GGA: 'PS'
custom_potcar:
vasp_ref_file: 'vasp_ref.extxyz'
```

If `include_isolated_atom` and `include_dimer` are set to `true`, the program will automatically identify all elements present in the structures generated by `buildcell` and set up cells for isolated atoms and dimers accordingly based on the recognized elements. The `custom_incar` parameter allows you to define any VASP settings. You can adjust these settings by adding or removing keys as needed. 

## Data preprocessing parameters

The section defines how the training data is prepared before fitting potentials. This includes filtering, regularization, combining external datasets, and splitting the data into training and testing sets.

```yaml
# Data Preprocessing Parameters
config_types:
  - 'initial'
  - 'traj_early'
  - 'traj'
rss_group:
  - 'traj'
test_ratio: 0.1
regularization: true
scheme: 'linear-hull'
retain_existing_sigma: true
reg_minmax:
  - [0.1, 1]
  - [0.001, 0.1]
  - [0.0316, 0.316]
  - [0.0632, 0.632]
distillation: false
force_max: null
force_label: null
pre_database_dir: null
```

Regularization is currently only applicable to GAP potentials and is adjusted using the `scheme` parameter. Common schemes include `'linear-hull'` and `'volume-stoichiometry'`. For systems with fixed stoichiometry, `'linear-hull'` is recommended. For systems with varying stoichiometries, `'volume-stoichiometry'` is more appropriate.

## MLIP parameters

The section defines the settings for training machine learning potentials. Currently supported architectures include GAP, ACE(Julia), NequIP, M3GNet, and MACE. 
You can specify the desired model using the `mlip_type` argument and tune hyperparameters flexibly by adding key-value pairs. 

```yaml
# MLIP Parameters
mlip_type: 'GAP'
auto_delta: True
mlip_hypers:
  GAP:
    general:
      at_file: train.extxyz
      default_sigma: '{0.0001 0.05 0.05 0}'
      energy_parameter_name: REF_energy
      force_parameter_name: REF_forces
      virial_parameter_name: REF_virial
      sparse_jitter: 1e-08
      do_copy_at_file: F
      openmp_chunk_size: 10000
      gp_file: gap_file.xml
      e0_offset: 0.0
      two_body: true
      three_body: false
      soap: true
    twob:
      distance_Nb_order: 2
      f0: 0.0
      add_species: T
      cutoff: 5.0
      n_sparse: 15
      covariance_type: ard_se
      delta: 2.0
      theta_uniform: 0.5
      sparse_method: uniform
      compact_clusters: T
    threeb:
      distance_Nb_order: 3
      f0: 0.0
      add_species: T
      cutoff: 3.25
      n_sparse: 100
      covariance_type: ard_se
      delta: 2.0
      theta_uniform: 1.0
      sparse_method: uniform
      compact_clusters: T
    soap:
      add_species: T
      l_max: 10
      n_max: 12
      atom_sigma: 0.5
      zeta: 4
      cutoff: 5.0
      cutoff_transition_width: 1.0
      central_weight: 1.0
      n_sparse: 6000
      delta: 1.0
      f0: 0.0
      covariance_type: dot_product
      sparse_method: cur_points
```

## RSS Exploration Parameters

The section sets up the RSS iterative process, including parameters for structure optimization and convergence criteria.

```yaml
# RSS Exploration Parameters
scalar_pressure_method: 'exp'
scalar_exp_pressure: 1
scalar_pressure_exponential_width: 0.2
scalar_pressure_low: 0
scalar_pressure_high: 25
max_steps: 200
force_tol: 0.01
stress_tol: 0.01
stop_criterion: 0.001
max_iteration_number: 25
num_groups: 6
initial_kt: 0.3
current_iter_index: 1
hookean_repul: true
hookean_paras:
  '(14, 14)': [1000, 1.0]
keep_symmetry: true
remove_traj_files: true
num_processes_rss: 128
device_for_rss: 'cpu'
```

The RSS workflow supports searching for structures under high pressure, controlled by the `scalar_pressure_method` parameter. Two methods are available: `exp`, where pressure is sampled based on an exponential distribution with control parameters `scalar_exp_pressure` and `scalar_pressure_exponential_width`, and `uniform`, where pressure is sampled uniformly within a range defined by `scalar_pressure_low` and `scalar_pressure_high`. One can use `remove_traj_files` to delete all RSS trajectories in order to save storage space.

To terminate the iterative process, two stopping criteria are provided: `stop_criterion` and `max_iteration_number`. The iterations stop when the prediction error falls below the value of `stop_criterion`. Or the iterations stop when the number of iterations exceeds the limit defined by `max_iteration_number`. The workflow will stop when either of the above criteria is satisfied.

We strongly recommend enabling `hookean_repul`, as it applies a strong repulsive force when the distance between two atoms falls below a certain threshold. This ensures that the generated structures are physically reasonable.

The GAP-RSS model of Si was iterated 25 times on a server cluster with 7 nodes, each equipped with 128 cores, taking approximately 1 day to complete. The resulting potential was found to accurately describe different crystalline polymorphs as well as disordered phases.
