(rss)=

# Examples

This section demonstrates various YAML configurations beyond Si and explores use cases for the RSS workflow.

##  SiO2

This section provides guidance on exploring silica models at different functional levels.

```yaml
# General Parameters
tag: 'SiO2'
train_from_scratch: true
resume_from_previous_state:
  test_error: 
  pre_database_dir: 
  mlip_path: 
  isolated_atom_energies: 

# Buildcell Parameters
generated_struct_numbers:
  - 8000
  - 2000
buildcell_options:
  - SPECIES: "Si%NUM=1,O%NUM=2"
    NFORM: '{2,4,6,8}'
    SYMMOPS: '1-4'
    MINSEP: '1.5 Si-Si=2.7-3.0 Si-O=1.3-1.6 O-O=2.28-2.58'
  - SPECIES: "Si%NUM=1,O%NUM=2"
    NFORM: '{3,5,7}'
    SYMMOPS: '1-4'
    MINSEP: '1.5 Si-Si=2.7-3.0 Si-O=1.3-1.6 O-O=2.28-2.58'
fragment_file: null
fragment_numbers: null
num_processes_buildcell: 128

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

# DFT Labelling Parameters
include_isolated_atom: true
isolatedatom_box:
  - 20.0
  - 20.0
  - 20.0
e0_spin: false
include_dimer: false
dimer_box:
  - 20.0
  - 20.0
  - 20.0
dimer_range:
  - 1.0
  - 5.0
dimer_num: 41
custom_incar:
  KPAR: 8
  NCORE: 16
  LSCALAPACK: ".FALSE."
  LPLANE: ".FALSE."
  ISMEAR: 0
  SIGMA: 0.1
  PREC: "Accurate"
  ADDGRID: ".TRUE."
  EDIFF: 1E-7
  NELM: 250
  LWAVE: ".FALSE."
  LCHARG: ".FALSE."
  ALGO: "normal"
  AMIX: null
  LREAL: ".FALSE."
  ISYM: 0
  ENCUT: 900.0
  KSPACING: 0.23
  GGA: null
  AMIX_MAG: null
  BMIX: null
  BMIX_MAG: null
  ISTART: null
  LMIXTAU: null
  NBANDS: null
  NELMDL: null
  METAGGA: "SCAN"
  LASPH: ".TRUE."
custom_potcar:
vasp_ref_file: 'vasp_ref.extxyz'

# Data Preprocessing Parameters
config_types:
  - 'initial'
  - 'traj_early'
  - 'traj'
rss_group:
  - 'traj'
test_ratio: 0.0
regularization: true
scheme: 'linear-hull'
reg_minmax:
  - [0.1, 1]
  - [0.001, 0.1]
  - [0.0316, 0.316]
  - [0.0632, 0.632]
distillation: false
force_max: null
force_label: null
pre_database_dir: null

# MLIP Parameters
mlip_type: 'GAP'
ref_energy_name: 'REF_energy'
ref_force_name: 'REF_forces'
ref_virial_name: 'REF_virial'
auto_delta: true
num_processes_fit: 32
device_for_fitting: 'cpu'
twob:
  cutoff: 5.0
  n_sparse: 15
  theta_uniform: 1.0
  delta: 1.0
threeb:
  cutoff: 3.25
soap:
  l_max: 6
  n_max: 12
  atom_sigma: 0.5
  n_sparse: 3000
  cutoff: 5.0
  delta: 0.2
general:
  three_body: false

# RSS Exploration Parameters
scalar_pressure_method: 'exp'
scalar_exp_pressure: 100
scalar_pressure_exponential_width: 0.2
scalar_pressure_low: 0
scalar_pressure_high: 25
max_steps: 300
force_tol: 0.01
stress_tol: 0.01
stop_criterion: 0.001
max_iteration_number: 10
num_groups: 16
initial_kt: 0.3
current_iter_index: 1
hookean_repul: true
hookean_paras:
  '(1, 1)': [1000, 0.6]
  '(8, 1)': [1000, 0.4]
  '(8, 8)': [1000, 1.0]
keep_symmetry: false
write_traj: true
num_processes_rss: 128
device_for_rss: 'cpu'
```

To switch from SCAN to PBE, simply remove the `METAGGA` and `LASPH` entries from the `custom_incar` settings. All other parameters remain unchanged.
