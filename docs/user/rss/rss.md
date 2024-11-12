(rss)=

*Written by Yuanbin Liu. For inquiries, please contact [lyb122502@126.com](mailto:lyb122502@126.com).*

# Random structure searching (RSS) workflow

The random structure searching (RSS) approach was initially proposed for predicting crystal structures by generating randomised, sensible structures and optimising them via first-principles calculations ([Phys. Rev. Lett. 97, 045504 (2006)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.045504) and [J. Phys.: Condens. Matter 23, 053201 (2011)](https://iopscience.iop.org/article/10.1088/0953-8984/23/5/053201)). Recently, RSS was expanded into a methodology for exploring and learning potential-energy surfaces from scratch ([Phys. Rev. Lett. 120, 156001 (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.156001) and [npj Comput. Mater. 5, 99 (2019)](https://www.nature.com/articles/s41524-019-0236-6)). Enhanced with physics-inspired sampling methods, such as Boltzmann-probability biased histograms and CUR, this approach ensures both the significance (low-energy) and diversity of the structures being searched.

## Quick start workflow for immediate use

In `autoplex`, you can only use the `RssMaker` module to build a complete ML-RSS potential model from scratch. `RssMaker` accepts customizable parameters either through a YAML configuration file or by directly specifying arguments in the `make` method. These [parameters](#rssmaker-parameters) control various aspects of the RSS workflow, including randomized structure generation, sampling methods, DFT labeling, data preprocessing (e.g., regularization), potential fitting hyperparameters, and iterative settings for RSS. The minimal input required is simply specifying the chemical formula of the target system by the argument `tag`.

```python
from autoplex.auto.rss.flows import RssMaker
from fireworks import LaunchPad
from jobflow import Flow
from jobflow.managers.fireworks import flow_to_workflow

rss_job = RssMaker().make(tag='Si')
wf = flow_to_workflow(rss_job) 
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
```

The above code is based on `FireWorks`](https://materialsproject.github.io/fireworks/) for job submission and management. You could also use [`jobflow-remote`](https://matgenix.github.io/jobflow-remote/), in which case the code snippet would change as follows.

```python
from jobflow_remote import submit_flow

rss_job = RssMaker().make(tag='Si')
resources = {"nodes": N, "partition": "name", "qos": "name", "time": "8:00:00", "mail_user": "your_email", "mail_type": "ALL", "account": "your account"}
print(submit_flow(rss_job, worker="your worker", resources=resources, project="your project name"))
```

Congratulations! After a few iterations (10 by default), you have obtained your first ML potential for Si with `autoplex`. In our previous tests, we found that this type of potential (ML-RSS) is useful to describe crystalline and disordered phases.

## Customizing RSS workflow

By adjusting the parameters in `RssMaker`, you can easily create an RSS workflow tailored to your specific research needs.

### Launching the workflow using a YAML configuration file

The RSS workflow can be launched using a custom YAML configuration file. A complete list of parameters, including all default settings and modifiable options, is available in `autoplex/auto/rss/rss_default_configuration.yaml`. When you create a new YAML configuration file, any specified keys will override the corresponding default values. To initiate a new workflow, simply provide the path to your YAML file in the `config_file` argument within the `make` method.

```python
rss_job = RssMaker(name="your workflow name").make(config_file='path/to/your/name.yaml')
```

### Launching the workflow by directly specifying arguments

Alternatively, the RSS workflow can be launched by directly specifying parameters within the `make` method, without the need for a YAML configuration file. This approach allows you to override default settings by passing them as keyword arguments, providing flexibility when only a few parameters need to be customized.

```python
rss_job = RssMaker(name="your workflow name").make(tag='SiO2',
                                                   generated_struct_numbers=[10000],
                                                   bcur_params={'soap_paras': {'l_max'=10},
                                                                'kernel_exp': 2,
                                                                'kt': 0.3},
                                                   )
```

### Building RSS models with various ML potentials

Currently, `RssMaker` supports GAP in addition to three graph-based network models, including NequIP, M3GNet, and MACE. You can specify the desired model using the `mlip_type` argument and adjust relevant hyperparameters within the `make` method. Default values for these hyperparameters are available in `autoplex/autoplex/fitting/common/mlip-rss-defaults.json`.

```python
rss_job = RssMaker(name="your workflow name").make(tag='SiO2',
                                                   mlip_type='MACE',
                                                   hidden_irreps="128x0e + 128x1o",
                                                   r_max=5.0,               
                                                   )
```

> ℹ️ Please note that we primarily recommend the GAP-RSS model for now, as GAP has shown great stability with small datasets. Other models have not been explored so far. However, we also encourage users to explore and test other individual models or combinations for interesting results.

### Resuming workflow from point of interruption

To resume an interrupted RSS workflow, use the `resume_from_previous_state` argument, which accepts a dictionary containing the necessary state information. Additionally, set `train_from_scratch` to `False` to enable resuming from a previous state.

```python
rss_job = RssMaker(name="your workflow name").make(tag='SiO2',
                                                   train_from_scratch=False,
                                                   resume_from_previous_state={'test_error': 0.24,
                                                   'pre_database_dir': 'path/to/pre-existing/database',
                                                   'mlip_path': 'path/to/previous/MLIP-model',
                                                   'isolated_atom_energies': {8: -0.16613333, 14: -0.16438578},
                                                   },
                                                   )
```

## Examples

This section provides several examples of parameter configurations in `RssMaker` for different systems and demonstrates some resulting predictions.

### Elemental system: Si

The core principle that enables the RSS method to work effectively is the diversity of the initial guessing structures. In our implementation, we support exploring different initial parameters for `buildcell`, maximizing structural diversity in the search process. In this example, we provide two sets of initial random structure parameters through `buildcell_options`. There are 80% of the generated structures for cells with an even number of atoms, while the remaining 20% are for cells with an odd number of atoms. This setup allows for greater flexibility in exploring the configuration space, ensuring a broad sampling across various structural possibilities.

```python
rss_job = RssMaker(name="your workflow name").make(tag='Si',
                                                   generated_struct_numbers=[8000, 2000],
                                                   buildcell_options=[{'NATOM': '{6,8,10,12,14,16,18,20,22,24}',
                                                                       'NFORM': '1',},
                                                                      {'NATOM': '{7,9,11,13,15,17,19,21,23}',
                                                                        'NFORM': '1',}
                                                                      ],
                                                   num_of_initial_selected_structs=[80, 20],
                                                   scalar_pressure_method ='exp',
                                                   scalar_exp_pressure=1,
                                                   scalar_pressure_exponential_width=0.2,
                                                   max_iteration_number=25,
                                                   )
```

### Binary system with fixed stoichiometric ratio: SiO2

## RssMaker parameters
