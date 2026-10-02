# Core Workflow

This tutorial covers the core workflow in automating NMR predictions via CASTEP, with a section on how it can be extended to automated ML training in a wider autoplex workflow (such as the {ref}`RSS workflow <rss>`). 
## Overview

Given a list of [pymatgen](https://pymatgen.org/) structures (usually given in a `.xyz` file) and user-defined parameters, CASTEP is called with `task: magres` (see the [CASTEP setup](../../../rss/flow/input/input.md#labelling-parameters) for how autoplex runs CASTEP) and returns `.magres` and `.castep` files, as well as the magnetic shielding (MS) and electric field gradient (EFG) tensors for each atom in each structure. 

## Example workflow

First we define our `.param` and `.cell` parameters using {class}`~autoplex.misc.castep.utils.CastepMagresSetGenerator`:
```python
from autoplex.misc.castep.utils import CastepMagresSetGenerator

input_set_generator = CastepMagresSetGenerator(
    use_efg=True,
    user_param_settings={"xc_functional": "PBE", "cut_off_energy": 900.0},
    user_cell_settings={
        "species_pot": [
            ("O", "2|1.1|17|20|23|20:21(qc=8)"),
            ("Si", "3|1.8|5|6|7|30:31:32"),
        ],
        "kpoint_mp_spacing": 0.05,
    },
)
```

All allowed CASTEP keywords and definitions for setting up `.param` and `.cell` files can be found in the autoplex package at: `/yourpath/autoplex/misc/castep/castep_keywords.json`.

`use_efg` determines whether EFG tensors are calculated as well, if set to `False` it only calculates MS tensors.

Next, using `input_set_generator` we create a {class}`~autoplex.misc.castep.jobs.CastepMagresMaker`:

```python
from autoplex.misc.castep.jobs import CastepMagresMaker

magres_maker = CastepMagresMaker(
    input_set_generator=input_set_generator
)
```

Then we load our structures from e.g. `structures.xyz`:

```python
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read

structures = [AseAtomsAdaptor.get_structure(structure) for structure in read("structures.xyz", ":")]
```

This creates an array of pymatgen `Structure` objects which can then be passed into {class}`~autoplex.misc.castep.flows.CastepMagresFlowMaker`, which creates the `Flow` to be run locally or submitted to HPC (see the [jobflow-remote setup](../../../jobflowremote.md)):

```python
from jobflow import run_locally

from autoplex.misc.castep.flows import CastepMagresFlowMaker

flow = CastepMagresFlowMaker(magres_maker=magres_maker).make(structures)
run_locally(flow, create_folders=True)  # or jobflow-remote's submit_flow
```

Each job returns a {class}`~autoplex.misc.castep.schema.TaskDoc`. The tensors are in `output.ms_tensor` (ppm) and `output.efg_tensor` (atomic units), with one 3×3 tensor per atom, ordered like `output.structure`. This may differ from the input order, because CASTEP groups atoms by element. The compressed `castep.castep.gz` and `castep.magres.gz` files stay in the job's `CASTEP/` folder.

## Next steps

The core workflow should be ideally extended to automatically train ML models for NMR prediction ([Ben Mahmoud et al, J. Chem. Phys. 163, 024118 (2025)](https://doi.org/10.1063/5.0274240)) in a similar workflow to {class}`~autoplex.auto.rss.flows.RssMaker` (see the {ref}`RSS workflow <rss>`). The outputted `.magres` files could be used in the same way for labelling the dataset, however there will be differences to `RssMaker` in data processing, generation and sampling as similiar energy structures can have wildly different NMR parameters.



