# Core Workflow

This tutorial covers the core workflow in automating NMR predictions via CASTEP, with examples on using CASTEP to calculate NMR parameters of two small cristobalite structures as well as an amorphous silica snapshot.

## Overview

Given a list of [pymatgen](https://pymatgen.org/) structures (usually given in a `.xyz` file) and user-defined parameters, CASTEP is called with `task: magres` (see the [CASTEP setup](../../../rss/flow/input/input.md#labelling-parameters) for how autoplex runs CASTEP) and produces `.magres` and `.castep` files. Afterwards magnetic shielding (MS) and electric field gradient (EFG) tensors are collected for all structures by writing to a single `.extxyz` file.

## General workflow

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
from autoplex.misc.castep.flows import CastepMagresFlowMaker

flow = CastepMagresFlowMaker(magres_maker=magres_maker).make(structures)
```

Compressed `castep.castep.gz` and `castep.magres.gz` files are produced in the job's `CASTEP/` folder. `CastepMagresFlowMaker.make()` outputs a list of directories pointing at the `CASTEP/` folder. Next, we collect from these directories using {func}`~autoplex.data.nmr.jobs.collect_nmr_data`, which reads `castep.magres.gz` from each directory (skipping runs that did not converge) and writes all structures to one extended XYZ file, containing NMR-labelled data:

```python
from jobflow import Flow, run_locally
from autoplex.data.nmr.jobs import collect_nmr_data

collect = collect_nmr_data(nmr_dirs=flow.output, nmr_ref_file="nmr_ref.extxyz")
run_locally(Flow([flow, collect]), create_folders=True) # or jobflow-remote's submit_flow
```

Each atom carries its shielding and EFG tensors as the per-atom arrays `REF_ms` (ppm) and `REF_efg`
(atomic units), each stored as 9 components (row by row), since extended XYZ cannot store 3×3 tensors.
The file is written to the collector job's directory; its path is returned as `nmr_ref_dir`, ready for processing.

## Example systems

We ran the workflow on cristobalite and amorphous SiO<sub>2</sub> structures from the dataset of [Ben Mahmoud et al., J. Chem. Phys. 163, 024118 (2025)](https://doi.org/10.1063/5.0274240) ([Zenodo, CC BY 4.0](https://doi.org/10.5281/zenodo.15775328)). The amorphous SiO<sub>2</sub> snapshot is the first structure in `test.xyz` of that dataset, alpha and beta cristobalite structures are structures 1 and 3 in `crystals.xyz` of that dataset.

We used the settings shown above (PBE, 900 eV, shielding + EFG) with CASTEP 21.11 on 38–45 MPI processes on an HPC cluster, with additionally a fixed 3×3×3 k-point grid (`"kpoint_mp_grid": [3, 3, 3]` in `user_cell_settings`) for amorphous SiO<sub>2</sub>. The output and runtime of each structure is shown below:

| Structure | Atoms | Si σ<sub>iso</sub> (ppm) | O σ<sub>iso</sub> (ppm) | Runtime |
|---|---|---|---|---|
| α-cristobalite | 12 | 437.6 | 223.3 | ~5 min |
| β-cristobalite | 24 | 459.1 | 230.9 | ~15 min |
| Amorphous SiO<sub>2</sub> | 144 | 423.4 – 449.5 | 166.5 – 231.3 | ~42 h |

Here σ<sub>iso</sub> is the isotropic shielding in ppm, i.e. one third of the trace of the shielding tensor. It can be computed directly from the `.extxyz` file, using [`ase.io.read`](https://ase-lib.org/):

```python
import numpy as np
from ase.io import read

structures_out = read("nmr_ref.extxyz", index=":")
ms = structures_out[0].arrays["REF_ms"].reshape(-1, 3, 3)
#Take e.g. ms of first structure, convert back to 3x3 tensor. ["REF_efg"] gives efg tensors (if present).

sigma_iso = np.trace(ms, axis1=1, axis2=2) / 3  
#Compute isotropic shielding for each atom in structure.
#NOTE: Atoms are listed in CASTEP’s order (grouped by element), which may differ from the order of the input structure. 
```

To obtain other NMR parameters from a `.extxyz` file, follow the similar process above to obtain MS and EFG tensors. They can then be analysed with e.g. [Soprano](https://github.com/CCP-NC/soprano) ([docs](https://ccp-nc.github.io/soprano/)), a separate package that is not installed with autoplex. For large structures like amorphous SiO<sub>2</sub>, using fewer k-points can avoid large runtimes, at the cost of accuracy (using a single k-point reduces the runtime to ~1.5 hours in our case, with MAE in isotropic shielding ~0.05 vs ~0.007 ppm).

Example `.magres` files for the cristobalite structures are located in [tests/test_data/castep/magres](https://github.com/autoatml/autoplex/tree/main/tests/test_data/castep/magres), and a snippet of the `.magres` output (see the [magres file format](https://www.ccpnc.ac.uk/docs/magres)) for amorphous SiO<sub>2</sub> is provided below:
```
...
[magres_old]
============
Atom: O        1
============
O        1 Coordinates     10.723    1.886    8.916   A

TOTAL Shielding Tensor

             242.8870     46.2124     14.3341
              40.4691    230.3592      8.3922
              13.8238     10.3303    207.1829

O        1 Eigenvalue  sigma_xx     192.4360 (ppm)
O        1 Eigenvector sigma_xx       0.6695     -0.7202     -0.1820
O        1 Eigenvalue  sigma_yy     203.9185 (ppm)
O        1 Eigenvector sigma_yy       0.0379      0.2778     -0.9599
O        1 Eigenvalue  sigma_zz     284.0746 (ppm)
O        1 Eigenvector sigma_zz       0.7419      0.6357      0.2132

O        1 Isotropic:      226.8097 (ppm)
O        1 Anisotropy:      85.8973 (ppm)
O        1 Asymmetry:        0.2005

============
Atom: O        2
============
O        2 Coordinates      1.501    9.423    5.435   A

...
```

## Next steps

The core workflow can be extended to automatically train ML models for NMR prediction ([Ben Mahmoud et al, J. Chem. Phys. 163, 024118 (2025)](https://doi.org/10.1063/5.0274240)) in a similar workflow to {class}`~autoplex.auto.rss.flows.RssMaker` (see the {ref}`RSS workflow <rss>`). The outputted `.magres` files could be used in the same way for labelling the dataset, however there will be differences to `RssMaker` in data processing, generation and sampling as similar energy structures can have wildly different NMR parameters. This has not been implemented yet in autoplex.

## Further reading

- [CASTEP for NMR calculations (CCP-NC)](https://www.ccpnc.ac.uk/docs/castep-for-nmr-calculations)
- C. Bonhomme et al., "First-principles calculation of NMR parameters using the gauge including projector augmented wave method: a chemist's point of view", *Chem. Rev.* **112**, 5733–5779 (2012), [doi:10.1021/cr300108a](https://doi.org/10.1021/cr300108a)
- S. Sturniolo et al., "Visualization and processing of computed solid-state NMR parameters: MagresView and MagresPython", *Solid State Nucl. Magn. Reson.* **78**, 64–70 (2016), [doi:10.1016/j.ssnmr.2016.05.004](https://doi.org/10.1016/j.ssnmr.2016.05.004)
- C. Ben Mahmoud et al., *J. Chem. Phys.* **163**, 024118 (2025), [doi:10.1063/5.0274240](https://doi.org/10.1063/5.0274240)
