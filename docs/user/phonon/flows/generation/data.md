(generation)=

*Tutorials written by Christina Ertural ([christina.ertural@bam.de](mailto:christina.ertural@bam.de)).*

# Generating reference data

This tutorial will explain how the reference data for the MLIP fit can be generated.
The idea behind the VASP reference data generation stems from [J. Chem. Phys. 153, 044104 (2020)](https://pubs.aip.org/aip/jcp/article/153/4/044104/1056348/Combining-phonon-accuracy-with-high) 
and demonstrates that a robust database of **crystalline structures** for MLIP reproducing accurate phonon structures can 
be build by generating the single-atom displaced supercells using `phonopy` and combining them with a set of rattled supercell 
structures, generated from the same unit cell models.

## Single-atom displaced supercell structures

The single-atom displaced supercell structures used in `autoplex` are generated by [phonopy](https://phonopy.github.io/phonopy/vasp.html) VASP-related routines,
that are collected in the `dft_phonopy_gen_data` flow 
(see diagram in the [general tutorial](../flows.md#general-workflow). 
The displacement default is 0.1 Å and can be adjusted. A `phonopy` calculation with a displacement of 0.1 Å is 
automatically included in the workflow for calculating the DFT benchmark reference 
(if no reference is provided by the user). The `min_length` parameter controls the supercells size and is set to 20 
per default as this value is good for ensuring that the periodic boundary conditions and the energy convergence criteria 
for phonon calculations are met. More settings can be found in the [API reference](#autoplex.auto.phonons.flows.CompleteDFTvsMLBenchmarkWorkflow).

There is the possibility to run the complete `autoplex` workflow using only `phonopy` generated supercells:
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(add_dft_random_struct=False, min_length=20,
                                                 displacements=displacement_list).make(
    structure_list=structure_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark)
```
By doing so, the generation of the randomized structures has to be tuned off by setting the `add_dft_random_struct` bool to `False`.
You can also decide if you want to only use those single-atom displaced supercells only for the MLIP fit, or if the data 
shall be added to an existing database. 

Adding data to an existing database is achieved by:
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(add_dft_random_struct=False).make(
    structure_list=structure_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark,
    pre_xyz_files=["vasp_ref.extxyz"], pre_database_dir=path/to/database)
```
Where `pre_xyz_files` can also take a train and test database as argument, e.g. as 
`pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"]`.

`autoplex` is equipped with a [DFTPhononMaker](#autoplex.data.phonons.flows.DFTPhononMaker) class that inherits from the `atomate2` [PhononMaker](https://materialsproject.github.io/atomate2/reference/atomate2.vasp.flows.phonons.PhononMaker.html#atomate2.vasp.flows.phonons.PhononMaker) with 
specific VASP input adjustments to guarantee high quality fit data. It can be used to run individual and customized `phonopy` workflows to generate MLIP fit data.

## Rattled supercell structures

There are several ways available in `autoplex` to rattle supercell structures,
that are collected in the `dft_random_gen_data` flow 
(see diagram in the [general tutorial](../flows.md#general-workflow).
The size of the supercell is determined by the `supercell_matrix`,
and there is the option of volume distortion, angle distortion or a combination of both provided by `distort_type`.
The displacement of all atomic positions ("rattling") is controlled by the parameter `rattle_type`,
which uses the the `ase` [rattle](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.rattle) function
(using a normal distribution of a certain standard deviation to draw the displacement value)
by default and can be changed to Monte-Carlo rattling.
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(rattle_type=0, # 0 = standard ase.Atoms.rattle(stddev)
                                                 distort_type=0, # only volume distortion
                                                 supercell_matrix=[[3, 0, 0], [0, 3, 0]],
                                                 volume_scale_factor_range=[0.95, 1.05],
                                                 n_structures=20).make(
    structure_list=structure_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark)
```
The combination of parameters `volume_scale_factor_range` and `n_structures` will produce 21 
(20 volume distorted + the undistorted supercell) supercells with a volume range of 95 to 105% of the original supercell. 
Alternatively, the parameter `volume_custom_scale_factors` can be used to set specific scale factors. 
> ℹ️ It is important to note that by using `volume_custom_scale_factors` the parameter `n_structures` is ignored 
> and **only one** rattled supercell for each given factor is generated. If more supercells with the same volume scale 
> are needed, this can be achieved by e.g. 
> ```python
> scale_factors = [0.90, 0.95, 1.00, 1.05, 1.10]
> 
> complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
>     volume_custom_scale_factors=[val for val in scale_factors for _ in range(5)]).make(...)
>                                   # will repeat each scale factor five times
>```
> Explicitly specifying `volume_custom_scale_factors` is useful if you don’t want evenly spaced intervals between 
> scale factors as e.g., you want to sample around the minimum more closely.

More details and settings are given in the [API reference](#autoplex.auto.phonons.flows.CompleteDFTvsMLBenchmarkWorkflow).

Similar to the single-atom displaced supercells, you can run the complete `autoplex` workflow using only randomized 
structures by setting `add_dft_phonon_struct` to `False`.
```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(add_dft_phonon_struct=False).make(
    structure_list=structure_list, mp_ids=mpids, preprocessing_data=True,
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark)
```
It can also be used to extend an already existing database in the same way as demonstrated above.

As a counterpart to the `DFTPhononMaker` for generating data, `autoplex` includes a [RandomStructuresDataGenerator](#autoplex.data.phonons.flows.RandomStructuresDataGenerator) 
that can be used to construct customized randomized structures workflows.
`autoplex` provides a variety of [utility](#autoplex.data.common.utils) subroutines to further customize a workflow.

## Adjust supercell settings

You can adjust the supercell settings by passing a dictionary containing your specific supercell settings for each
MP-ID to `CompleteDFTvsMLBenchmarkWorkflow`, e.g. like:
```python
mp_id = "mp-22905"

supercell_settings = {
    mp_id: {
        "supercell_matrix": [[0, 2, 0], [0, 0, 2], [2, 0, 0]]
    },
    "min_length": 11,  
    "max_length": 25,
    "max_atoms": 200,
}

complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    ..., 
    supercell_settings=supercell_settings, 
    ...).make(...)
```
To keep the calculations consistent, this will adjust the settings of single-atom displaced and rattled supercells.

## VASP settings

This part will show you how you can adjust the different Makers for the VASP calculations in the workflow.

For the single-atom displaced as well as the rattled structures the `autoplex` [TightDFTStaticMaker](#autoplex.data.phonons.flows.TightDFTStaticMaker) is 
used to set up the VASP calculation input and settings. PBEsol is the default GGA functional. For the VASP calculation 
of the isolated atoms' energies, `autoplex` also provides its own [IsoAtomStaticMaker](#autoplex.data.phonons.flows.IsoAtomStaticMaker), 
which settings you can further adjust. 
For the VASP geometry relaxation and static calculations of the unit cells as prerequisite calculations for generating 
the single-atom displaced as well as the rattled supercells, 
we rely on the [atomate2](https://materialsproject.github.io/atomate2/user/codes/vasp.html#list-of-vasp-workflows) 
Makers `StaticMaker`, `TightRelaxMaker` in combination with the `StaticSetGenerator` VASP input set generator for this example.

```python
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow
from autoplex.data.phonons.flows import IsoAtomStaticMaker, TightDFTStaticMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator

example_input_set = StaticSetGenerator(  # you can also define multiple input sets
    user_kpoints_settings={"grid_density": 1},
    user_incar_settings={
        "ALGO": "Normal",
        "IBRION": -1,
        "ISPIN": 1,
        "ISMEAR": 0,
         ...,         # set all INCAR tags you need
        "SIGMA": 0.05,
        "GGA": "PE",  # switches to PBE
         ...},
)
static_isolated_atom_maker = IsoAtomStaticMaker(
    name="isolated_atom_maker",
    input_set_generator=example_input_set,
)
displacement_maker = TightDFTStaticMaker(
    name="displacement_maker",
    input_set_generator=example_input_set,
)
rattled_bulk_relax_maker = TightRelaxMaker(
    name="bulk_rattled_maker",
    input_set_generator=example_input_set,
)
phonon_bulk_relax_maker = TightRelaxMaker(
    name="bulk_phonon_maker",
    input_set_generator=example_input_set,
)
phonon_static_energy_maker = StaticMaker(
    name="phonon_static_energy_maker",
    input_set_generator=example_input_set,
)

complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    displacement_maker=displacement_maker,  # one displacement maker for rattled and single-atom displaced supercells to keep VASP settings consistent
    phonon_bulk_relax_maker=phonon_bulk_relax_maker,
    phonon_static_energy_maker=phonon_static_energy_maker,
    rattled_bulk_relax_maker=rattled_bulk_relax_maker,
    isolated_atom_maker=static_isolated_atom_maker,).make(...)
```

Note, that for consistency of job handling, `autoplex` internally will override the jobs names to the `autoplex` defaults:
```
INFO Started executing jobs locally
INFO Starting job - rattled supercells_mp-22905 
INFO Finished job - rattled supercells_mp-22905 
INFO Starting job - tight relax_mp-22905 
INFO Finished job - tight relax_mp-22905 
INFO Starting job - reduce_supercell_size_job_mp-22905 
INFO Finished job - reduce_supercell_size_job_mp-22905 
INFO Starting job - generate_randomized_structures_mp-22905 
INFO Finished job - generate_randomized_structures_mp-22905 
INFO Starting job - run_phonon_displacements_mp-22905 
INFO Finished job - run_phonon_displacements_mp-22905 
INFO Starting job - dft rattle static 1/12_mp-22905 
INFO Finished job - dft rattle static 1/12_mp-22905
INFO Starting job - dft rattle static 2/12_mp-22905 
INFO Finished job - dft rattle static 2/12_mp-22905 
...
INFO Starting job - single-atom displaced supercells_mp-22905 
INFO Finished job - single-atom displaced supercells_mp-22905 
INFO Starting job - tight relax_mp-22905
INFO Finished job - tight relax_mp-22905 
INFO Starting job - static_mp-22905 
INFO Finished job - static_mp-22905 
INFO Starting job - generate_phonon_displacements_mp-22905 
INFO Finished job - generate_phonon_displacements_mp-22905 
INFO Starting job - run_phonon_displacements_mp-22905 
INFO Finished job - run_phonon_displacements_mp-22905 
INFO Starting job - dft phonon static 1/2_mp-22905 
INFO Finished job - dft phonon static 1/2_mp-22905 
INFO Starting job - dft phonon static 2/2_mp-22905 
INFO Finished job - dft phonon static 2/2_mp-22905
...
INFO Starting job - write_benchmark_metrics
INFO Finished job - write_benchmark_metrics
INFO Finished executing jobs locally
```