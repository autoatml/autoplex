# ScaledMDMaker tutorial

This notebook demonstrates `ScaledMDMaker`. It runs the same MD on one or more structures with optionally:

- supercells
- volume-scaled copies
- a multi-stage temperature profile (for every structure and/or their scaled form provided)

Every MD run writes its own trajectory file (`MD.traj`).

The MD itself is done by atomate2: with `md_engine="ase"` (the default, and for now the only
option) you pass an atomate2 `ForceFieldMDMaker` with all MD settings as `md_maker`.
The temperature profile, supercell, and volume settings belong to `ScaledMDMaker`.


```python
# Imports - adapt as needed for your environment
from ase.build import bulk
from atomate2.forcefields.md import ForceFieldMDMaker
from jobflow import run_locally
from pymatgen.io.ase import AseAtomsAdaptor

from autoplex.data.md.flows import ScaledMDMaker
```

## NVT Simulation with temperature ramping

The following sets up the MD and runs it locally. The MD itself is atomate2's `ForceFieldMDMaker`, given as `md_maker` (with `md_engine="ase"`); it holds all MD settings. `ScaledMDMaker` adds a ramping protocol on top:
- `temperature_list`, lists the temperature stages during the simulation
- `eqm_step_list`, lists the number of steps run at each temperature (low here for tutorial purposes)
- `rate_list`, lists the MD steps per kelvin for ramping between consecutive temperatures of `temperature_list`. 0 means an instant change; with a 1 fs time step, 2 corresponds to 5·10^14 K/s and 100 to 10^13 K/s. Optionally, the ramping can also be controlled with `ramp_step_list`, which gives the number of MD steps for the ramp assuming linear scaling and overwrites `rate_list`. Here, `ramp_step_list=[0,200]` would produce the same ramping protocol as `rate_list=[0,2]`



```python
# NVT MD of the 8-atom cubic Si cell with the MACE-MPA-0 foundation model
atoms = bulk("Si", "diamond", a=5.43, cubic=True)
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

md_job = ScaledMDMaker(
    md_engine="ase",  # atomate2's ASE MD; other engines are not implemented yet
    md_maker=ForceFieldMDMaker(  # all MD settings, as in atomate2
        force_field_name="MLFF.MACE_MPA_0",
        calculator_kwargs={"device": "cpu"},
        time_step=1.0,
        traj_interval=1,
        ensemble="nvt",
        dynamics="langevin",
        traj_file="MD.traj",
        traj_file_fmt="ase",
        store_trajectory="partial",
    ),
    temperature_list=[300.0, 500.0, 400.0],  # the temperature profile stays on ScaledMDMaker
    eqm_step_list=[2, 10, 3],
    rate_list=[0, 2],  # MD steps per K: 0 jumps directly; 2 = 5·10^14 K/s at 1 fs -> 200 steps for 500 -> 400 K
).make(structure=pmg_structure)

run_locally(md_job, create_folders=False, ensure_success=True)

print("MD run finished")
```

### Temperature profile visualization
`ScaledMDMaker` plots the temperature schedule it built for the run above and saves it as `md_temperature_profile.png` in the current directory. With `create_folders=True` it would be in the respective directory.


```python
from IPython.display import Image

# ScaledMDMaker saved its temperature profile while setting up the NVT run above
Image("md_temperature_profile.png")
```

## NVT Simulation with scaled cells

`ScaledMDMaker` can also automatically scale the cell. For this, there are two options:

- `supercell_matrix` turns every input structure into a supercell. A diagonal matrix makes the
  cell bigger (fewer finite-size effects, smaller volume fluctuations in NPT); a non-diagonal
  matrix can also change its shape, e.g. from the primitive to the cubic cell.
- `volume_custom_scale_factors` makes one copy of every (super)cell per factor, with the volume
  scaled by that factor, e.g. to sample compressed and expanded configurations for training data.

The supercell is made first, then the volume scaling. `ScaledMDMaker` runs one MD per resulting
cell, so the number of runs is (number of input structures) × (number of scale factors).
Its output is a list of trajectory paths, one per run.

With more than one run, use `run_locally(..., create_folders=True)`, so that every run gets its
own folder. Otherwise all runs write `MD.traj` into the same folder and overwrite each other.

Here we demonstrate the usage of both, starting with volume scaling.

### Volume scaling
First, the cubic cell is scaled to the volume MACE-MPA-0 prefers. 
The original and the scaled cell are then run at 300 K and compared with an NPT run at the end.
To keep the tutorial fast, we only run a 100 fs NVT MD per cell.

The cubic cell above uses the experimental lattice constant (5.43 Å), which is slightly too small for
MACE-MPA-0. A cell relaxation with MACE (atomate2's `ForceFieldRelaxMaker`) gives the volume MACE
prefers; the ratio of the two volumes is the scale factor that brings the cell to this optimum.
We are now running simple MD simulations at 300 K.


```python
from atomate2.forcefields.jobs import ForceFieldRelaxMaker

relax_job = ForceFieldRelaxMaker(
    force_field_name="MLFF.MACE_MPA_0",
    calculator_kwargs={"device": "cpu"},
    relax_cell=True,
    relax_kwargs={"fmax": 0.001},  # tight, so that the cell stress converges too
).make(pmg_structure)

responses = run_locally(relax_job, create_folders=False, ensure_success=True)
relaxed_structure = responses[relax_job.uuid][1].output.structure

volume_factor = relaxed_structure.volume / pmg_structure.volume
print(f"relaxed with MACE-MPA-0: a = {relaxed_structure.lattice.a:.3f} Å, volume = {relaxed_structure.volume:.2f} Å³")
print(f"volume scale factor to the MACE optimum: {volume_factor:.4f}")
```

We will now run two NVT simulations (both 300 K, 100 fs): one at the volume that relates to the experimental finding and one at the MACE-MPA-0 relaxed volume. As this creates several (here: two) cells each with their own MD trajectory, we use `create_folders=True` and we will use a MemoryStore to access the simulations afterwards.


```python
import numpy as np
from ase.io import read
from jobflow import JobStore
from maggma.stores import MemoryStore

store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
# 2 NVT runs, one for each scaled cell (one unscaled and one at MACE optimum)
md_job_volumes = ScaledMDMaker(
    md_engine="ase",
    md_maker=ForceFieldMDMaker(
        force_field_name="MLFF.MACE_MPA_0",
        calculator_kwargs={"device": "cpu"},
        time_step=1.0,
        traj_interval=1,
        ensemble="nvt",
        dynamics="langevin",
        traj_file_fmt="ase",
        store_trajectory="partial",
    ),
    temperature_list=[300.0],
    eqm_step_list=[100], #only 100 fs for demonstration purpose
    rate_list=[],
    # run at one cell at experimental scaling and one scaled to the MACE optimum
    volume_custom_scale_factors=[1, volume_factor],
).make(structure=pmg_structure)

run_locally(md_job_volumes, store=store, create_folders=True, root_dir="md_volume_scaling", ensure_success=True)

for path in md_job_volumes.output.resolve(store):  # one trajectory per cell
    start = read(path, index=0)
    print(f"{len(start):3d} atoms   volume {start.get_volume():7.2f} Å³ angles {np.round(start.cell.angles(), 1)}")

print("MDs run finished")
```

### Supercells
`ScaledMDMaker` can create a supercell and then employ the volumetric scaling in one run.


```python
# short NVT settings for the examples below
short_nvt = ForceFieldMDMaker(
    force_field_name="MLFF.MACE_MPA_0",
    calculator_kwargs={"device": "cpu"},
    time_step=1.0,
    ensemble="nvt",
    dynamics="langevin",
)
```

We use again a MemoryStore to store the simulations. We could optionally also create a desired temperature profile as we did in the first section.


```python
# bigger cell: 2x1x1 supercell of the cubic cell
md_job_supercell = ScaledMDMaker(
    md_maker=short_nvt,
    temperature_list=[300.0],
    eqm_step_list=[5], #5 fs, very short for demonstration purpose
    supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    volume_custom_scale_factors=[1, volume_factor],
).make(structure=pmg_structure)
run_locally(md_job_supercell, store=store, create_folders=True, root_dir="md_cell_choices", ensure_success=True)

for path in md_job_supercell.output.resolve(store):
    start = read(path, index=0)
    print(f"{len(start):3d} atoms   volume {start.get_volume():7.2f} Å³   angles {np.round(start.cell.angles(), 1)}")
```

Compared to the MDs in the previous section, we now doubled the cell size in x-direction by setting a `supercell_matrix`. The volume scaling stayed the same and is employed on the supercell structures.

## NPT Simulation

### MD at constant pressure (NPT)

The run above is NVT: the cell stays fixed. With `ensemble="npt"` a barostat lets
the cell change so that the pressure stays at the target value. NPT needs three
more settings of the `ForceFieldMDMaker`:

- `pressure`: target pressure in kbar (`0.0` is close to ambient; 1 bar = 0.001 kbar).
  A list of pressures is ramped evenly over the whole run.
- `dynamics`: "nose-hoover" (ASE's NPT: Nose-Hoover thermostat with Parrinello-Rahman barostat) works with every atomate2 version. With atomate2 ≥ 0.0.22, "nose-hoover-chain" (ASE MTKNPT, with ase_md_kwargs={"tdamp": ..., "pdamp": ...}) is the recommended choice.
- `ase_md_kwargs`: settings of the ASE integrator, in ASE units. For "nose-hoover": the thermostat time ttime (about 25 fs) and pfactor = (barostat time)² × bulk modulus, e.g. (75 fs)² × 100 GPa for Si.

The temperature profile (`temperature_list`, `eqm_step_list`, `rate_list`) works
the same as in NVT.


```python
from ase import units

md_job_npt = ScaledMDMaker(
    md_engine="ase",
    md_maker=ForceFieldMDMaker(
        force_field_name="MLFF.MACE_MPA_0",
        calculator_kwargs={"device": "cpu"},
        time_step=1.0,
        traj_interval=1,
        ensemble="npt",
        dynamics="nose-hoover",  # ASE NPT: works with every atomate2 version used in CI
        pressure=0.0,  # kbar
        ase_md_kwargs={
            "ttime": 25 * units.fs,  # thermostat time constant
            "pfactor": (75 * units.fs) ** 2 * 100 * units.GPa,  # (barostat time)^2 x bulk modulus of Si
        },
        traj_file="MD_npt.traj",  # own file, so the NVT trajectory MD.traj is kept
        traj_file_fmt="ase",
        store_trajectory="partial",
    ),
    temperature_list=[300.0],
    eqm_step_list=[100],
    rate_list=[],
).make(structure=pmg_structure)

response = run_locally(md_job_npt, create_folders=False, ensure_success=True)
# ScaledMDMaker replaced itself by atomate2's MD job; its response holds the final structure
md_run = response[md_job_npt.uuid][1].replace.jobs[0]
structure_npt = response[md_run.uuid][1].output.structure  # final structure of the NPT run

volume_npt = structure_npt.volume
print(f"NPT volume = {volume_npt:.2f} Å³ after 100 fs")
```

## Compare NVT, scaled Volume NVT, and NPT

We can now compare the volumes and pressures during the different simulations. We are comparing here only the small cell calculations which ran for 100 fs each.


```python
import matplotlib.pyplot as plt
from ase import units

time_step = 1.0  # fs, as in all three runs

runs = [
    (md_job_volumes.output.resolve(store)[0], "NVT, a = 5.43 Å", "#2a78d6"),
    (md_job_volumes.output.resolve(store)[1], "NVT, scaled to MACE optimum", "#1baf7a"),
    ("MD_npt.traj", "NPT", "#eb6834"),
]

fig, (ax_volume, ax_pressure) = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)
print(f"{'run':30s} {'mean volume (Å³)':>17s} {'mean pressure (GPa)':>20s}")
for traj_file, label, color in runs:
    frames = read(traj_file, index=":")
    volume = np.array([atoms.get_volume() for atoms in frames])
    pressure = np.array([-atoms.get_stress()[:3].mean() for atoms in frames]) / units.GPa
    time_ps = np.arange(len(frames)) * time_step / 1000
    ax_volume.plot(time_ps, volume, color=color, linewidth=1.5, label=label)
    ax_pressure.plot(time_ps, pressure, color=color, linewidth=1.5)
    print(f"{label:30s} {volume.mean():17.2f} {pressure.mean():20.2f}")

ax_volume.set_ylabel("Cell volume (Å$^3$)")
ax_pressure.set_ylabel("Pressure (GPa)")
ax_pressure.set_xlabel("Simulation time (ps)")
ax_pressure.axhline(0.0, color="0.6", linewidth=0.8)  # target pressure of the NPT run
for ax in (ax_volume, ax_pressure):
    ax.grid(visible=True, linestyle="--", linewidth=0.5)
ax_volume.legend(frameon=False, fontsize=8)
plt.show()
```

Note that for testing reasons the runs are very short (0.1 ps) and the cell has only 8 atoms, so
volume and pressure fluctuate strongly.

- **NVT at a = 5.43 Å** (blue): the experimental lattice constant is about 2 % too small in volume for
  MACE-MPA-0, so the cell stays under about +1 to +2.5 GPa for the whole run.
- **NVT in the scaled cell** (green): scaled to the volume of the MACE relaxation (a = 5.467 Å), the
  pressure fluctuates around zero, while the volume stays fixed.
- **NPT** (orange): the barostat expands the compressed cell and overshoots. After about 70 fs it passes the MACE volume and keeps growing, while its pressure falls from about +2 to about −1 GPa; in a longer run the volume would swing back and oscillate around the MACE volume.

For production runs, start from a relaxed (or scaled) cell, use a larger supercell (see
“Supercells” above), and run several ps before averaging.
