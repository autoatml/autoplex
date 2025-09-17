
from autoplex.castep.jobs.base import BaseCastepMaker
from jobflow import run_locally, Flow
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import os

def test_castep_with_base_maker():
    """
    Test CASTEP integration using BaseCastepMaker.
    Runs a single-point calculation on a silicon atom.
    """

    # Path to CASTEP
    castep_command = "/usr/local/CASTEP-20/castep.mpi"
    usp_dir = "/usr/local/CASTEP-20/usp"   # change if your usp files are elsewhere

    print(f"Checking CASTEP at {castep_command}")
    if not os.path.exists(castep_command):
        raise FileNotFoundError(f"CASTEP executable not found at {castep_command}")
    if not os.access(castep_command, os.X_OK):
        raise PermissionError(f"CASTEP executable is not executable at {castep_command}")

    print(f"✓ CASTEP found: {castep_command}")

    # Ensure USP pseudopotentials are accessible
    os.environ["CASTEP_PP_PATH"] = usp_dir
    print(f"✓ Using USP pseudopotentials from: {usp_dir}")

    # Define a simple system in ASE
    si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Convert ASE → Pymatgen
    si_pmg = AseAtomsAdaptor.get_structure(si_ase)

    # Create BaseCastepMaker 
    castep_maker = BaseCastepMaker(
        name="castep_test",
        castep_command=castep_command,
        cut_off_energy=200.0,
        kspacing=0.5,
        xc_functional="PBE",
        task="SinglePoint",
    )

    # Create the job with pymatgen Structure
    job = castep_maker.make(si_pmg)

    # Wrap in a Flow (jobflow convention)
    flow = Flow([job], output=job.output)

    print("Running CASTEP test job through BaseCastepMaker...")
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    print("✓ CASTEP BaseCastepMaker test completed successfully")
    print("Job outputs:", responses)

if __name__ == "__main__":
    test_castep_with_base_maker()
