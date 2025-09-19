#!/usr/bin/env python3
"""
Simple unit test to compare ASE CASTEP vs BaseCastepMaker energies
"""

import os
import tempfile
from ase import Atoms
from ase.calculators.castep import Castep
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import run_locally, Flow
from autoplex.castep.jobs import BaseCastepMaker


def setup_environment():
    """Setup CASTEP environment"""
    os.environ["CASTEP_COMMAND"] = "/usr/local/CASTEP-20/castep.mpi"
    os.environ["CASTEP_PP_PATH"] = "/usr/local/CASTEP-20/usp"


def get_ase_energy():
    """Get energy using ASE CASTEP calculator"""
    # Create structure
    atoms = Atoms('Si', positions=[[10.0, 10.0, 10.0]], cell=[20, 20, 20], pbc=True)
    
    # Setup ASE CASTEP calculator
    calc = Castep()
    calc.param.cut_off_energy = 200.0
    calc.param.xc_functional = 'PBE'
    calc.param.task = 'singlepoint'
    calc.param.max_scf_cycles = 100
    calc.param.elec_energy_tol = 1e-6
    calc.cell.kpoint_mp_grid = '1 1 1'
    calc.cell.kpoint_mp_offset = '0.0 0.0 0.0'
    calc.param.spin_polarized = False
    
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return energy


def get_basecastepmaker_energy():
    """Get energy using BaseCastepMaker"""
    # Create structure
    atoms = Atoms('Si', positions=[[10.0, 10.0, 10.0]], cell=[20, 20, 20], pbc=True)
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)
    
    # Setup BaseCastepMaker
    castep_maker = BaseCastepMaker(
        name="test_castep",
        castep_kwargs={
            'cut_off_energy': 200.0,
            'xc_functional': 'PBE',
            'task': 'SinglePoint'
        }
    )
    
    # Run calculation in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            job = castep_maker.make(pmg_structure)
            flow = Flow([job], output=job.output)
            responses = run_locally(flow, create_folders=True, ensure_success=True)
            
            # Extract energy from results
            result = list(responses.values())[0]
            if hasattr(result, 'output'):
                result_data = result.output
            else:
                result_data = result
            
            # Find energy in the result
            if isinstance(result_data, dict):
                for key, value in result_data.items():
                    if 'energy' in str(key).lower():
                        return float(value)
            
            raise ValueError("Could not find energy in BaseCastepMaker results")
            
        finally:
            os.chdir(original_dir)


def test_energy_comparison():
    """Test that both methods give the same energy"""
    setup_environment()
    
    print("Running ASE CASTEP calculation...")
    ase_energy = get_ase_energy()
    
    print("Running BaseCastepMaker calculation...")
    basemaker_energy = get_basecastepmaker_energy()
    
    # Print results
    print(f"\nResults:")
    print(f"ASE CASTEP energy:      {ase_energy:.6f} eV")
    print(f"BaseCastepMaker energy: {basemaker_energy:.6f} eV")
    print(f"Difference:             {abs(ase_energy - basemaker_energy):.6f} eV")
    
    # Assert they are equal within tolerance
    tolerance = 1e-4  # eV
    assert abs(ase_energy - basemaker_energy) < tolerance, \
        f"Energies differ by more than {tolerance} eV: ASE={ase_energy}, BaseCastepMaker={basemaker_energy}"
    
    print(f"\n✅ Test PASSED: Energies match within {tolerance} eV tolerance")
    return True


if __name__ == "__main__":
    test_energy_comparison()