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
            
            # Debug: Print the structure of results
            print(f"DEBUG: Result type: {type(result_data)}")
            if isinstance(result_data, dict):
                print(f"DEBUG: Available keys: {list(result_data.keys())}")
                
                # If keys are [1], try to access the nested result
                if list(result_data.keys()) == [1]:
                    actual_result = result_data[1]
                    print(f"DEBUG: Nested result type: {type(actual_result)}")
                    if isinstance(actual_result, dict):
                        print(f"DEBUG: Nested result keys: {list(actual_result.keys())}")
                        if 'energy' in actual_result:
                            print(f"DEBUG: Found energy: {actual_result['energy']}")
                            return float(actual_result['energy'])
                
                # Try to find energy directly
                if 'energy' in result_data:
                    return float(result_data['energy'])
                
                # Look for energy-related keys
                energy_keys = [k for k in result_data.keys() if 'energy' in str(k).lower()]
                print(f"DEBUG: Energy-related keys: {energy_keys}")
                
                for key in energy_keys:
                    try:
                        return float(result_data[key])
                    except (ValueError, TypeError):
                        print(f"DEBUG: Non-numeric energy key '{key}': {result_data[key]}")
            
            raise ValueError(f"Could not find energy in BaseCastepMaker results. Available keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'Not a dict'}")
            
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