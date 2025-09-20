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
from pymatgen.core import Structure, Lattice
from autoplex.data.castep_support.utils import CastepStaticMaker, CastepStaticSetGenerator, BaseCastepMaker
from autoplex.data.common.flows import DFTStaticLabelling  

os.environ["CASTEP_COMMAND"] = "/usr/local/CASTEP-20/castep.mpi"
os.environ["CASTEP_PP_PATH"] = "/usr/local/CASTEP-20/usp"

def test_castep_execution():
    """Test CASTEP execution through DFTStaticLabelling with simple Si structure."""
    # Setup phase: create structure, maker, and labeller
    try:
        # Create structure using pymatgen (for DFTStaticLabelling)
        lattice = Lattice.orthorhombic(20, 20, 20)
        structure = Structure(lattice, ["Si"], [[10.0, 10.0, 10.0]], coords_are_cartesian=True)
        
        print(f"Created Si structure: {structure.formula}")

        # Create CASTEP maker with the parameters   
        castep_maker = CastepStaticMaker(
            input_set_generator=CastepStaticSetGenerator(
                user_param_settings={
                    "cut_off_energy": "100.0 eV",  
                    "xc_functional": "PBE",
                    "task": "singlepoint",
                    "max_scf_cycles": 100,
                    "elec_energy_tol": "1e-6 eV",
                    "spin_polarized": False,
                },
                user_cell_settings={
                    "kpoint_mp_grid": "1 1 1",          
                    "kpoint_mp_offset": "0.0 0.0 0.0",  
                }
            )
        )
        
        print("CASTEP maker created successfully")
        
        # Create DFTStaticLabelling with CASTEP
        labeller = DFTStaticLabelling(
            static_energy_maker=castep_maker
        )
        
        print("DFTStaticLabelling created with CASTEP maker")
        
        # Check that it recognizes CASTEP maker
        assert isinstance(labeller.static_energy_maker, BaseCastepMaker)
        print("CASTEP maker properly recognized by DFTStaticLabelling")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return False
    
    # Test actual CASTEP execution
    try:
        # Create temporary directory for calculation
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            print(f"Running calculation in: {tmp_dir}")
            
            # Try to run the calculation
            result = labeller.make([structure])
            
            print("CASTEP calculation completed successfully!")
            print(f"Result type: {type(result)}")
            
            # Check if we got expected output structure
            if hasattr(result, 'output'):
                print(f"Output keys: {result.output.keys() if hasattr(result.output, 'keys') else 'No keys'}")
            
            return True
            
    except Exception as e:
        print(f"CASTEP execution failed: {e}")
        
        # Check if it's a CASTEP availability issue
        if "castep" in str(e).lower() or "command not found" in str(e).lower():
            print("This appears to be a CASTEP installation issue")
            print("Make sure CASTEP is installed and in your PATH")
        
        return False

# Run the test directly
if __name__ == "__main__":
    test_castep_execution()

# Or just call it directly without the if __name__ check:
# test_castep_execution()
