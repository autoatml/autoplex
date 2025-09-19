"""
Integration test for CASTEP through DFTlabelling workflow.
Tests that BaseCastepMaker can successfully run a CASTEP calculation.
"""

import os
import datetime
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import run_locally, Flow
from autoplex.castep.jobs import BaseCastepMaker

def setup_castep_environment():
    """Setup CASTEP environment variables and check prerequisites."""
    
    # CASTEP executable and pseudopotential paths
    castep_command = "/usr/local/CASTEP-20/castep.mpi"
    usp_dir = "/usr/local/CASTEP-20/usp"
    
    # Set environment variables
    os.environ["CASTEP_COMMAND"] = castep_command
    os.environ["CASTEP_PP_PATH"] = usp_dir
    
    # Verify CASTEP executable exists
    if not os.path.exists(castep_command):
        raise FileNotFoundError(f"CASTEP executable not found at {castep_command}")
    
    if not os.access(castep_command, os.X_OK):
        raise PermissionError(f"CASTEP executable is not executable at {castep_command}")
    
    # Check pseudopotential directory
    if not os.path.exists(usp_dir):
        print(f"⚠️  Warning: USP directory not found at {usp_dir}")
        print("   CASTEP may fail if pseudopotentials are not available")
    
    print(f"🔧 CASTEP executable: {castep_command}")
    print(f"🔧 Pseudopotential path: {usp_dir}")
    print()
    
    return castep_command, usp_dir


def test_dftlabelling_castep():
    """Integration test for CASTEP through DFTlabelling."""
    
    print("=" * 60)
    print("INTEGRATION TEST: CASTEP THROUGH DFTLABELLING")
    print("=" * 60)
    
    # Setup environment first
    try:
        castep_command, usp_dir = setup_castep_environment()
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        raise
    
    # Create test structure - Single Si atom (isolated atom calculation)
    si_structure = Atoms(
        "Si",  # Single Si atom, not Si2
        positions=[[10.0, 10.0, 10.0]],  # Center in the cell
        cell=[20, 20, 20],  # Larger cell for proper isolation
        pbc=True
    )
    
    print(f"🔬 Test Structure: {si_structure.get_chemical_formula()}")
    print(f"   Cell: {si_structure.cell.lengths()}")
    print(f"   Positions: {si_structure.positions.tolist()}")
    print()
    
    # Convert to pymatgen structure
    pmg_structure = AseAtomsAdaptor.get_structure(si_structure)
    print(f"✅ Converted to pymatgen structure: {pmg_structure.formula}")
    print()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dftlabelling_castep_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print()
    
    # Create CASTEP maker with appropriate parameters for isolated atom
    print("🔧 Creating BaseCastepMaker...")
    castep_maker = BaseCastepMaker(
        name="integration_test_castep",
        castep_kwargs= {'cut_off_energy': 200.0, 
                        # 'kspacing': 100.0, 
                        'xc_functional': 'PBE',
                        'task': 'SinglePoint'}

    )
    
    print(f"   Parameters:")
    print(f"   - Name: {castep_maker.name}")
    print()
    
    # Create job and flow
    print("⚙️  Creating job and flow...")
    try:
        job = castep_maker.make(pmg_structure)
        flow = Flow([job], output=job.output)
        
        print(f"✅ Job created successfully")
        print(f"✅ Flow created with {len(flow.jobs)} job(s)")
        print()
        
    except Exception as e:
        print(f"❌ Failed to create job/flow: {e}")
        raise
    
    # Change to output directory and run
    original_dir = os.getcwd()
    
    try:
        os.chdir(output_dir)
        print(f"🚀 Running CASTEP calculation in: {os.getcwd()}")
        print("   This may take a few minutes...")
        print()
        
        # Run the workflow
        responses = run_locally(
            flow,
            create_folders=True,
            ensure_success=True,  # Will raise exception if calculation fails
            root_dir="."
        )
        
        print("✅ CASTEP calculation completed successfully!")
        print(f"📊 Job responses: {len(responses)} job(s) executed")
        print()
        
        # Analyze results
        print("📋 ANALYSIS OF RESULTS:")
        print("-" * 30)
        
        for job_id, result in responses.items():
            print(f"Job ID: {job_id}")
            print(f"Result type: {type(result)}")
            
            # Handle different result types
            if hasattr(result, 'output'):
                result_data = result.output
                print(f"Output type: {type(result_data)}")
            else:
                result_data = result
            
            # Try to extract energy information
            if isinstance(result_data, dict):
                # Look for energy-related keys
                energy_keys = [k for k in result_data.keys() if 'energy' in str(k).lower()]
                print(f"Energy-related keys: {energy_keys}")
                
                # Display energy values
                for key in energy_keys:
                    try:
                        energy_value = result_data[key]
                        print(f"🎯 {key}: {energy_value}")
                    except:
                        print(f"🎯 {key}: <could not display>")
                
                # Look for structure information
                structure_keys = [k for k in result_data.keys() if 'structure' in str(k).lower()]
                if structure_keys:
                    print(f"Structure keys: {structure_keys}")
                
                # Look for task completion
                if 'task_completed' in result_data:
                    print(f"🔄 Task completed: {result_data['task_completed']}")
                
            print()
        
        # List output files with CASTEP-specific files
        print("📄 OUTPUT FILES:")
        print("-" * 20)
        
        def list_directory_contents(path, prefix=""):
            """Recursively list directory contents, highlighting CASTEP files."""
            items = []
            try:
                for item in sorted(os.listdir(path)):
                    item_path = os.path.join(path, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        # Highlight important CASTEP files
                        if item.endswith(('.castep', '.cell', '.param', '.geom', '.bands', '.den_fmt')):
                            items.append(f"{prefix}⭐ {item} ({size} bytes) [CASTEP output]")
                        else:
                            items.append(f"{prefix}📄 {item} ({size} bytes)")
                    elif os.path.isdir(item_path):
                        items.append(f"{prefix}📁 {item}/")
                        # Recursively list subdirectory (limit depth)
                        if prefix.count("  ") < 3:
                            items.extend(list_directory_contents(item_path, prefix + "  "))
            except PermissionError:
                items.append(f"{prefix}❌ Permission denied")
            return items
        
        file_list = list_directory_contents(".")
        for item in file_list:
            print(item)
        
        # Try to read basic information from CASTEP output
        print()
        print("🔍 CASTEP OUTPUT ANALYSIS:")
        print("-" * 30)
        
        # Look for .castep output file
        castep_files = [f for f in os.listdir(".") if f.endswith('.castep')]
        if castep_files:
            castep_file = castep_files[0]
            print(f"📖 Reading CASTEP output: {castep_file}")
            try:
                with open(castep_file, 'r') as f:
                    content = f.read()
                    
                # Look for final energy
                if "Final energy =" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if "Final energy =" in line:
                            print(f"🎯 {line.strip()}")
                            break
                
                # Look for convergence
                if "Total time =" in content:
                    print("✅ Calculation completed normally")
                else:
                    print("⚠️  Calculation may not have completed normally")
                    
            except Exception as e:
                print(f"❌ Could not read CASTEP output: {e}")
        else:
            print("⚠️  No .castep output file found")
        
        print()
        print("🎉 INTEGRATION TEST PASSED!")
        print(f"📁 All files saved in: {os.path.abspath('.')}")
        
        return True
        
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED!")
        print(f"Error: {e}")
        print()
        
        # Still show what files were created
        print("📄 Files created before failure:")
        try:
            for item in os.listdir("."):
                if os.path.isfile(item):
                    size = os.path.getsize(item)
                    if item.endswith('.castep'):
                        print(f"  ⭐ {item} ({size} bytes) [CASTEP output - check for errors]")
                    else:
                        print(f"  📄 {item} ({size} bytes)")
                elif os.path.isdir(item):
                    print(f"  📁 {item}/")
        except:
            print("  (Could not list files)")
        
        raise e
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    print("🔬 CASTEP Integration Test")
    print("=" * 30)
    print()
    
    try:
        success = test_dftlabelling_castep()
        if success:
            print("✅ All integration tests completed successfully!")
        else:
            print("❌ Integration test failed")
            exit(1)
            
    except Exception as e:
        print(f"💥 Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)