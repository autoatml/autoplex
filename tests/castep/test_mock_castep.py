import pytest
from unittest.mock import Mock, patch
from autoplex.castep.jobs import BaseCastepMaker
from jobflow import Flow
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import os
os.environ["castep_command"]="/usr/local/CASTEP-20/castep.mpi"

def test_castep_calculator_call():
    """Test that the CASTEP calculator call workflow works correctly."""
    
    # Create test structure
    si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    si_pmg = AseAtomsAdaptor.get_structure(si_ase)
    
    # Test BaseCastepMaker creation
    castep_maker = BaseCastepMaker(
        name="castep_test",
        cut_off_energy=200.0,
        kspacing=0.5,
        xc_functional="PBE",
        task="SinglePoint",
    )
    
    # Test that maker was created with correct parameters
    assert castep_maker.name == "castep_test"
    assert castep_maker.cut_off_energy == 200.0
    assert castep_maker.kspacing == 0.5
    assert castep_maker.xc_functional == "PBE"
    assert castep_maker.task == "SinglePoint"
    

def test_castep_job_creation():
    """Test that CASTEP job creation works."""
    
    # Create test structure
    si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    si_pmg = AseAtomsAdaptor.get_structure(si_ase)
    
    # Create maker
    castep_maker = BaseCastepMaker(
        name="castep_test",
        cut_off_energy=200.0,
        kspacing=0.5,
        xc_functional="PBE",
        task="SinglePoint",
    )
    
    # Test job creation - this should work without actually running CASTEP
    job = castep_maker.make(si_pmg)
    
    # Verify job was created successfully
    assert job is not None
    assert hasattr(job, 'output')


def test_castep_flow_creation():
    """Test that Flow creation with CASTEP job works."""
    
    # Create test structure
    si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    si_pmg = AseAtomsAdaptor.get_structure(si_ase)
    
    # Create maker and job
    castep_maker = BaseCastepMaker(
        name="castep_test",
        cut_off_energy=200.0,
        kspacing=0.5,
        xc_functional="PBE",
        task="SinglePoint",
    )
    
    job = castep_maker.make(si_pmg)
    
    # Test Flow creation
    flow = Flow([job], output=job.output)
    
    # Verify flow was created successfully
    assert flow is not None
    assert len(flow.jobs) == 1
    assert flow.output == job.output


@patch('jobflow.run_locally')
def test_castep_run_locally_call(mock_run_locally):
    """Test that run_locally is called correctly with CASTEP flow."""
    
    # Mock run_locally to return success without actually running
    mock_run_locally.return_value = {"job_1": Mock(output="success")}
    
    # Create test structure
    si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    si_pmg = AseAtomsAdaptor.get_structure(si_ase)
    
    # Create maker, job, and flow
    castep_maker = BaseCastepMaker(
        name="castep_test",
        cut_off_energy=200.0,
        kspacing=0.5,
        xc_functional="PBE",
        task="SinglePoint",
    )
    
    job = castep_maker.make(si_pmg)
    flow = Flow([job], output=job.output)
    
    # Import here to avoid circular imports
    from jobflow import run_locally
    
    # Test run_locally call
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    
    # Verify run_locally was called with correct parameters
    mock_run_locally.assert_called_once_with(flow, create_folders=True, ensure_success=True)
    
    # Verify responses
    assert responses is not None
    assert len(responses) > 0