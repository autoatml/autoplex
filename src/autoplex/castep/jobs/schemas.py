from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from emmet.core.tasks import TaskDoc

class CastepTaskDoc(TaskDoc):
    """
    Task document for CASTEP calculations.
    Simplified version that avoids VASP-specific validation.
    """
    task_label: str = Field(description="Human readable task label")
    dir_name: str = Field(description="Directory where the calculation was run") 
    
    # Input data
    structure: Structure = Field(description="Input structure")
    
    # Output data  
    final_structure: Structure = Field(description="Final structure")
    energy: float = Field(description="Final energy in eV")
    energy_per_atom: float = Field(description="Final energy per atom in eV")
    forces: Optional[list] = Field(None, description="Forces on atoms")
    stress: Optional[list] = Field(None, description="Stress tensor")
    
    # Calculation metadata
    task_type: str = Field(default="Static", description="Type of calculation")
    calc_type: str = Field(default="CASTEP Static", description="Calculator and task type")
    completed_at: datetime = Field(description="When calculation completed")
    elapsed_time: Optional[float] = Field(None, description="Runtime in seconds")
    
    # CASTEP-specific parameters
    castep_version: Optional[str] = Field(None, description="CASTEP version")
    cutoff_energy: Optional[float] = Field(None, description="Plane wave cutoff in eV")
    kpoint_spacing: Optional[float] = Field(None, description="K-point spacing")
    xc_functional: Optional[str] = Field(None, description="Exchange-correlation functional")
    
    # Success/failure status
    converged: bool = Field(default=True, description="Whether calculation converged")
    warnings: Optional[list] = Field(None, description="Any warnings from calculation")
    
    class Config:
        arbitrary_types_allowed = True
        
    def model_post_init(self, __context):
        """Override emmet's post_init to avoid VASP validation"""
        pass  # Skip parent validation