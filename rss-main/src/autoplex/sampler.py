from jobflow import job, Maker, Response, Flow
from dataclasses import dataclass
from pymatgen.core.structure import Structure
import os
from autoplex.utilities import extract_gap_label
from autoplex.rss import minimize_structures


@dataclass
class do_rss(Maker):

    """
    Maker to do rss
    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mlip_type: str 
        Choose one specific MLIP type: 
        'GAP' | 'SNAP' | 'ACE' | 'Nequip' | 'Allegro' | 'MACE'
    HPO: bool
        call hyperparameter optimization (HPO) or not

    """
    name: str = 'do_rss'
    mlip_type: str = None
    iteration_index: str = None

    @job
    def make(self, mlip_path: str, structure: list[Structure]):
        
        output = minimize_structures(mlip_path=mlip_path,
                                     index=self.iteration_index,
                                     input_structure=structure,
                                     output_file_name='RSS_relax_results',
                                     mlip_type=self.mlip_type,
                                     scalar_pressure_method ='exp',
                                     scalar_exp_pressure=100,
                                     scalar_pressure_exponential_width=0.2,
                                     scalar_pressure_low=0,
                                     scalar_pressure_high=50,
                                     max_steps=1000,
                                     force_tol=0.01,
                                     stress_tol=0.01,
                                     Hookean_repul=False,
                                     hookean_paras={(14, 14): (1000, 2.5)},
                                     write_traj=True
                                    )
            
        return output
    
