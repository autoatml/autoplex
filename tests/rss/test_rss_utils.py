import os 
os.environ["OMP_NUM_THREADS"] = "1"

from quippy.potential import Potential
from ase.io import read
import numpy as np
import pytest
from autoplex.data.common.utils import (cur_select, 
                                        boltzhist_cur_one_shot, 
                                        ElementCollection, 
                                        boltzhist_cur_dual_iter)
from autoplex.data.rss.utils import CustomPotential, extract_pairstyle
from autoplex.fitting.common.utils import extract_gap_label
from autoplex.data.rss.utils import HookeanRepulsion, process_rss, minimize_structures
from pymatgen.io.ase import AseAtomsAdaptor


def test_hookean(test_dir, memory_jobstore):
    test_files_dir = test_dir / "data/rss.extxyz"
    mlip_path = test_dir / "fitting/GAP"
    gap_label = os.path.join(mlip_path, "gap_file.xml")
    gap_control = "Potential xml_label=" + extract_gap_label(gap_label)
    pot = Potential(args_str=gap_control, param_filename=gap_label)
    
    atoms = read(test_files_dir, index="0")
    atoms.calc = pot
    
    hk = HookeanRepulsion(0, 4, 100, 2.5)
    f = atoms.get_forces()
    atoms.set_constraint(hk)
    f_constrained = atoms.get_forces()
    
    assert np.all(
        np.isclose(f[0] -  f_constrained[0], 
                   np.array([-0.62623775,  3.50041634,  7.94378925]))
    )