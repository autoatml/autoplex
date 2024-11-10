import os 
os.environ["OMP_NUM_THREADS"] = "1"
from ase.io import read
import numpy as np

def test_custom_potential(test_dir, memory_jobstore, clean_dir):
    from autoplex.data.rss.utils import CustomPotential
    from autoplex.fitting.common.utils import extract_gap_label

    mlip_path = test_dir / "fitting/GAP"
    gap_label = os.path.join(mlip_path, "gap_file.xml")
    gap_control = "Potential xml_label=" + extract_gap_label(gap_label)
    pot = CustomPotential(args_str=gap_control, param_filename=gap_label)
    atom = read(f'{test_dir}/data/rss.extxyz')
    atom.calc = pot
    assert (atom.get_forces()).shape == (6, 3)


def test_extract_pairstyle_of_ace(test_dir, memory_jobstore, clean_dir):
    from autoplex.data.rss.utils import extract_pairstyle

    ace_json_path = test_dir / "fitting/JACE/acemodel.json"
    ace_table_path = test_dir / "fitting/JACE/acemodel_pairpot.table" 

    ace_label = "ace_potential.yace"
    atom_types, cmds = extract_pairstyle(ace_label, ace_json_path, ace_table_path)
    print(atom_types, cmds)

    assert atom_types == {'Si': 1}
    assert cmds == [
        'pair_style     hybrid/overlay pace table spline 2000', 
        'pair_coeff     * * pace ace_potential.yace Si', 
        f'pair_coeff     1 1 table {test_dir}/fitting/JACE/acemodel_pairpot.table Si_Si'
    ]


def test_process_rss(test_dir, memory_jobstore, clean_dir):
    from autoplex.data.rss.utils import process_rss

    np.random.seed(42)
    mlip_path = test_dir / "fitting/GAP"
    test_files_dir = test_dir / "data/rss.extxyz"
    atom = read(test_files_dir)

    traj_path = process_rss(atom=atom,
                            mlip_type="GAP",
                            mlip_path=mlip_path,
                            output_file_name="RSS_relax_results",
                            scalar_pressure_method= "exp",
                            scalar_exp_pressure= 1,
                            scalar_pressure_exponential_width= 0.2,
                            scalar_pressure_low= 0,
                            scalar_pressure_high= 50,
                            max_steps=100,
                            force_tol=0.05,
                            stress_tol=0.05,
                            hookean_repul=False,
                            hookean_paras=None,
                            write_traj= True,
                            device="cpu",
                            isolated_atom_energies=None,
                            config_type="traj",
                            keep_symmetry=True,
                            )

    atom_relax = read(traj_path)

    assert round(atom_relax.get_potential_energy(), 4) == -32.1410


def test_minimize_structures(test_dir, memory_jobstore, clean_dir):
    from autoplex.data.rss.utils import minimize_structures
    from pymatgen.io.ase import AseAtomsAdaptor

    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:2:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    traj_paths = minimize_structures(mlip_type='GAP',
                                    iteration_index='0',
                                    mlip_path=mlip_path,
                                    structures=structures,
                                    scalar_pressure_method='exp',
                                    scalar_exp_pressure=100,
                                    scalar_pressure_exponential_width=0.2,
                                    scalar_pressure_low=0,
                                    scalar_pressure_high=50,
                                    max_steps=10,
                                    force_tol=0.1,
                                    stress_tol=0.1,
                                    hookean_repul=False,
                                    write_traj=True,
                                    num_processes_rss=4,
                                    device="cpu",
                                    isolated_atom_energies={14: -0.84696938})
    
    assert len(traj_paths) == 2
    