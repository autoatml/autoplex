from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.utils import CastepStaticSetGenerator, CastepMagresSetGenerator
from ase.build import bulk
    
def test_CastepStaticSetGenerator():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    gen = CastepStaticSetGenerator(
        user_param_settings={"cut_off_energy": 320.0, "xc_functional": "PBE"},
        user_cell_settings={"kpoint_mp_grid": "1 1 1"},
    )

    input_set = gen.get_input_set(structure=structure)

    assert input_set["param"]["cut_off_energy"] == 320.0
    assert input_set["param"]["xc_functional"] == "PBE"
    assert input_set["param"]["task"] == "SinglePoint"

    assert "kpoint_mp_grid" in input_set["cell"]
    assert "kpoints_mp_spacing" in input_set["cell"]

    assert input_set["structure"].composition.formula == "Si1"


def test_CastepMagresSetGenerator():
    """
    example input taken from https://castep-docs.github.io/castep-docs/tutorials/NMR/Example_2_-Diamond/
    """
    atoms = bulk("C", "diamond", a=3.567)
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)


    gen = CastepMagresSetGenerator(
        user_param_settings={
            "xc_functional": "LDA",
            "fix_occupancy": True,
            "opt_strategy": "speed",
            "cut_off_energy":  600.0
            },
        user_cell_settings={},
    )

    input_set = gen.get_input_set(structure=pmg_structure)

    assert input_set["param"]["cut_off_energy"] == 600.0
    assert input_set["param"]["xc_functional"] == "LDA"
    assert input_set["param"]["task"] == "magres"
    assert input_set["param"]["magres_task"] == "NMR"

    assert input_set["structure"].composition.formula == "C2"