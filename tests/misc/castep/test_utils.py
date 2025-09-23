import gzip
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.utils import CastepStaticSetGenerator, gzip_castep_outputs


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


def test_gzip_castep_outputs(tmp_path):
    test_file = tmp_path / "test.castep"
    with open(test_file, "w") as f:
        f.write("dummy CASTEP output")

    gzip_castep_outputs(workdir=tmp_path)

    gz_path = tmp_path / "test.castep.gz"

    assert gz_path.exists()

    with gzip.open(gz_path, "rt") as f:
        content = f.read()
    assert "dummy CASTEP output" in content
